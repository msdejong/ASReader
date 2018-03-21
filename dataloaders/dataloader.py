import glob
from data import Data
import os
import io
import random
try: 
    import cPickle as pickle
except:
    import pickle


class DataLoader():
    def __init__(self):
        pass


    # loads training data from directory
    def load_data(self, input_directory, max_number=0):

        data = []

        filenames = glob.glob(os.path.join(input_directory, '*.question'))
        number_files = len(filenames)
        if max_number!=0:
            number_files = min(max_number, number_files)

        for i in range(number_files):

            if i%20000==0 and i!=0:
                print("{} files loaded".format(i))

            with io.open(filenames[i], encoding="utf8", errors='replace') as file:

                lines = file.readlines()


                # document, query and answer are at fixed location in files
                document_tokens = lines[2].split()
                query_tokens = lines[4].split()
                answer_tokens = lines[6].split()



                entity_location_dictionary = dict()

                #files contain entries like "Entity538:David Beckham". We use only the former, so extract.
                for line in lines[8:]:

                    entity = line.strip('\n').split(":", 1)[0]
                    entity_location_dictionary[entity] = [int(token==entity) for token in document_tokens]

                # For each entity in the document, create a list that is equal to 1 for all the locations of the entity in the document and 0 for other locations. 
                # We make a list of these entity-lists, with the actual answer entity as the first element of that list 
                entity_locations =list()
                entity_locations.append(entity_location_dictionary[answer_tokens[0]])
                for entity in entity_location_dictionary:
                    if entity!=answer_tokens[0]:
                        entity_locations.append(entity_location_dictionary[entity])

                data_point = Data(document_tokens, query_tokens, answer_tokens, entity_locations)
                data.append(data_point)

        return data


    # def pickle_data(self, input_directory, output_path, max_number=0):
    #     data = self.load_data(input_directory, max_number=max_number)
    #     with open(output_path, "wb") as fout:
    #         pickle.dump(data, fout)

    # def load_pickle(self, input_path):
    #     with open(input_path, "r") as fin:
    #         data = pickle.load(fin)          
    #     return data


    # Create word_ids from tokens
    def generate_vocabulary(self, data, special_tokens=["<unk>"]):

        word_set = set()

        for data_point in data:
            tokens = data_point.document_tokens + data_point.query_tokens + data_point.answer_tokens
            for token in tokens:
                word_set.add(token)

        for token in special_tokens:
            word_set.add(token)

        vocab = {}

        for i, elem in enumerate(word_set):
            vocab[elem] = i

        return vocab

    def word_to_id(self, tokens, vocabulary):
        ids = []
        for token in tokens:
            if token in vocabulary:
                ids.append(vocabulary[token])
            else:
                ids.append(vocabulary["<unk>"])
        return ids

    # Replace tokens with word ids. 
    # Warning: this alters the data objects in place
    def process_data(self, data, vocabulary):
        for data_point in data:
            data_point.document_tokens = self.word_to_id(data_point.document_tokens, vocabulary)
            data_point.query_tokens = self.word_to_id(data_point.query_tokens, vocabulary)
            data_point.answer_tokens = self.word_to_id(data_point.answer_tokens, vocabulary)


    # This function basically randomizes a dictionary by shuffling the values while keeping the keys in place. 
    # The goal here is to replace each entity with a different entity. 
    def randomize_entities(self, entity_vocabulary):
        values = entity_vocabulary.values()
        random.shuffle(values)
        randomized_dictionary = dict(zip(entity_vocabulary.keys(), values))
        return randomized_dictionary


    # Replaces each entity with its randomized entity. Changes the actual data with side effects, but we rerandomize every epoch anyway. 
    def replace_entities(self, data_point, randomized_vocabulary):
        data_point.document_tokens = [randomized_vocabulary.get(
            data_point.document_tokens[i], data_point.document_tokens[i]) for i in range(len(data_point.document_tokens))]
        data_point.answer_tokens = [randomized_vocabulary.get(
            data_point.answer_tokens[i], data_point.answer_tokens[i]) for i in range(len(data_point.answer_tokens))]
        data_point.query_tokens = [randomized_vocabulary.get(
            data_point.query_tokens[i], data_point.query_tokens[i]) for i in range(len(data_point.query_tokens))]

    # Pad sequence. Note that this copies the list to avoid padding the data in place, because we call create_batches multiple times
    def pad_seq(self, seq, max_len, pad_token=0):
        seq = list(seq) + [pad_token for i in range(max_len - len(seq))]
        return seq


    # Function creates batches by shuffling the data, making buckets of the data, and within those buckets, creating batches by sorting by document length
    def create_batches(self, data, batch_size, bucket_size, vocabulary):

        # Search out all the words in the vocabulary that are entities, and create a vocabulary mapping their word ids to themselves.
        # Later these will be shuffled.
        entity_vocabulary = {vocabulary[word]: vocabulary[word] for word in vocabulary if "@ent" in word}

        # shuffle the actual data
        temp_data = list(data)
        random.shuffle(temp_data)

        batches = []
        data_per_bucket = batch_size * bucket_size

        # Calculate number of buckets
        number_buckets = len(data) // data_per_bucket + int((len(data) % data_per_bucket) > 0)

        def create_bucket(bucket_data):
            bucket = []
            document_lengths = [len(data_point.document_tokens) for data_point in bucket_data]

            # within batch, sort data by length
            sorted_data = list(zip(document_lengths, bucket_data))
            sorted_data.sort(reverse=True)

            document_lengths, bucket_data = zip(*sorted_data)

            # Calculate number of batches
            number_batches = len(bucket_data) // batch_size + \
                int((len(bucket_data) % batch_size) > 0)

            def create_batch(batch_data):

                batch_length = len(batch_data)

                # We need to randomly shuffle the word_ids for entities every batch. 
                #We create a new random mapping from entity word ids to entity word ids and replace the entities

                randomized_entities = self.randomize_entities(entity_vocabulary)
                for data_point in batch_data:
                    self.replace_entities(data_point, randomized_entities)

                batch_query_lengths = [len(data_point.query_tokens) for data_point in batch_data]

                # Sort by query length inside of a batch to use pack_padded sequence later
                sorted_batch = list(zip(batch_query_lengths, batch_data))
                sorted_batch.sort(reverse=True)


                batch_query_lengths, batch_data = zip(*sorted_batch)
                batch_document_lengths = [len(data_point.document_tokens)
                                          for data_point in batch_data]
                maximum_document_length = max(batch_document_lengths)
                maximum_query_length = max(batch_query_lengths)

                #0-pad sequences
                documents = [self.pad_seq(data_point.document_tokens,
                                          maximum_document_length) for data_point in batch_data]
                queries = [self.pad_seq(data_point.query_tokens, maximum_query_length)
                           for data_point in batch_data]
                answers = [data_point.answer_tokens[0] for data_point in batch_data]

                # Creates a mask that is equal to 0 for positions greater than the document length
                length_mask = [[int(x < batch_document_lengths[i])
                         for x in range(maximum_document_length)] for i in range(batch_length)]

                # Creates a mask that is equal to 0 for positions that are not answer positions
                # Used to calculate answer probability and loss
                answer_mask = [[int(x == answers[i]) for x in documents[i]]
                               for i in range(batch_length)]

                # An entity mask similar to the answer mask, for every entity in the document. 
                # Later used to calculate entity probabilities.
                entity_locations = [data_point.entity_locations for data_point in batch_data]

                batch = {}
                batch['documents'] = documents
                batch['queries'] = queries
                batch['answers'] = answers
                batch['doclengths'] = batch_document_lengths
                batch['qlengths'] = batch_query_lengths
                batch['docmask'] = length_mask
                batch['ansmask'] = answer_mask
                batch["entlocations"] = entity_locations

                return batch

            # Take care of last batch separately because may contain a different amount of data
            for j in range(number_batches - 1):
                begin_index, end_index = j * batch_size, (j + 1) * batch_size
                batch_data = list(bucket_data[begin_index:end_index])
                bucket.append(create_batch(batch_data))

            batch_data = list(bucket_data[end_index:])
            bucket.append(create_batch(batch_data))

            return bucket

        # Take care of last bucket separately because may contain a different amount of data
        for i in range(number_buckets - 1):
            bucket_data = temp_data[i * data_per_bucket:(i + 1) * data_per_bucket]
            batches += create_bucket(bucket_data)
        bucket_data = temp_data[(number_buckets - 1)*data_per_bucket:]
        batches += create_bucket(bucket_data)

        return batches

import glob
from data import Data
import os
import io
import random
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np


class Vocabulary(object):
    def __init__(self, pad_token='pad', unk='unk'):

        self.vocabulary = dict()
        self.inverse_vocabulary = dict()
        self.pad_token = pad_token
        self.unk = unk
        self.vocabulary[pad_token] = 0
        self.vocabulary[unk] = 1

    def add_and_get_index(self, word):
        if word in self.vocabulary:
            return self.vocabulary[word]
        else:
            length = len(self.vocabulary)
            self.vocabulary[word] = length
            self.inverse_vocabulary[length] = word
            return length

    def add_and_get_indices(self, words):
        return [self.add_and_get_index(word) for word in words]

    def get_index(self, word):
        return self.vocabulary.get(word, self.vocabulary[self.unk])

    def get_length(self):
        return len(self.vocabulary)



class DataLoader():
    def __init__(self):
        self.data_vocab = Vocabulary()

    def memory_usage_psutil(self):
        # return the memory usage in MB
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()[0] / float(2 ** 30)
        return mem

    # loads training data from directory
    def load_data(self, input_directory, max_number=0):

        data = []

        filenames = glob.glob(os.path.join(input_directory, '*.question'))
        number_files = len(filenames)
        if max_number != 0:
            number_files = min(max_number, number_files)

        for i in range(number_files):

            if (i + 1) % 20000 == 0:
                print("{} files loaded".format(i + 1))

            with io.open(filenames[i], encoding="utf8", errors='replace') as file:

                lines = file.readlines()

                # document, query and answer are at fixed location in files
                document_tokens = np.array(self.data_vocab.add_and_get_indices(lines[2].split()))
                query_tokens = np.array(self.data_vocab.add_and_get_indices(lines[4].split()))
                answer_tokens = np.array(self.data_vocab.add_and_get_indices(lines[6].split()))


                # For each entity, create a list with the locations of this entity in the document
                entity_locations = dict()

                for line in lines[8:]:
                    entity = self.data_vocab.add_and_get_index(line.strip('\n').split(":", 1)[0])
                    entity_locations[entity] = []
                for i in range(len(document_tokens)):
                    token = document_tokens[i]
                    if token in entity_locations:
                        entity_locations[token].append(i)

                # Make sure the answer entity is at the beginning of the list
                entity_locations = np.array([entity_locations[answer_tokens[0]]] + [entity_locations[ent]
                                                                                    for ent in entity_locations if ent != answer_tokens[0]])

                data_point = Data(document_tokens, query_tokens, answer_tokens, entity_locations)
                data.append(data_point)

        return data

    # This function basically randomizes a dictionary by shuffling the values while keeping the keys in place.
    # The goal here is to replace each entity with a different entity.
    def randomize_entities(self, entity_vocabulary):
        values = entity_vocabulary.values()
        random.shuffle(values)
        randomized_dictionary = dict(zip(entity_vocabulary.keys(), values))
        return randomized_dictionary

    # Replaces each entity with its randomized entity. Changes the actual data with side effects, but we rerandomize every epoch anyway.
    def replace_entities(self, data_point, randomized_vocabulary):
        data_point.document_tokens = np.array([randomized_vocabulary.get(
            data_point.document_tokens[i], data_point.document_tokens[i]) for i in range(len(data_point.document_tokens))])
        data_point.answer_tokens = np.array([randomized_vocabulary.get(
            data_point.answer_tokens[i], data_point.answer_tokens[i]) for i in range(len(data_point.answer_tokens))])
        data_point.query_tokens = np.array([randomized_vocabulary.get(
            data_point.query_tokens[i], data_point.query_tokens[i]) for i in range(len(data_point.query_tokens))])

    # Pad sequence. Note that this copies the list to avoid padding the data in place, because we call create_batches multiple times
    def pad_seq(self, seq, max_len):
        new_seq = np.zeros(max_len, np.int64)
        new_seq[:len(seq)] = seq
        return new_seq

    # Function creates batches by shuffling the data, making buckets of the data, and within those buckets, creating batches by sorting by document length
    def create_batches(self, data, batch_size, bucket_size):

        # Search out all the words in the vocabulary that are entities, and create a vocabulary mapping their word ids to themselves.
        # Later these will be shuffled.
        vocabulary = self.data_vocab.vocabulary
        entity_vocabulary = {vocabulary[word]: vocabulary[word]
                             for word in vocabulary if "@ent" in word}

        # shuffle the actual data
        temp_data = list(data)
        random.shuffle(temp_data)

        batches = []
        data_per_bucket = batch_size * bucket_size

        # Calculate number of buckets
        number_buckets = len(data) // data_per_bucket + int((len(data) % data_per_bucket) > 0)

        def create_bucket(bucket_data, batch_size):
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
                # We create a new random mapping from entity word ids to entity word ids and replace the entities

                randomized_entities = self.randomize_entities(entity_vocabulary)
                for data_point in batch_data:
                    self.replace_entities(data_point, randomized_entities)

                batch_query_lengths = np.array([len(data_point.query_tokens) for data_point in batch_data])

                batch_document_lengths = np.array([len(data_point.document_tokens)
                                          for data_point in batch_data])

                maximum_document_length = max(batch_document_lengths)
                maximum_query_length = max(batch_query_lengths)

                # 0-pad sequences
                documents = np.array([self.pad_seq(data_point.document_tokens,
                                                   maximum_document_length) for data_point in batch_data])
                queries = np.array([self.pad_seq(data_point.query_tokens, maximum_query_length)
                                    for data_point in batch_data])
                answers = np.array([data_point.answer_tokens[0] for data_point in batch_data])

                # Creates a mask that is equal to 0 for positions that are not answer positions
                # Used to calculate answer probability and loss
                answer_mask = np.array([[int(x == answers[i]) for x in documents[i]]
                                        for i in range(batch_length)])

                # Create length mask
                document_mask = np.array([[int(x < batch_document_lengths[i]) 
                                         for x in range(maximum_document_length)] for i in range(batch_length)])
                query_mask = np.array([[int(x < batch_query_lengths[i]) 
                                         for x in range(maximum_query_length)] for i in range(batch_length)])

                # An entity mask similar to the answer mask, for every entity in the document.
                # Later used to calculate entity probabilities.
                entity_locations = [data_point.entity_locations for data_point in batch_data]

                batch = {}
                batch['documents'] = documents
                batch['queries'] = queries
                batch['answers'] = answers
                batch['doclengths'] = batch_document_lengths
                batch['qlengths'] = batch_query_lengths
                batch['ansmask'] = answer_mask
                batch['docmask'] = document_mask
                batch['qmask'] = query_mask
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
            batches += create_bucket(bucket_data, batch_size)

            if (i + 1) % 100 == 0:
                print("{} buckets created".format(i + 1))

        bucket_data = temp_data[(number_buckets - 1) * data_per_bucket:]
        batches += create_bucket(bucket_data, batch_size)

        return batches

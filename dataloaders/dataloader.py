import glob
from data import Data
import os
import io
import random



class DataLoader():
    def __init__(self):
        pass

    def load_data(self, input_directory, max_number=False):

        data = []

        filenames = glob.glob(os.path.join(input_directory, '*.question'))
        number_files = len(filenames)
        if max_number:
            number_files = max_number

        for i in range(number_files):
            with io.open(filenames[i], encoding="utf8", errors='replace') as file:

                lines = file.readlines()

                document_tokens = lines[2].split()
                query_tokens = lines[4].split()
                answer_tokens = lines[6].split()
                entities = []

                for line in lines[8:]:
                    index, entity = line.strip('\n').split(":", 1)
                    entities.append(entity)

                data_point = Data(document_tokens, query_tokens, answer_tokens, entities)
                data.append(data_point)

        return data


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


    # Warning: this alters the data objects in place, side effects
    def process_data(self, data, vocabulary):

        for data_point in data:

            data_point.document_tokens = self.word_to_id(data_point.document_tokens, vocabulary)
            data_point.query_tokens = self.word_to_id(data_point.query_tokens, vocabulary)
            data_point.answer_tokens = self.word_to_id(data_point.answer_tokens, vocabulary)

    def randomize_entities(self, entity_vocabulary):
            values = entity_vocabulary.values()
            random.shuffle(values)
            randomized_dictionary = dict(zip(entity_vocabulary.keys(), values))
            return randomized_dictionary

    def replace_entities(self, data_point, randomized_vocabulary):
            data_point.document_tokens = [randomized_vocabulary.get(data_point.document_tokens[i], data_point.document_tokens[i]) for i in range(len(data_point.document_tokens))]
            data_point.answer_tokens = [randomized_vocabulary.get(data_point.answer_tokens[i], data_point.answer_tokens[i]) for i in range(len(data_point.answer_tokens))]
            data_point.query_tokens = [randomized_vocabulary.get(data_point.query_tokens[i], data_point.query_tokens[i]) for i in range(len(data_point.query_tokens))]

    def pad_seq(self, seq, max_len, pad_token = 0):
        seq += [pad_token for i in range(max_len-len(seq))]
        return seq
            
            
    def create_batches(self, data, batch_size, bucket_size, vocabulary):

        entity_vocabulary = {word: vocabulary[word] for word in vocabulary if "@ent" in word}

        temp_data = list(data)
        random.shuffle(temp_data)

        batches = []
        data_per_bucket = batch_size * bucket_size
        number_buckets = len(data) // data_per_bucket + int((len(data) % data_per_bucket) > 0)
        
        def create_bucket(bucket_data):
            bucket = []
            document_lengths = [len(data_point.document_tokens) for data_point in bucket_data]

            sorted_data = list(zip(document_lengths, bucket_data))
            sorted_data.sort(reverse=True)

            document_lengths, bucket_data = zip(*sorted_data)

            number_batches = len(bucket_data) // batch_size + int((len(bucket_data) % batch_size) > 0)

            def create_batch(batch_data):

                batch_length = len(batch_data)

                randomized_entities = self.randomize_entities(entity_vocabulary)
                for data_point in batch_data:
                    self.replace_entities(data_point, randomized_entities)

                batch_query_lengths = [len(data_point.query_tokens) for data_point in batch_data]
                
                sorted_batch = list(zip(batch_query_lengths, batch_data))
                sorted_batch.sort(reverse=True)

                batch_query_lengths, batch_data = zip(*sorted_batch)
                batch_document_lengths = [len(data_point.document_tokens) for data_point in batch_data]
                maximum_document_length = max(batch_document_lengths)
                maximum_query_length = max(batch_query_lengths)

                documents = [self.pad_seq(data_point.document_tokens, maximum_document_length) for data_point in batch_data]
                queries = [self.pad_seq(data_point.query_tokens, maximum_query_length) for data_point in batch_data]
                answers = [data_point.answer_tokens[0] for data_point in batch_data]
                mask = [[int(x < batch_document_lengths[i]) for x in range(maximum_document_length)] for i in range(batch_length)]

                answer_mask = [[int(x == answers[i]) for x in documents[i]] for i in range(batch_length)]

                batch = {}
                batch['documents'] = documents
                batch['queries'] = queries
                batch['answers'] = answers
                batch['doclengths'] = batch_document_lengths
                batch['qlengths'] = batch_query_lengths
                batch['docmask'] = mask
                batch['ansmask'] = answer_mask

                return batch

            for j in range(number_batches - 1):
                begin_index, end_index = j * batch_size, (j + 1) * batch_size
                batch_data = list(bucket_data[begin_index:end_index])
                bucket.append(create_batch(batch_data))

            batch_data = list(bucket_data[end_index:])
            bucket.append(create_batch(batch_data))
                
            return bucket
                
        for i in range(number_buckets - 1):
            bucket_data = temp_data[i * data_per_bucket:(i + 1) * data_per_bucket]
            batches += create_bucket(bucket_data)
        bucket_data = temp_data[number_buckets - 1:]
        batches += create_bucket(bucket_data)

        return batches


try:
    import cPickle as pickle
except:
    import pickle
import glob
from data import Data
import os
import io

class DataLoader():
    def __init__(self):
        pass


    def process_data(self, input_directory, output_path, max_number = False):
        
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
                answer_token = lines[6]
                entities = []

                for line in lines[8:]:
                    index, entity = line.strip('\n').split(":", 1)
                    entities.append(entity)
            
                data_point = Data(document_tokens, query_tokens, answer_token, entities)
                data.append(data_point)    

        with open(output_path, "wb") as fout:
            pickle.dump(data, fout)


    def generate_vocabulary(self, data, special_tokens = ["<unk>"]):

        word_set = set()

        for data_point in data:
            tokens = data_point.document_tokens + data_point.query_tokens + data_point.index_to_entity.keys()
            for token in tokens:
                word_set.add(token)

        for token in special_tokens:
            word_set.add(token)

        vocab = {}
        for i, elem in enumerate(word_set):
            vocab[elem] = i

        return vocab


    def replace_words(self, tokens):
        ids = []
        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                ids.append(self.word_to_id["<unk>"])
        return ids


    def get_data_set(self, data_path):

        data = pickle.load(open(data_path, "rb"))

        documents = []
        questions = []
        answers = []
        entities = []

        for data_point in data:

            documents.append(self.replace_words(data_point.document_tokens))
            questions.append(self.replace_words(data_point.question_tokens))
            answers.append(self.replace_words(data_point.answer_token))
            entities.append(self.replace_words(data_point.entities))

        return documents, questions, answers, entities

    # def process_dataset(self, input_directory, output_directory):

    #     self.process_data(os.path.join(input_directory, "training"), os.path.join(output_directory, "training.pickle"))
    #     self.process_data(os.path.join(input_directory, "validation"), os.path.join(output_directory, "validation.pickle"))
    #     self.process_data(os.path.join(input_directory, "test"), os.path.join(output_directory, "test.pickle"))




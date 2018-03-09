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
                answer = lines[6]

                index_to_entity = {}
                entity_to_index = {}

                for line in lines[8:]:
                    try:
                        index, entity = line.strip('\n').split(":", 1)
                    except:
                        print(line)
                        print(line.strip('\n').split(":"))
                    index_to_entity[index] = entity
                    entity_to_index[entity] = index

                data_point = Data(document_tokens, query_tokens, answer, entity_to_index, index_to_entity)
                data.append(data_point)    

        with open(output_path, "wb") as fout:
            pickle.dump(data, fout)



    def process_dataset(self, input_directory, output_directory):

        self.process_data(os.path.join(input_directory, "training"), os.path.join(output_directory, "training.pickle"))
        self.process_data(os.path.join(input_directory, "validation"), os.path.join(output_directory, "validation.pickle"))
        self.process_data(os.path.join(input_directory, "test"), os.path.join(output_directory, "test.pickle"))
        



test = DataLoader()
test_dir = "/home/michiel/main/datasets/asreader/data/cnn/questions/"
output_path = "/home/michiel/main/datasets/asreader/data/cnn/processed/"

test.process_dataset(test_dir, output_path)

with open("/home/michiel/main/datasets/asreader/data/cnn/processed/validation.pickle", "r") as fout:
            a = pickle.load(fout)

print(a[0].document_tokens)
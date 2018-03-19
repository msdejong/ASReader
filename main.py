import sys
import argparse
from dataloaders.dataloader import DataLoader
from models.ASReader import ASReader
import torch
from torch.autograd import Variable
from torch import optim


def evaluate(model, eval_batches):
    pass

def train(model, training_data, dataloader, vocabulary, num_epochs=2, batch_size=32, bucket_size=10, learning_rate=0.001, clip_threshold=10, train_loss_interval = 10):

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        
        train_loss = 0
        training_batches = dataloader.create_batches(training_data, batch_size, bucket_size, vocabulary)

        optimizer.zero_grad()

        num_iterations = len(training_batches)
        for iteration in range(num_iterations):
            if iteration%train_loss_interval == 0:
                print("iteration {}".format(iteration))
                print("train loss: {}".format(train_loss))
                train_loss=0
            batch = training_batches[iteration]
            batch_documents = Variable(torch.LongTensor(batch['documents']))
            batch_queries = Variable(torch.LongTensor(batch['queries']))
            batch_query_lengths = batch['qlengths']
            batch_mask = Variable(torch.FloatTensor(batch['docmask']).unsqueeze(-1))
            batch_answer_mask = Variable(torch.FloatTensor(batch['ansmask']).unsqueeze(-1))
    
            value = model(batch_documents, batch_queries, batch_query_lengths, batch_mask)
            loss =  model.loss(value, batch_answer_mask)
            train_loss += loss

            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
            optimizer.step()





if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--train_path", type=str, default="/home/michiel/main/datasets/asreader/data/cnn/questions/training")
    parser.add_argument("--max_train", type=int, default= 10000)
    parser.add_argument("--test_path", type=str, default= None)
    parser.add_argument("--model_path", type=str, default=None)


    #Model specific arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bucket_size", type=int, default=10)
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=2)

    args = parser.parse_args()

    

    DL = DataLoader()
    training_data = DL.load_data(args.train_path, args.max_train)
    vocabulary = DL.generate_vocabulary(training_data)
    DL.process_data(training_data, vocabulary)
    model = ASReader(len(vocabulary), args.embedding_dim, args.encoding_dim)

    train(model, training_data, DL, vocabulary)






    # batches = DL.create_batches(training_data, batch_size, bucket_size, vocabulary)
    # test_batch = batches[0]

    # test_documents = Variable(torch.LongTensor(test_batch['documents']))
    # test_queries = Variable(torch.LongTensor(test_batch['queries']))
    # test_answer = test_batch['answers'][0]

    # # print(" ".join([inverse_vocabulary[test_batch['documents'][0][i]] for i in range(len(test_batch['documents'][0]))]))
    # # print(" ".join([inverse_vocabulary[test_batch['queries'][0][i]] for i in range(len(test_batch['queries'][0]))]))
    # # print(test_answer)
    # # print(inverse_vocabulary[test_answer])


    # test_query_lengths = test_batch['qlengths']
    
    # test_mask = Variable(torch.FloatTensor(test_batch['docmask']).unsqueeze(-1))
    # test_answer_mask = Variable(torch.FloatTensor(test_batch['ansmask']).unsqueeze(-1))
    


    # # print(batches[0]['documents'])
    # # print(test_queries[0], test_queries[-1])

    # test_model = ASReader(len(vocabulary), embedding_dim, encoding_dim)
    
    # # print(test_value)
    # print(test_loss)

    







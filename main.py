from __future__ import division
import sys
import argparse
from dataloaders.dataloader import DataLoader
from models.ASReader import ASReader
import torch
from torch.autograd import Variable
from torch import optim, cuda
import numpy as np


def evaluate_batches(model, eval_batches):

    score = 0
    denom = 0

    for batch in eval_batches:
        

        batch_query_lengths = batch['qlengths']
        batch_doc_lengths = batch['doclengths']
        batch_entity_mask = batch['entlocations']


        batch_documents = Variable(torch.LongTensor(batch['documents'])) 
        batch_queries = Variable(torch.LongTensor(batch['queries'])) 
        batch_mask = Variable(torch.FloatTensor(batch['docmask']).unsqueeze(-1))

        if USE_CUDA:
            batch_documents = batch_documents.cuda()
            batch_queries = batch_queries.cuda()
            batch_mask = batch_mask.cuda()
            probs = model(batch_documents, batch_queries, batch_query_lengths, batch_mask).data.cpu().numpy()

        else:
            probs = model(batch_documents, batch_queries, batch_query_lengths, batch_mask).data.numpy()
        
        score += evaluate(probs, batch_entity_mask, batch_doc_lengths) * len(batch_entity_mask)
        denom += len(batch_entity_mask)

    accuracy = score / denom
    return accuracy

def evaluate(probs, entity_masks, doc_lengths):
    batch_len = len(entity_masks)
    accuracy = 0
    for i in range(batch_len):
            entity_probs = np.matmul(entity_masks[i], probs[i, :][:doc_lengths[i]])
            prediction = np.argmax(entity_probs)
            if prediction ==0:
                accuracy += 1/batch_len
    return accuracy



def train(model, training_data, dataloader, vocabulary, num_epochs=2, batch_size=32, bucket_size=10, learning_rate=0.001, clip_threshold=10, eval_interval=20, valid_data=False):


    if valid_data:
        valid_batches = dataloader.create_batches(valid_data, batch_size, bucket_size, vocabulary)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        
        train_loss = 0
        train_score = 0
        train_denom = 1

        training_batches = dataloader.create_batches(training_data, batch_size, bucket_size, vocabulary)

        optimizer.zero_grad()

        num_iterations = len(training_batches)
        for iteration in range(num_iterations):
            if iteration%eval_interval == 0:
                print("iteration {}".format(iteration))
                print("train loss: {}".format(train_loss/train_denom))
                print("train accuracy: {}".format(train_score/train_denom))
                train_loss=0
                train_score= 0
                train_denom=0

            if valid_data and iteration%eval_interval == 0:
                valid_accuracy = evaluate_batches(model, valid_batches)
                print("valid accuracy: {}".format(valid_accuracy))
                


            batch = training_batches[iteration]

            batch_query_lengths = batch['qlengths']
            batch_doc_lengths = batch['doclengths']
            batch_entity_mask = batch['entlocations']

            batch_documents = Variable(torch.LongTensor(batch['documents']))
            batch_queries = Variable(torch.LongTensor(batch['queries']))
            batch_mask = Variable(torch.FloatTensor(batch['docmask']).unsqueeze(-1))
            batch_answer_mask = Variable(torch.FloatTensor(batch['ansmask']).unsqueeze(-1))
            
            if USE_CUDA:
                batch_documents = batch_documents.cuda()
                batch_queries = batch_queries.cuda()
                batch_mask = batch_mask.cuda()
                batch_answer_mask = batch_answer_mask.cuda()

            probs = model(batch_documents, batch_queries, batch_query_lengths, batch_mask)
            loss =  model.loss(probs, batch_answer_mask)

            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
            optimizer.step()

            if USE_CUDA:
                train_loss += loss.data.cpu().numpy()
                probs = probs.data.cpu().numpy()
            else:
                train_loss += loss.data.numpy()
                probs = probs.data.numpy()

            train_score += evaluate(probs, batch_entity_mask, batch_doc_lengths) * len(batch_entity_mask)
            train_denom += len(batch_entity_mask)

            





if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--train_path", type=str, default="/home/michiel/main/datasets/asreader/data/cnn/questions/training")
    parser.add_argument("--max_train", type=int, default= 10000)
    parser.add_argument("--valid_path", type=str, default="/home/michiel/main/datasets/asreader/data/cnn/questions/validation")
    parser.add_argument("--max_valid", type=int, default= 200)
    parser.add_argument("--eval_interval", type=int, default=50)

    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bucket_size", type=int, default=10)
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=2)

    args = parser.parse_args()

    if args.use_cuda and cuda.is_available():
        USE_CUDA = True
    else:
        USE_CUDA = False

    DL = DataLoader()
    training_data = DL.load_data(args.train_path, args.max_train)
    valid_data = DL.load_data(args.valid_path, args.max_valid)
    vocabulary = DL.generate_vocabulary(training_data)

    DL.process_data(training_data, vocabulary)
    DL.process_data(valid_data, vocabulary)

    model = ASReader(len(vocabulary), args.embedding_dim, args.encoding_dim)
    train(model, training_data, DL, vocabulary, batch_size=args.batch_size, bucket_size=args.bucket_size, learning_rate=args.learning_rate, valid_data=valid_data, eval_interval=args.eval_interval)


    batches = DL.create_batches(valid_data, args.batch_size, args.bucket_size, vocabulary)
    accuracy = evaluate_batches(model, batches)
    print(accuracy)



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

    







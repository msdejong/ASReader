from __future__ import division
import sys
import argparse
from dataloaders.dataloader import DataLoader
from models.ASReader import ASReader
import torch
from torch.autograd import Variable
from torch import optim, cuda
import numpy as np
from timeit import default_timer as tm
try:
    import cPickle as pickle
except:
    import pickle
import random
from itertools import ifilter
import cProfile

def evaluate_batches(model, eval_batches):

    score = 0
    denom = 0

    for batch in eval_batches:

        batch_query_lengths = batch['qlengths']
        batch_entity_locations = batch['entlocations']
        batch_length = len(batch_query_lengths)

        batch_documents = Variable(torch.LongTensor(batch['documents']))
        batch_queries = Variable(torch.LongTensor(batch['queries']))
        batch_document_mask = Variable(torch.FloatTensor(batch['docmask']).unsqueeze(-1))
        batch_query_mask = Variable(torch.FloatTensor(batch['qmask']).unsqueeze(-1))

        spans=batch["spans"]
        scores=batch["scores"]
        

        if USE_CUDA:
            batch_documents = batch_documents.cuda()
            batch_queries = batch_queries.cuda()
            batch_document_mask = batch_document_mask.cuda()
            batch_query_mask = batch_query_mask.cuda()
            probs = model(batch_documents, batch_queries, batch_document_mask, batch_query_mask, spans, scores).data.cpu().numpy()

        else:
            probs = model(batch_documents, batch_queries, batch_document_mask, batch_query_mask, spans, scores).data.numpy()

        score += evaluate(probs, batch_entity_locations, batch_length) * batch_length
        denom += batch_length

    accuracy = score / denom
    return accuracy


def evaluate(probs, batch_entity_locations, batch_length):
    accuracy = 0
    for i in range(batch_length):
        entity_locations = batch_entity_locations[i]

        entity_probs = np.zeros(len(entity_locations))
        for j in range(len(entity_locations)):
            entity_probs[j] = np.sum(probs[i, entity_locations[j]])
        prediction = np.argmax(entity_probs)
        if prediction == 0:
            accuracy += 1 / batch_length
    return accuracy

def get_trainable_parameters(model):
    parameters = ifilter(lambda p: p.requires_grad, model.parameters())
    return parameters


def train(model, training_data, valid_data, test_data, dataloader, num_epochs=2, batch_size=32, bucket_size=10, learning_rate=0.001, clip_threshold=10, eval_interval=20, model_path=""):

    print("Creating validation batches")
    valid_batches = dataloader.create_batches(valid_data, batch_size, bucket_size)
    print("Creating test batches")
    test_batches = dataloader.create_batches(test_data, batch_size, bucket_size)


    optimizer = optim.Adam(get_trainable_parameters(model), lr=learning_rate)

    train_loss = 0
    train_score = 0
    train_denom = 0
    best_valid_accuracy = 0
    test_best_valid = 0

    for epoch in range(num_epochs):

        print("Starting epoch {}".format(epoch + 1))
        print("Creating training batches")
        training_batches = dataloader.create_batches(training_data, batch_size, bucket_size)

        num_iterations = len(training_batches)
        for iteration in range(num_iterations):

            optimizer.zero_grad()

            # track and report performance
            if (iteration + 1) % eval_interval == 0:

                valid_accuracy = evaluate_batches(model, valid_batches)
                test_accuracy = evaluate_batches(model, test_batches)

                print("iteration {}".format(iteration + 1))
                print("train loss: {}".format(train_loss / train_denom))
                print("train accuracy: {}".format(train_score / train_denom))
                print("valid accuracy: {}".format(valid_accuracy))
                print("test accuracy: {}".format(test_accuracy))

                # Update best model
                if valid_accuracy > best_valid_accuracy:
                    best_valid_accuracy = valid_accuracy
                    test_best_valid = test_accuracy
                    torch.save(model.state_dict(), model_path)

                train_loss = 0
                train_score = 0
                train_denom = 0

            batch = training_batches[iteration]

            batch_length = len(batch["documents"])

            # document tokens
            batch_documents = Variable(torch.LongTensor(batch['documents']))
            # query tokens
            batch_queries = Variable(torch.LongTensor(batch['queries']))
            # 1 for locations with the answer token and 0 elsewhere
            batch_answer_mask = Variable(torch.FloatTensor(batch['ansmask']).unsqueeze(-1))
            batch_document_mask = Variable(torch.FloatTensor(batch['docmask']).unsqueeze(-1))
            batch_query_mask = Variable(torch.FloatTensor(batch['qmask']).unsqueeze(-1))

            spans=batch["spans"]
            scores=batch["scores"]


            # Similar to answer mask, but for every other entity
            batch_entity_locations = batch['entlocations']

            if USE_CUDA:
                batch_documents = batch_documents.cuda()
                batch_queries = batch_queries.cuda()
                batch_answer_mask = batch_answer_mask.cuda()
                batch_document_mask = batch_document_mask.cuda()
                batch_query_mask = batch_query_mask.cuda()

            probs = model(batch_documents, batch_queries, batch_document_mask, batch_query_mask, spans, scores)
            loss = model.loss(probs, batch_answer_mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
            optimizer.step()

            # Calculate training loss and accuracy.
            # Multiplying by batch length and later dividing by sum of batch lengths to be correct for differing batch length at end of epoch

            if USE_CUDA:
                train_loss += loss.data.cpu().numpy()[0] * batch_length
                probs = probs.data.cpu().numpy()
            else:
                train_loss += loss.data.numpy()[0] * batch_length
                probs = probs.data.numpy()

            # Evaluate train accuracy
            train_score += evaluate(probs, batch_entity_locations, batch_length) * batch_length
            train_denom += batch_length

    print("Training completed")
    print("Maximum validation accuracy: {}".format(best_valid_accuracy))
    print("Corresponding test accuracy: {}".format(test_best_valid))


# This file loads the data and trains and evaluates the model

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str,
                        default="/home/michiel/main/datasets/asreader/data/cnn/questions/training")
    parser.add_argument("--valid_path", type=str,
                        default="/home/michiel/main/datasets/asreader/data/cnn/questions/validation")
    parser.add_argument("--test_path", type=str,
                        default="/home/michiel/main/datasets/asreader/data/cnn/questions/test")

    parser.add_argument("--model_path", type=str,
                        default="test.pth")


    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_valid", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)

    parser.add_argument("--eval_interval", type=int, default=500)

    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bucket_size", type=int, default=10)
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.use_cuda and cuda.is_available():
        USE_CUDA = True
    else:
        USE_CUDA = False


    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)


    DL = DataLoader()

    print("Loading training data")
    training_data = DL.load_data(args.train_path, args.max_train)
    print("Loading validation data")
    valid_data = DL.load_data(args.valid_path, args.max_valid)
    print("Loading test data")
    test_data = DL.load_data(args.test_path, args.max_test)

    model = ASReader(DL.data_vocab.get_length(), args.encoding_dim)
    if USE_CUDA:
        model.cuda()

    print("Starting training")
    train(model, training_data, valid_data, test_data, DL, batch_size=args.batch_size, bucket_size=args.bucket_size,
          learning_rate=args.learning_rate, eval_interval=args.eval_interval, num_epochs=args.num_epochs, model_path=args.model_path)

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


def evaluate_batches(model, eval_batches):

    score = 0
    denom = 0

    for batch in eval_batches:

        batch_query_lengths = batch['qlengths']
        batch_doc_lengths = batch['doclengths']
        batch_entity_locations = batch['entlocations']
        batch_length = len(batch_query_lengths)

        batch_documents = Variable(torch.LongTensor(batch['documents']))
        batch_queries = Variable(torch.LongTensor(batch['queries']))
        # Index to un- sort the arrays
        batch_query_unsort = Variable(torch.LongTensor(batch['qunsort']))
        batch_document_unsort = Variable(torch.LongTensor(batch['docunsort']))

        batch_length_mask = Variable(torch.FloatTensor(batch['lengthmask']).unsqueeze(-1))

        if USE_CUDA:
            batch_documents = batch_documents.cuda()
            batch_queries = batch_queries.cuda()
            batch_query_unsort = batch_query_unsort.cuda()
            batch_document_unsort = batch_document_unsort.cuda()
            batch_length_mask = batch_length_mask.cuda()
            probs = model(batch_documents, batch_queries, batch_query_lengths,
                          batch_doc_lengths, batch_query_unsort, batch_document_unsort, batch_length_mask).data.cpu().numpy()

        else:
            probs = model(batch_documents, batch_queries, batch_query_lengths,
                          batch_doc_lengths, batch_query_unsort, batch_document_unsort, batch_length_mask).data.numpy()

        score += evaluate(probs, batch_entity_locations,
                          batch_doc_lengths[batch['docunsort']], batch_length) * batch_length
        denom += batch_length

    accuracy = score / denom
    return accuracy


def evaluate(probs, batch_entity_locations, doc_lengths, batch_length):
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


def train(model, training_data, dataloader, num_epochs=2, batch_size=32, bucket_size=10, learning_rate=0.001, clip_threshold=10, eval_interval=20, valid_data=False):

    if valid_data:
        valid_batches = dataloader.create_batches(valid_data, batch_size, bucket_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        train_loss = 0
        train_score = 0
        train_denom = 0

        print("Starting epoch {}".format(epoch + 1))
        print("Creating batches")
        training_batches = dataloader.create_batches(training_data, batch_size, bucket_size)

        num_iterations = len(training_batches)
        for iteration in range(num_iterations):

            optimizer.zero_grad()

            # track performance
            if (iteration + 1) % eval_interval == 0:
                print("iteration {}".format(iteration + 1))
                print("train loss: {}".format(train_loss / train_denom))
                print("train accuracy: {}".format(train_score / train_denom))
                train_loss = 0
                train_score = 0
                train_denom = 0

            if valid_data and (iteration + 1) % eval_interval == 0 and iteration != 0:
                valid_accuracy = evaluate_batches(model, valid_batches)
                print("valid accuracy: {}".format(valid_accuracy))

            batch = training_batches[iteration]

            # Note: lengths are in descending order, not in the proper batch order
            batch_query_lengths = batch['qlengths']
            batch_doc_lengths = batch['doclengths']

            batch_length = len(batch_query_lengths)

            # document tokens
            batch_documents = Variable(torch.LongTensor(batch['documents']))
            # query tokens
            batch_queries = Variable(torch.LongTensor(batch['queries']))
            # 1 for locations with the answer token and 0 elsewhere
            batch_answer_mask = Variable(torch.FloatTensor(batch['ansmask']).unsqueeze(-1))
            batch_length_mask = Variable(torch.FloatTensor(batch['lengthmask']).unsqueeze(-1))

            # Similar to answer mask, but for every other entity
            batch_entity_locations = batch['entlocations']
            # Index to un- sort the arrays
            batch_query_unsort = Variable(torch.LongTensor(batch['qunsort']))
            batch_document_unsort = Variable(torch.LongTensor(batch['docunsort']))

            if USE_CUDA:
                batch_documents = batch_documents.cuda()
                batch_queries = batch_queries.cuda()
                batch_answer_mask = batch_answer_mask.cuda()
                batch_length_mask = batch_length_mask.cuda()
                batch_query_unsort = batch_query_unsort.cuda()
                batch_document_unsort = batch_document_unsort.cuda()

            probs = model(batch_documents, batch_queries, batch_query_lengths,
                          batch_doc_lengths, batch_query_unsort, batch_document_unsort, batch_length_mask)
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

            # Evaluate train accuracy, have to use the unsorted document lengths
            train_score += evaluate(probs, batch_entity_locations,
                                    batch_doc_lengths[batch['docunsort']], batch_length) * batch_length
            train_denom += batch_length


# This file loads the data and trains and evaluates the model

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str,
                        default="/home/michiel/main/datasets/asreader/data/cnn/questions/training")
    parser.add_argument("--max_train", type=int, default=64)
    parser.add_argument("--valid_path", type=str,
                        default="/home/michiel/main/datasets/asreader/data/cnn/questions/validation")
    parser.add_argument("--max_valid", type=int, default=64)
    parser.add_argument("--eval_interval", type=int, default=2)

    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bucket_size", type=int, default=10)
    parser.add_argument("--encoding_dim", type=int, default=384)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()

    if args.use_cuda and cuda.is_available():
        USE_CUDA = True
    else:
        USE_CUDA = False

    DL = DataLoader()

    print("Loading data")
    training_data = DL.load_data(args.train_path, args.max_train)
    valid_data = DL.load_data(args.valid_path, args.max_valid)

    model = ASReader(DL.data_vocab.get_length(), args.embedding_dim, args.encoding_dim)
    if USE_CUDA:
        model.cuda()

    print("Starting training")
    train(model, training_data, DL, batch_size=args.batch_size, bucket_size=args.bucket_size,
          learning_rate=args.learning_rate, valid_data=valid_data, eval_interval=args.eval_interval, num_epochs=args.num_epochs)

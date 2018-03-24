import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ASReader(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoding_dim):
        super(ASReader, self).__init__()

        self.embedding_dim = embedding_dim
        self.encoding_dim = encoding_dim

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        self.document_encoding = nn.GRU(embedding_dim, encoding_dim,
                                        bidirectional=True, batch_first=True)
        self.query_encoding = nn.GRU(embedding_dim, encoding_dim,
                                     bidirectional=True, batch_first=True)
        self.softmax = nn.Softmax(dim=1)

    def softmax(self, scores, mask):
        input_masked = scores*mask
        shifted = mask*(input_masked - input_masked.max( dim=1))
        Z = torch.log(mask*torch.exp(shifted)).sum(dim=1)
        result = mask*(shifted - Z)
        return mask*torch.exp(result)

    def forward(self, document_batch, query_batch, query_lengths, document_lengths, query_unsort, document_unsort, length_mask):
        
        query_embedded = self.embedding_layer(query_batch)
        document_embedded = self.embedding_layer(document_batch)
        
        # Pack the padded sequences so they are ignored in the computation
        query_packed = pack_padded_sequence(query_embedded, query_lengths, batch_first=True)
        document_packed = pack_padded_sequence(document_embedded, document_lengths, batch_first=True)

        # Retrieve the last hidden state of the BiGRU as the query encoding
        query_encoded = self.query_encoding(query_packed)[1]
        query_encoded = query_encoded.permute(1, 0, 2).contiguous().view(-1, self.encoding_dim * 2, 1)

        # The hidden states of the document BiGRU correspond to the document token encodings
        document_encoded = self.document_encoding(document_packed)[0]
        
        # Unpack document
        document_unpacked, _ = pad_packed_sequence(document_encoded, batch_first=True)

        # The queries and documents were separately ordered by length, here we put them back
        query_unsorted = torch.index_select(query_encoded, 0, query_unsort)
        document_unsorted = torch.index_select(document_unpacked, 0, document_unsort)

        # Take the dot product of document encodings and the query encoding.
        scores = torch.bmm(document_unsorted, query_unsorted)*length_mask
        probs = self.softmax(scores, length_mask)

        # Normalize properly by multiplying by ratio of sum including and excluding padded values
        # denom_ratio = (torch.sum(probs, dim=1) / torch.sum(probs * length_mask, dim=1)).squeeze()
        # probs = probs * denom_ratio
        
        return probs

    def loss(self, probs, answer_mask):

        # Calculate the sum of probabilities over the positions in the document that have the answer token by multiplying all other positions by 0
        answer_probs = torch.sum(probs * answer_mask, 1)
        loss_vector = -torch.log(answer_probs)
        return torch.mean(loss_vector)

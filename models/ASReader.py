import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import orthogonal
from encoderblock import EncoderBlock
import pdb

def print_grad(name):
    def hook(grad):
        # pdb.set_trace()
        print(name, grad)
    return hook

class ASReader(nn.Module):
    def __init__(self, vocab_size, encoding_dim):
        super(ASReader, self).__init__()

        self.encoding_dim = encoding_dim

        self.embedding_layer = nn.Embedding(vocab_size, encoding_dim)

        self.document_encoding = EncoderBlock(5000, encoding_dim, 7, 4, 8)
        self.query_encoding = EncoderBlock(5000, encoding_dim, 7, 4, 8)
        # self.softmax = nn.Softmax(dim=1)

        self.initialize_weights()


    def forward(self, document_batch, query_batch, document_mask, query_mask):

        query_embedded = self.embedding_layer(query_batch)
        query_encoded = self.query_encoding(query_embedded, query_mask)
        query_pooled = F.max_pool1d(query_encoded.permute(0, 2, 1), kernel_size=query_encoded.size(1))
        query_encoded.register_hook(print_grad('input'))
        document_embedded = self.embedding_layer(document_batch)
        document_encoded = self.document_encoding(document_embedded, document_mask)

        # Take the dot product of document encodings and the query encoding.
        scores = torch.bmm(document_encoded, query_pooled)
        probs = self.softmax_mask(scores, document_mask)

        return probs

    def loss(self, probs, answer_mask):
        probs.register_hook(print_grad('probs'))

        # Calculate the sum of probabilities over the positions in the document that have the answer token by multiplying all other positions by 0
        answer_probs = torch.sum(probs * answer_mask, 1)
        loss_vector = -torch.log(answer_probs)
        return torch.mean(loss_vector)

    def softmax_mask(self, scores, mask):
        masked_scores = scores * mask
        shifted_scores = (masked_scores - torch.max(masked_scores, dim=1, keepdim=True)[0]) * mask
        logdenom = torch.log(torch.sum(torch.exp(shifted_scores) * mask, dim=1, keepdim=True))
        logprobs = (shifted_scores-logdenom) * mask 
        probs = torch.exp(logprobs) * mask
        return probs

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-0.1, 0.1)
            # if isinstance(m, nn.GRU):
            
        



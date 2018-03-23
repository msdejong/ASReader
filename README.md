# ASReader

Done / understood:
Batches: create 10 batches ahead, sort by context length, low to high
Entities: shuffle entities across entire dataset every batch (jesus christ why)
Special tokens: just unk. Question end token only if adding question to context. They set EOS and BOS to none
pad queries, then use packed sequence for query GRU
pad documents, after multiplying with query, mask
Loss: log likelihood of answer
accuracy measure is standard accuracy
Gradient clipping at 10
leftover batches and buckets are allowed to be smaller size
learning rate 0.001
candidate answers are the entities
pickle is slower than just loading
divide loss by batch size

TODO:

Bias: initialized to 0
Weights: isotropic gaussian(?)
embedding weight initialization
hidden layer initialization
parallelize data loading?
save models
check random seed
unknown words?
entities in validation data



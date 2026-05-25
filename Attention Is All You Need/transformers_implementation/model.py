import torch 
import torch.nn as nn
import math 

class InputEmbeddings(nn.Module):

    '''
        The first step in the transformers architecture: turn i/p token ids into embedding vectors
        Text enters the models as token IDs from a vocabulary
        InputEmbeddings does token ID  →  vector of size d_model
        two hyperparameters are declared- d_model and vocab_size
        d_model: is the dimension of the embedding vectors (paper uses 512)
        vocab_size: is the size of the vocabulary
    '''

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    '''
        The second step in the transformers architecture: add positional encoding to the input embeddings   
        This tells the model the position of the words in the sentence
        we will make a matrix of shape (seq_len, d_model)
        why d_model size? - we need vectors of d_model size and we need seq_len number of such vectors
        why seq_len? - the maximum length of the original or i/p sentence is seq_len
        for formula we will use simplified formula using log space- we will apply exponential and log(something) inside exponential it makes the calculations easier
    '''

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe=torch.zeros(seq_len, d_model)

        # creating two tensors for the final formula in log space for numerical stability
        # create a vector of shape(seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * - (math.log(10000.0) / d_model))

        # apply the formula in sine and cosine
        # sine is applied to even positions in matrix and cosine is applied to the odd positions in the matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # we will add batch dimension to the tensor so we can apply it to whole batch of sentences
        pe = pe.unsqueeze(0) # tensor of shape (1, seq_len, d_model)
        # now register this tensor as a buffer so it is not treated as a parameter and is not updated during backpropagation
        self.register_buffer('pe', pe)

    def forward(self, x):
        # we have to add positional encoding to every word in the sentence 
        # and we have to tell the model that it is not a parameter do not want to learn this positional encoding (as it is fixed)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    '''
        In transformers, this is applied inside 'Add and Norm' block
        deep stacks of transformer layers (many attention + FFN layers) can lead to exploding or vanishing gradients and make activations shift to left or right - Training becomes unstable, so we need to normalize the inputs
        Layer Norm re-centers and re-scales each vector so it has ~mean 0 and ~std 1, then lets the model learn how much to scale/shift back with alpha and bias.
    '''
        
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # learnable parameter - Multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # learnable parameter - Additive
    

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # usually mean cancels the dimension to which it is applied but we want to keep it so keepdim=True
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    '''
        used in both encoder and decoder layers    
        Feed Forward is a fully connected layer
        It is two matrices- W1 and W2, multipled by x and with ReLU in b/w and bias
        FFN(x)=max(0,xW1 + b1)W2 + b2
        W1 and W2 are matrices of shape (d_model, d_ff) and (d_ff, d_model) respectively
        b1 and b2 are vectors of shape (d_ff) and (d_model) respectively
    '''

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # x is a tensor of shape (batch, seq_len, d_model)
        # first we apply W1 and B1 to x and get a tensor of shape (batch, seq_len, d_ff)
        # then we apply dropout to the tensor
        # then we apply ReLU to the tensor
        # then we apply W2 and B2 to the tensor and get a tensor of shape (batch, seq_len, d_model)
        # return the tensor
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))    
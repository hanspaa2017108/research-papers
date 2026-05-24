import torch 
import torch.nn as nn
import math 

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # now we will create actual embeddings, pytorch provides a layer to do mapping b/w numbers
        # embeddings basically is mapping b/w numbers and their embedding vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # use embedding layer provided by pytorch
        return self.embedding(x) * math.sqrt(self.d_model)
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

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        # we have to divide the embedding vector into h heads, from the paper dk = d_model / h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        # define matrix with which we will multiple Q, K, V
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # W0 - Output Matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod

    def attention(query, key, value, mask, dropout=nn.Dropout):

        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_len, Seq_Len)
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask = None):
        # q, k, v are tensors of shape (batch, seq_len, d_model)
        # mask is a tensor of shape (batch, seq_len, seq_len), -infinity, if we want some words to not interact with other words
        # first we apply Wq, Wk, Wv to q, k, v and get a tensor of shape (batch, seq_len, d_model)
        # then we apply dropout to the tensor
        # then we apply the formula for multi-head attention
        # then we apply Wo to the tensor and get a tensor of shape (batch, seq_len, d_model)
        # return the tensor

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_k) --> (batch, h, seq_len, d_k), split into smaller matrices
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # let's calculate attention scores
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, dropout=self.dropout)

        # (Batch, h, Seq_Len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    '''
        Wraps every sublayer (attention or FFN) in the paper's "Add & Norm" pattern
        Formula: x + Dropout(Sublayer(LayerNorm(x)))
        - LayerNorm first (pre-norm), then sublayer, then dropout, then add back to x (residual/skip connection)
        - Residual connection lets gradients flow directly through the stack and makes deep training easier
        - sublayer is passed in as a function/lambda so the same wrapper works for attention and FFN
    '''

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # x shape: (batch, seq_len, d_model) — input to this sublayer
        # sublayer: callable that takes normalized x and returns (batch, seq_len, d_model)
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    '''
        One encoder layer from the paper — two sublayers stacked:
        1) Multi-Head Self-Attention (each token attends to all tokens in the source sentence)
        2) Position-wise Feed-Forward
        Each sublayer is wrapped in ResidualConnection (Add & Norm)
    '''

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # two residual wrappers: one for attention, one for FFN
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    # SRC_MASK -> used to hide the interaction of padding words with other words
    def forward(self, x, src_mask):
        # self-attention: Q, K, V all come from the same source x (encoder looks at itself)
        # lambda delays the call so ResidualConnection can run norm(x) first, then pass that into attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # feed-forward: no mask needed — processes each position independently
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    '''
        Full encoder stack: N repeated EncoderBlocks (paper uses N=6)
        Input: source token embeddings + positional encoding
        Output: contextual representations of the source sentence for the decoder to attend to
    '''

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization() # final norm after all encoder layers

    def forward(self, x, mask):
        # x shape: (batch, seq_len, d_model)
        # mask (src_mask): hide padding tokens so attention does not attend to <pad>
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    '''
        One decoder layer — three sublayers (encoder block has only two):
        1) Masked Multi-Head Self-Attention (decoder tokens only attend to earlier tokens — causal mask)
        2) Multi-Head Cross-Attention (decoder queries attend to encoder keys/values)
        3) Position-wise Feed-Forward
        Each wrapped in ResidualConnection
    '''

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # three residual wrappers: masked self-attn, cross-attn, FFN
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # masked self-attention: Q,K,V from decoder x; tgt_mask prevents looking at future words
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # cross-attention: Q from decoder x, K and V from encoder_output (bridge encoder → decoder)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # feed-forward on the mixed representation
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    '''
        Full decoder stack: N repeated DecoderBlocks (paper uses N=6)
        Input: target token embeddings + positional encoding
        Also receives encoder_output from the encoder stack
        Output: representations used to predict the next token in the sequence
    '''

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization() # final norm after all decoder layers

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x: decoder input embeddings (batch, tgt_seq_len, d_model)
        # encoder_output: output of Encoder (batch, src_seq_len, d_model)
        # src_mask: padding mask for encoder side (used in cross-attention)
        # tgt_mask: causal mask for decoder self-attention (no peeking at future tokens)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    '''
        Final step: map decoder output vectors to scores over the target vocabulary
        Each position gets a score per token in vocab — model picks the most likely next word
        log_softmax gives log-probabilities (works with NLLLoss / cross-entropy during training)
        Paper shares embedding and output weights; here we use a separate Linear (common in tutorials)
    '''

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # W: (d_model, vocab_size) + bias

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # proj: (batch, seq_len, vocab_size) — raw logits per vocabulary token at each position
        # log_softmax on last dim: convert logits to log-probabilities for loss / inference
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):

    '''
        Puts the full model together: encoder stack + decoder stack + embeddings + positional encoding + projection
        Training/inference flow:
            1) encode(source)  → encoder_output
            2) decode(target, encoder_output)  → decoder hidden states
            3) project(decoder output)  → log-probs over target vocabulary
        Source and target can have different vocab sizes and sequence lengths (e.g. translation)
    '''

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # token IDs → vectors for source language
        self.tgt_embed = tgt_embed # token IDs → vectors for target language
        self.src_pos = src_pos # add position info to source embeddings
        self.tgt_pos = tgt_pos # add position info to target embeddings
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # src: (batch, src_seq_len) — source token IDs
        # embed + positional encode, then run through encoder blocks
        src = self.src_embed(src) # (batch, src_seq_len, d_model)
        src = self.src_pos(src)
        return self.encoder(src, src_mask) # (batch, src_seq_len, d_model)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        # tgt: (batch, tgt_seq_len) — target token IDs (shifted right for teacher forcing in training)
        # encoder_output: memory from encode(); decoder attends to it in cross-attention
        tgt = self.tgt_embed(tgt) # (batch, tgt_seq_len, d_model)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask) # (batch, tgt_seq_len, d_model)

    def project(self, x):
        # map decoder output to vocabulary log-probabilities
        return self.projection_layer(x)

# Factory function: wire all hyperparameters into a complete Transformer (paper defaults: d_model=512, N=6, h=8, d_ff=2048, dropout=0.1)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:

    # --- Embeddings: separate tables for source and target (different languages → different vocabs) ---
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # --- Positional encoding: precompute sin/cos up to max seq length for src and tgt ---
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # --- Encoder: stack N identical EncoderBlocks (self-attn + FFN each) ---
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # --- Decoder: stack N DecoderBlocks (masked self-attn + cross-attn + FFN each) ---
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # separate weights from self-attn
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # wrap block lists in Encoder / Decoder modules (each adds a final LayerNorm)
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # --- Output head: decoder d_model vectors → logits over target vocabulary ---
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # --- Assemble full model ---
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Xavier init for weight matrices (dim > 1); biases and scalars keep default init
    # Paper-style training is more stable with good initialization for deep stacks
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
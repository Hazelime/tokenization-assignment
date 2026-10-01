import torch
from torch import nn

"""
Hyperparameters (using recommended values).
    n_layers = 2         # Number of Transformer layers
    C = 256             # Hidden dimension
    n_heads = 4         # Number of attention heads
    ff_dim = 1024       # Feed-forward dimension
    dropout_rate = 0.1 
    T = 128             # Context length

Legend for matrix comments below:
B = batch size
T = context length
C = number of hidden dimensions
V = vocabulary size
"""


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of multi-head self-attention and a feedforward
    neural network, with residual connections, causal masking, and layer pre-normalization. 

    Parameters:
        C: int
            The number of hidden dimensions (embedding size).
        n_heads: int
            The number of attention heads in the multi-head self-attention.
        ff_dim: int
            The dimension of the feedforward neural network.
        dropout_rate: float
            The dropout rate for regularization in the feedforward network.
    """
    def __init__(self,
                 C: int = 256,
                 n_heads: int = 4,
                 ff_dim: int = 1024,
                 dropout_rate: float = 0.1):
        # Inherit all the fancy PyTorch magic.
        super(TransformerBlock, self).__init__()

        # First normalization layer (for input). (B, T, C) -> (B, T, C)
        # The fact that normalization comes -before- each main operation makes this is
        # a pre-norm architecture, which is now the standard. The original 2017
        # Transformer paper did post-normalization though.
        self.norm1 = nn.LayerNorm(C)

        # Multi-head self-attention where the K, Q, V matrices mix. (B, T, C) -> (B, T, C)
        # Batch_first = True means that the data comes in the order (batch, seq) rather
        # than vice versa.
        self.multihead_attention = nn.MultiheadAttention(C,
                                                         n_heads,
                                                         batch_first = True)
        
        # Second normalization layer (for attention output). (B, T, C) -> (B, T, C)
        self.norm2 = nn.LayerNorm(C)

        # Feedforward neural network (multilayer perceptron). (B, T, C) -> (B, T, C)
        # GELU (= Gaussian Error Linear Unit) is the standard non-linear activation function.
        # Dropout reduces overfitting and overreliance on strong neurons by randomly setting
        # some activations to zero each forward pass (e.g. 10% of all neurons if dropout_rate
        # is 0.1).
        self.ff = nn.Sequential(
            nn.Linear(C, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, C),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Creates a mask from the upper-right triangular ("triu") part of a T x T matrix
        # of 1's, since 1 = True = masked in the multihead attention layer. This mask 
        # will remove access to future context. See attn_mask in
        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html
        # Diagonal = 1 determines the size/placement of the triangle. See examples in
        # https://docs.pytorch.org/docs/2.14/generated/torch.triu.html
        B, T, C = x.shape
        mask = torch.triu(torch.ones(T, T, device = x.device),
                          diagonal = 1).bool()

        # Pre-normalization 1. (B, T, C) -> (B, T, C)
        x_norm = self.norm1(x)

        # Pass through multi-head attention layer and add the residual x.
        # Use [0] to get only the output (not the weights, which is [1]).
        # (B, T, C) -> (B, T, C)
        x = x + self.multihead_attention(query = x_norm,
                                         key = x_norm,
                                         value = x_norm,
                                         attn_mask = mask)[0]

        # Pre-normalization 2. (B, T, C) -> (B, T, C)
        x_norm = self.norm2(x)

        # Apply feedforward neural network and add the residual x.
        # (B, T, C) -> (B, T, C)
        return x + self.ff(x_norm)


class DecoderTransformer(nn.Module):
    """
    A decoder-only Transformer model for autoregressive language modeling. It consists of
    token embeddings, positional embeddings, a series of Transformer blocks, and an output
    layer that maps to the vocabulary size.

    Parameters:
        V: int
            The vocabulary size (number of unique tokens).
        C: int
            The number of hidden dimensions (embedding size).
        n_layers: int
            The number of Transformer blocks.
        n_heads: int
            The number of attention heads in each Transformer block.
        ff_dim: int
            The dimension of the feedforward neural network in each Transformer block. 
    """
    def __init__(self,
                 V: int,
                 C: int = 256,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 ff_dim: int = 1024,
                 dropout_rate: float = 0.1,
                 T: int = 256):
        # Inherit all the fancy PyTorch magic.
        super(DecoderTransformer, self).__init__()
        
        # Token embeddings. (B, T) -> (B, T, C)
        self.token_embeddings = nn.Embedding(V, C)

        # Positional embeddings. (T) -> (T, C)
        self.position_embeddings = nn.Embedding(T, C) 

        # A series of n_layer transformer blocks. (B, T, C) -> (B, T, C)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(C, n_heads, ff_dim, dropout_rate)
            for _ in range(n_layers)
        ])

        # Final normalization (since we're doing pre-norm in the transformer blocks.)
        self.final_norm = nn.LayerNorm(C)

        # Unembedding/output layer (B, T, C) -> (B, T, V)
        self.output = nn.Linear(C, V)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token embeddings. (B, T) -> (B, T, C)
        token_embeddings = self.token_embeddings(x)
        
        # Position embeddings. (T) -> (T, C)
        B, T = x.shape
        # Create a list of positions from 0 to T-1. Arange creates a new tensor, so we need
        # to put it on the same device as x.
        positions = torch.arange(T, device = x.device)
        position_embeddings = self.position_embeddings(positions)
        
        # Add positional embeddings to token embeddings.
        # (B, T, C) + (T, C) -> (B, T, C)
        x = token_embeddings + position_embeddings

        # Pass through transformer blocks. (B, T, C) -> (B, T, C)
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final normalization.
        x = self.final_norm(x)
        
        # Output predictions. (B, T, C) -> (B, T, V)
        return self.output(x)
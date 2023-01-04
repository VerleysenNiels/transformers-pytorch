"""
The Encoder part of a transformer exists of one or multiple transformer blocks.
The output of the previous block is used as keys, values and queries for the next.
"""

# Imports
import torch
import torch.nn as nn

from transformer_block import TransformerBlock

class Encoder(nn.Module):
    def __init__(self, nr_blocks, embedding_size, nr_heads, forward_expansion, dropout, nr_tokens, max_seq_length, device):
        super(Encoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.device = device
        
        # Inputs are first embedded and then positionally encoded
        self.word_embedding = nn.Embedding(nr_tokens, embedding_size)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        # Repetition of transformer blocks
        transformer_blocks = []
        for _ in range(nr_blocks):
            transformer_blocks.append(TransformerBlock(embedding_size, nr_heads, dropout, forward_expansion))
            
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        
    def forward(self, x, mask):
        batch_size, sequence_length = x.shape
        
        # Prepare position embedding
        positions = torch.arange(0, sequence_length).expand(batch_size, sequence_length)
        positions = positions.to(self.device)
        
        # Input embedding
        embedded = self.word_embedding(x)
        
        # Positional encoding
        positions = self.position_embedding(positions)
        embedded = embedded + positions
        
        # Dropout on embeddings
        out = self.dropout(embedded)
        
        # Finally we have a number of transformer blocks
        for transformer_block in self.transformer_blocks:
            out = transformer_block(out, out, out, mask)
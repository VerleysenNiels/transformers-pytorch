"""
The decoder part exists of a repitition of decoder blocks, each of which exist of a masked multi-head attention and a transformer block.
This is the implementation of this block.
"""

# Imports
import torch
import torch.nn as nn

from transformer_block import TransformerBlock
from multi_head_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, nr_heads, dropout, forward_expansion, device):
        super(DecoderBlock, self).__init__()
        
        self.attention = MultiHeadAttention(embedding_size, nr_heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.transformer_block = TransformerBlock(embedding_size, nr_heads, dropout, forward_expansion)
        
    def forward(self, x, values, keys, source_mask, target_mask):
        # Source mask is meant to mask out padding in the input
        
        # Multi-Head Attention
        attended = self.attention(x, x, x, target_mask)
        
        # Skip connection
        queries = self.norm(attended + x)
        queries = self.dropout(queries)
        
        # Transformer block
        return self.transformer_block(values, keys, queries, source_mask)   
        
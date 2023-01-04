"""
Implementation of a single transformer block
A single transformer block exists of:
    - Multi-Head Attention
    - Skip connection with Addition and Normalization
    - Feed Forward block
    - Skip connection with Addition and Normalization
"""

# Imports
import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, nr_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(embedding_size, nr_heads)
        
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion*embedding_size)
            nn.ReLU()
            nn.Linear(forward_expansion*embedding_size, embedding_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, values, keys, queries, mask):
        # Multi-Head Attention
        attention = self.attention(values, keys, queries, mask)
        
        # Skip connection
        x = self.norm1(attention + queries)
        x = self.dropout(x)
        
        # Feed Forward block
        forward = self.feed_forward(x)
        
        # Skip connection
        out = self.norm2(forward + x)
        out = self.dropout(out)
        
        return out
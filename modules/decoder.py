"""
The decoder part exists of a repitition of decoder blocks, each of which exist of a masked multi-head attention and a transformer block.
This is the implementation of the decoder. (this is very similar to the encoder)
"""

# Imports
import torch
import torch.nn as nn

from .decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, nr_blocks, embedding_size, nr_heads, forward_expansion, dropout, nr_tokens, max_seq_length, device):
        super(Decoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.device = device
        
        # Inputs are first embedded and then positionally encoded
        self.word_embedding = nn.Embedding(nr_tokens, embedding_size)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_size)        
        self.dropout = nn.Dropout(dropout)
        
        # Repetition of decoder blocks
        decoder_blocks = []
        for _ in range(nr_blocks):
            decoder_blocks.append(DecoderBlock(embedding_size, nr_heads, dropout, forward_expansion))
            
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        
        # Head of the model
        self.feed_forward_out = nn.Linear(embedding_size, nr_tokens)
        
    def forward(self, x, encoder_output, source_mask, target_mask):
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
        
        # We then have a number of decoder blocks
        for decoder in self.decoder_blocks:
            out = decoder(out, encoder_output, encoder_output, source_mask, target_mask)
        
        # After the decoder blocks we have the head of the model, which in the published research is a single linear layer with softmax activation    
        out = self.feed_forward_out(out)
        out = torch.softmax(out, dim=2)
        return out
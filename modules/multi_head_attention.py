# This module contains the implementation of the Multi-Head Attention (both masked and unmasked)

# Imports
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, nr_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.nr_heads = nr_heads
        self.head_dim = embedding_size // nr_heads
        
        if not self.embedding_size % self.nr_heads == 0:
            raise ValueError("Embedding size is not divisible by the number of heads!")
        
        # The values, keys and queries are first each passed through their respective linear layers.
        # In theory we have these layers for each head, but by increasing the input and output size we can combine them.
        self.values_lin = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.keys_lin = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.queries_lin = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        
        # After the scaled dot-product attention the outputs of all the heads are concatenated and passed through a single linear layer.    
        self.unify_out = nn.Linear(self.embedding_size, self.embedding_size)
            
    def forward(self, values, keys, query, mask):
        # Get dimensions of the input
        batch_size = values.shape[0]
        nr_values, nr_keys, nr_queries = values.shape[1], keys.shape[1], query.shape[1]
        
        # Pass V, K and Q through the linear layers 
        # In theory we need to have separate linear layers per head here
        # This can also be accomplished with a single (larger) layer, making implementation more straightforward
        values = self.values_lin(values)
        keys = self.keys_lin(keys)
        queries = self.queries_lin(queries)         
        
        # We then need to reshape these to split them in the different heads for further computations
        values = values.view(batch_size, nr_values, self.nr_heads, self.head_dim)
        keys = keys.view(batch_size, nr_keys, self.nr_heads, self.head_dim)
        queries = queries.view(batch_size, nr_queries, self.nr_heads, self.head_dim)
        
        # We now need to perform scaled dot-product attention
        # First the keys and queries need to be multiplied, we can use einsum for this
        # For more information on einsum I recommend the following video: https://www.youtube.com/watch?v=pkVwUVEHmfI
        x = torch.einsum("bqhd,bkhd->nhqk", [queries, keys])
        
        # Optional masking
        if mask is not None:
            x = x.masked_fill(mask == 0, float("-1e20"))
        
        # Apply softmax
        x = torch.softmax(x / (self.embedding_size ** 0.5), dim=3)
        
        # Final multiplication
        # shape of x is still (batch size, nr heads, nr queries, nr keys)
        x = torch.einsum("bhql,blhd->bqhd", [x, values])
        
        # Next all the outputs of the heads need to be concatenated
        x = x.view(batch_size, nr_queries, self.embedding_size)
        
        # And passed through the final linear layer
        return self.unify_out(x)
"""
This class contains my PyTorch implementation of the transformer architecture as presented in the "Attention is all you need" paper.
I have built up the encoder and decoder classes in the modules folder.
"""

# Imports
import logging

import torch
import torch.nn as nn

from modules.encoder import Encoder
from modules.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, source_padding_index, target_padding_index, nr_blocks=6, embedding_size=512, nr_heads=8, forward_expansion=4, dropout=0.2, max_seq_length=128, device="cuda"):
        """
        PyTorch implementation of the transformer architecture presented in the paper "Attention is all you need".
        
        Parameters:
            source_vocabulary_size (int): Size of the vocabulary of the source language.
            target_vocabulary_size (int): Size of the vocabulary of the target language.
            source_padding_index (int): Index of the padding symbol in the source language.
            target_padding_index (int): Index of the padding symbol in the target language.
            nr_blocks (int, optional): Number of transformer blocks to use. Default is 6.
            embedding_size (int, optional): Size of the embeddings. Default is 512.
            nr_heads (int, optional): Number of attention heads. Default is 8.
            forward_expansion (int, optional): Size of the feedforward layer. Default is 4.
            dropout (float, optional): Dropout rate. Default is 0.2.
            max_seq_length (int, optional): Maximum sequence length. Default is 128.
            device (str, optional): Device to use for computations. Default is 'cuda'.
        """
        super(Transformer, self).__init__()
        
        # The transformer uses an encoder-decoder architecture. Both have been implemented in the modules folder.
        self.encoder = Encoder(nr_blocks, embedding_size, nr_heads, forward_expansion, dropout, source_vocabulary_size, max_seq_length, device)
        self.decoder = Decoder(nr_blocks, embedding_size, nr_heads, forward_expansion, dropout, target_vocabulary_size, max_seq_length, device)
        
        self.source_padding_index = source_padding_index
        self.target_padding_index = target_padding_index
        self.device = device
        
    def create_source_mask(self, source):
        """
        Create the mask for samples from the source language.
        
        Parameters:
            source (torch.Tensor): Tensor of shape (batch_size, source_seq_length) representing examples from  the source language.
        
        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1, 1, source_seq_length) representing the mask.
        """
        source_mask = (source != self.source_padding_index).unsqueeze(1).unsqueeze(2)
        return source_mask.to(self.device)
    
    def create_target_mask(self, target):
        """
        Create the mask for samples from the target language.
        
        Parameters:
            target (torch.Tensor): Tensor of shape (batch_size, target_seq_length) representing examples from the target language.
        
        Returns:
            torch.Tensor: Tensor of shape (batch_size, target_seq_length, target_seq_length) representing the mask.
        """
        batch_size, target_length = target.shape
        # We need to mask out padding in the target samples
        padding_mask = (target != self.target_padding_index).unsqueeze(1).unsqueeze(2).to(self.device)
        # We need to mask out future output tokens in order to prevent leaking.
        target_mask = torch.tril(torch.ones((target_length, target_length))).expand(batch_size, 1, target_length, target_length).to(self.device)
        # Combine the two masks
        target_mask = torch.minimum(padding_mask, target_mask)
        
        return target_mask
    
    def forward(self, source, target):
        """
        Forward pass through the transformer.
        
        Parameters:
            source (torch.Tensor): Tensor of shape (batch_size, source_seq_length) representing examples from the source language.
            target (torch.Tensor): Tensor of shape (batch_size, target_seq_length) representing examples from the target language.
        
        Returns:
            torch.Tensor: Tensor of shape (batch_size, target_seq_length, target_vocab_size) representing the output logits.
        """
        
        # First we create the masks
        source_mask = self.create_source_mask(source)
        target_mask = self.create_target_mask(target)
        
        # Pass through the encoder and decoder
        encoded = self.encoder(source, source_mask)
        out = self.decoder(target, encoded, source_mask, target_mask)
        
        return out
    
    
if __name__ == "__main__":
    """
    Here I test the transformer in a short example using dummy embeddings of some text.
    In practice a transformer needs to be trained on a huge amount of data as it starts of without knowing anything about the relationships between words.
    Of course that is very time-consuming and expensive to do. When applying transformers in practice it is therefore a lot more efficient to start from a pretrained model and
    then finetuning it on your dataset.
    
    Here I create a dummy dataset with a vocabulary size of 10 (numbers 0-9). Then train the model on it for 5 epochs and test it on the same data again just to see if it works.
    Please don't test on training data in actual projects ;)
    """
    # Set logging level to info
    logging.basicConfig(level=logging.INFO)
    
    # Check if cuda is available as device    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on {device}")    
    
    # Dummy input data
    source = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0], [1, 2, 0, 0, 0], [2, 3, 4, 5, 6]]).to(device)
    target = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).to(device)

    # Dummy input sizes
    source_vocabulary_size = 10
    target_vocabulary_size = 10

    # Dummy padding indices
    source_padding_index = 0
    target_padding_index = 0

    # Initialize transformer model
    model = Transformer(source_vocabulary_size, target_vocabulary_size, source_padding_index, target_padding_index, device=device).to(device)

    # Set model to training mode
    model.train()

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for epoch in range(5):
        # Forward pass
        logits = model(source, target)
        loss = loss_function(logits.view(-1, target_vocabulary_size), target.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Zero out gradients
        optimizer.zero_grad()
        
        # Print loss at each epoch
        logging.info(f'Epoch {epoch+1}: Loss = {loss.item()}')

    # Set model to eval mode
    model.eval()

    # Evaluation loop
    with torch.no_grad():
        logits = model(source, target)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == target).float().mean()
        logging.info(f'Accuracy: {accuracy}')
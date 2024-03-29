# Implementing a transformer with pytorch
There is no other way to turn it, transformers are the de-facto standard now in most deep learning tasks. Especially in language, computer vision and control tasks they are dominating the benchmark leaderboards. Naturally it is therefore very interesting to know how they work, and what better way to do so than to implement them yourself? Let's go!

Note: if you are not familiar with deep learning yet, I would recommend to do that first. Check out [my repository and blogs](https://github.com/VerleysenNiels/Deep-learning-101) to learn this with a hands on approach.

## Theoretical background
Transformers have an encoder-decoder architecture. The encoder is built up from one or multiple encoder blocks (left on the picture) and decoder blocks are used to build the decoder (right on the picture).

![image](https://user-images.githubusercontent.com/26146888/209946098-ec889a6c-c939-4781-a507-82d927933740.png)
### Encoder block
The encoder block mainly consists of two layers: a multi-head attention layer and a feedforward layer. The outputs of these layers are added with the input of the layer through a skip connection and are normalized. So this is quite simple, once we understand the multi-head attention.

### Multi-Head Attention
From a high level, attention performs a mapping of a query and a set of key-value pairs to an output. It figures out for each token (identified by a key) what other tokens are related to it and how important they are in that regard (values). Keys, values and queries are all packed in vectors and passed together to perform computations in parallel. Each of these vectors first pass through a linear layer before performing scaled dot-product attention. In scaled dot-product attention the keys and queries are first multiplied, then scaled, optionally masked and passed through a softmax activation function. This output is then multiplied with the values. This is a single head, so we stack multiple of these together (giving us multi-head attention).

The outputs coming from all these heads are concatenated together before being passed through a final linear layer. This gives us the output of the multi-head attention layer (or block depending on how you see it).

![image](https://user-images.githubusercontent.com/26146888/209950910-a37be4f9-6774-4c0a-9dbd-c012817bd82a.png)

### Decoder block
If we look closely at the architecture of the decoder block, we can see that it is almost completely the same as the encoder block. The main difference being the inputs to the multi-head attention layer. The keys and values are coming from the output of the encoder block. While the query comes from a masked multi-head attention layer, also with a skip connection.

### Masked Multi-Head Attention
This is the same as multi-head attention, with the addition of a mask. Why do we need a mask here? Well, the transformer can do all computations in parallel while training. The decoder gets the outputs as input and this way we get leakage as future target tokens are given as input. The model can then learn to simply use these instead of predicting anything by itself. To prevent this we have to mask the output embeddings so future target tokens cannot be used. 

## Environment requirements
You can find the environment I'm using for this project in the environment.yml file. You can use this file with conda to recreate this exact environment, just make sure to update the environment prefix. The environment is based on python 3.10 (as 3.11 is not available in conda yet). In terms of libraries I'm currently only using pytorch with CUDA (11.7) and standard libraries like numpy. So an alternative option could be to recreate this environment yourself with python 3.10 and installing a version of pytorch that works on your system.

## Usage
You can simply use the transformer class as a regular pytorch model. The only important difference is that you need to create an embedding of your input and output examples, for this you can always use existing language embedding functionalities from libraries like spaCy. You also need to pass through the vocabulary size and padding index to the model. I added an example with dummy data to the transformer.py file to show how.

```python
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
model = Transformer(source_vocabulary_size, target_vocabulary_size,
      source_padding_index, target_padding_index, device=device).to(device)

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
```

## Reading material
Before you can implement, you have to of course read up on the technology itself. For starters I would really recommend you to read the following three papers at least:
- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): This paper basically started it all. By introducing a new language model that only relied on attention, the transformer architecture was brought to life. 
- [An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/pdf/2010.11929.pdf): What if we applied the same transformer architecture to vision tasks? Seeing as how transformers are now the de-factor standard in vision tasks, very good things I would say. This paper brought transformers to the world of computer vision.
- [How to train your ViT? Data, Augmentation and Regularization in Vision Transformers](https://arxiv.org/pdf/2106.10270.pdf): Vision transformers have very weak inductive bias, making them perform way worse than CNNs on smaller datasets. This paper contains valuable insights on how to train a vision transformer. Vision transformers start outperforming CNNs when the dataset becomes larger. Using data augmentation techniques evidently boosts the performance as well. For AI practitioners wanting to apply vision transformers for a certain task, starting from the best pretrained model and finetuning it will be the best approach. This way the training cost remains quite low and the performance will be very high.

Then if you still want to read more, you can dive into specific topics. For instance to learn more about the application to specific tasks, the latest architectures or to get more insights in what the transformers are actually learning. There is a lot of stuff out there, practically impossible to read everything so pick whatever you find interesting or could be useful for your research/project. Good luck!

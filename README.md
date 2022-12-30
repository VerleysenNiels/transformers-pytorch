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
TODO

## Usage
TODO

## Reading material
Before you can implement, you have to of course read up on the technology itself. For starters I would really recommend you to read the following three papers at least:
- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): This paper basically started it all. By introducing a new language model that only relied on attention, the transformer architecture was brought to life. 
- [An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/pdf/2010.11929.pdf): What if we applied the same transformer architecture to vision tasks? Seeing as how transformers are now the de-factor standard in vision tasks, very good things I would say. This paper brought transformers to the world of computer vision.
- [How to train your ViT? Data, Augmentation and Regularization in Vision Transformers](https://arxiv.org/pdf/2106.10270.pdf): Vision transformers have very weak inductive bias, making them perform way worse than CNNs on smaller datasets. This paper contains valuable insights on how to train a vision transformer. Vision transformers start outperforming CNNs when the dataset becomes larger. Using data augmentation techniques evidently boosts the performance as well. For AI practitioners wanting to apply vision transformers for a certain task, starting from the best pretrained model and finetuning it will be the best approach. This way the training cost remains quite low and the performance will be very high.

Then if you still want to read more, you can dive into specific topics. For instance to learn more about the application to specific tasks, the latest architectures or to get more insights in what the transformers are actually learning. There is a lot of stuff out there, practically impossible to read everything so pick whatever you find interesting or could be useful for your research/project.

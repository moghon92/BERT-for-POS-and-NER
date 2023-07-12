## Classification with BERT

The transformer neural network is a novel architecture that aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease. In a transformer, we can pass all the words of a sentence and determine the word embedding simultaneously.

<p align="center"><img src="https://d2l.ai/_images/bert-one-seq.svg" width="75%" align="center"></p>

We will be using BERT (Bidirectional Encoder Representations from Transformers) pre-trained models for embeddings. BERT architecture consists of several Transformer encoders stacked together. Each Transformer encoder encapsulates two sub-layers: a self-attention layer and a feed-forward layer. 

The details on BERT can be referred from the paper : [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf).

We will be using the BERT embeddings and a fully connected linear layer to perform classification.

We will then classify the Clickbait and Web of science dataset for this task.

## Sequence Labeling

Part-of-speech (POS) tagging is a popular Natural Language Processing process which refers to categorizing words in a text (corpus) in correspondence with a particular part of speech, depending on the definition of the word and its context.

Named entity recognition (NER) seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.

For more details on each tasks, please refer to class slides.

<p align="center"><img src="https://media-exp1.licdn.com/dms/image/C5112AQGVAByeLRJlBw/article-inline_image-shrink_400_744/0/1579118062060?e=1674691200&v=beta&t=lpQUVXCxwj-GYb3R_Kz_ys6BB-cgZYgOurOdniGPyrU" width="75%" align="center"></p>

We will be using BERT (Bidirectional Encoder Representations from Transformers) for sequence labeling. The architecture of the model is shown above in the diagram.

We will be using the BERT embeddings and a fully connected linear layer to perform classification.

We will then classify using the conll2003 dataset for this task.

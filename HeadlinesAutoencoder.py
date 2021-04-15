# -*- coding: utf-8 -*-
"""

Much of the idea behind this is movtivated by Shen et al [1].
(the data augmentation rules proposed in that work are used to improve
the robustness of the autoencoder)

[1] Shen et al (2019) "Educating Text Autoencoders: Latent Representation Guidance via Denoising" https://arxiv.org/pdf/1905.12777.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import random


from google.colab import drive
drive.mount('/content/gdrive')

train_path = '/content/gdrive/My Drive/CSC413/A3/reuters_train.txt' # Update me
valid_path = '/content/gdrive/My Drive/CSC413/A3/reuters_valid.txt' # Update me

"""Using PyTorch's `torchtext` utilities to help us load, process,
and batch the data. 

Using a `TabularDataset` to load our data, which works well on structured
CSV data with fixed columns (e.g. a column for the sequence, a column for the label). The tabular dataset
is even simpler: we have no labels, just some text. Treating the data as a table with one field
representing our sequence.
"""

import torchtext
from torchtext.legacy import data

# Tokenization function to separate a headline into words
def tokenize_headline(headline):
    """Returns the sequence of words in the string headline. We also
    prepend the "<bos>" or beginning-of-string token, and append the
    "<eos>" or end-of-string token to the headline.
    """
    return ("<bos> " + headline + " <eos>").split()

# Data field (column) representing our *text*.
text_field = torchtext.legacy.data.Field(
    sequential=True,            # this field consists of a sequence
    tokenize=tokenize_headline, # how to split sequences into words
    include_lengths=True,       # to track the length of sequences, for batching
    batch_first=True,           # similar to batch_first=True in nn.RNN demonstrated in lecture
    use_vocab=True)             # to turn each character into an integer index
train_data = torchtext.legacy.data.TabularDataset(
    path=train_path,                # data file path
    format="tsv",                   # fields are separated by a tab
    fields=[('title', text_field)]) # list of fields (we have only one)



# Draw histograms of the number of words per headline in our training set.
word_list = []

for headline in train_data:
  headline = headline.title
  num_words = 0
  for word in headline:
    if (word != '<bos>') or(word != '<eos>'):
      num_words += 1
  word_list.append(num_words)

plt.hist(word_list)
plt.xlabel("Number of words per headline")
plt.ylabel("Number of headlines")
plt.show()

"""
We'd be interested in this histogram because we gives information about the 
distribution of the number of words in a headline. This important because we
can use this information to help our model determine the appropriate length 
of a headline as it attempts to generate a new one.
"""


from collections import Counter

count = {}
for headline in train_data:
  for word in headline.title:
    if (word != '<bos>') and (word != '<eos>'):
      if word in count:
        count[word] += 1
      else:
        count[word] = 1
print("The number of distinct words in the training data is", len(count))

"""
The distribution of *words* will have a long tail, meaning that there are some words
that will appear very often, and many words that will appear infrequently. How many words
appear exactly once in the training set? Exactly twice?
"""

single_appearances = 0
double_appearances = 0

for key in count:
  if count[key] == 1:
    single_appearances += 1
  elif count[key] == 2:
    double_appearances += 1

print("The number of words that appear exactly once is", single_appearances)#, " and these words are ",single_words)
print("The number of words that appear exactly twice is", double_appearances)#, "and these words are ", double_words)


dict_list = list(count.values())
dict_list.sort(reverse=True)

top_occurence_sum = sum(dict_list[:9995])
occurences = sum(dict_list)

percentage = (top_occurence_sum/occurences)
alternative_percentage = (occurences - top_occurence_sum)/occurences

print("The percentage of words occurences that will be supported is", percentage)
print("The percentage of word occurences in the training set that will be set to the <unk> tag is:", alternative_percentage)


# Build the vocabulary based on the training data. The vocabulary
# can have at most 9997 words (9995 words + the <bos> and <eos> token)
text_field.build_vocab(train_data, max_size=9997)

# This vocabulary object will be helpful for us
vocab = text_field.vocab
print(vocab.stoi["hello"]) # for instances, we can convert from string to (unique) index
print(vocab.itos[10])      # ... and from word index to string

# The size of our vocabulary is actually 10000
vocab_size = len(text_field.vocab.stoi)
print(vocab_size) # should be 10000

# The reason is that torchtext adds two more tokens for us:
print(vocab.itos[0]) # <unk> represents an unknown word not in our vocabulary
print(vocab.itos[1]) # <pad> will be used to pad short sequences for batching


class AutoEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        """
        A text autoencoder. The parameters 
            - vocab_size: number of unique words/tokens in the vocabulary
            - emb_size: size of the word embeddings $x^{(t)}$
            - hidden_size: size of the hidden states in both the
                           encoder RNN ($h^{(t)}$) and the
                           decoder RNN ($m^{(t)}$)
        """
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, 
                                  embedding_dim=emb_size)  

        self.encoder_rnn = nn.GRU(input_size=emb_size, 
                                  hidden_size=hidden_size, 
                                  batch_first=True)
        
        self.decoder_rnn = nn.GRU(input_size=emb_size, 
                                  hidden_size=hidden_size, 
                                  batch_first=True)
                                  
        self.proj = nn.Linear(in_features=hidden_size, 
                              out_features=vocab_size) 

    def encode(self, inp):
        """
        Computes the encoder output given a sequence of words.
        """
        emb = self.embed(inp)
        out, last_hidden = self.encoder_rnn(emb)
        return last_hidden

    def decode(self, inp, hidden=None):
        """
        Computes the decoder output given a sequence of words, and
        (optionally) an initial hidden state.
        """
        emb = self.embed(inp)
        out, last_hidden = self.decoder_rnn(emb, hidden)
        out_seq = self.proj(out)
        return out_seq, last_hidden

    def forward(self, inp):
        """
        Computes both the encoder and decoder forward pass
        given an integer input sequence inp with shape [batch_size, seq_length],
        with inp[a,b] representing the (index in our vocabulary of) the b-th word
        of the a-th training example.

        This function returns the logits $z^{(t)}$ in a tensor of shape
        [batch_size, seq_length - 1, vocab_size], computed using *teaching forcing*.
        """

        hidden = self.encode(inp)
        inp2 = list(inp)
        
        inp = torch.narrow(inp,1,0,len(inp2[0])-1)
        out, hidden_2 = self.decode(inp, hidden)

        return out

model = AutoEncoder(vocab_size, 128, 128)
headline = train_data[24].title
input_seq = torch.Tensor([vocab.stoi[w] for w in headline]).long().unsqueeze(0)
model.forward(input_seq)

"""
Train our AutoEncoderneural network for at least 300 iterations to memorize this sequence, check that it is set up correctly
"""

headline = train_data[42].title 
input_seq = torch.Tensor([vocab.stoi[w] for w in headline]).long().unsqueeze(0)

"""
Note that the Cross Entropy Loss expects a rank-2 tensor as its first
argument, and a rank-1 tensor as its second argument. Hence
need to properly reshape the data to be able to compute the loss.
"""

model = AutoEncoder(vocab_size, 128, 128)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for it in range(300):

    output = model(input_seq)

    target = input_seq[:,1:] #target length is now len(sequence) - 1

    output2 = list(output)
    batch_size = len(output2)
    sequence_len = len(output2[0])

    loss = criterion(output.reshape(batch_size*sequence_len,vocab_size),target.reshape(-1)) #target.reshape(-1) is flattening tensor
    loss.backward()
    optimizer.step()

    if (it+1) % 50 == 0:
        print("[Iter %d] Loss %f" % (it+1, float(loss)))



def sample_sequence(model, hidden, max_len=20, temperature=1):
    """
    Return a sequence generated from the model's decoder
        - model: an instance of the AutoEncoder model
        - hidden: a hidden state (e.g. computed by the encoder)
        - max_len: the maximum length of the generated sequence
        - temperature: described in Part (d)
    """
    # We'll store our generated sequence here
    generated_sequence = []
    # Set input to the <BOS> token
    inp = torch.Tensor([text_field.vocab.stoi["<bos>"]]).long()
    for p in range(max_len):
        # compute the output and next hidden unit
        output, hidden = model.decode(inp.unsqueeze(0), hidden)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = int(torch.multinomial(output_dist, 1)[0])
        # Add predicted word to string and use as next input
        word = text_field.vocab.itos[top_i]
        # Break early if we reach <eos>
        if word == "<eos>":
            break
        generated_sequence.append(word)
        inp = torch.Tensor([top_i]).long()
    return generated_sequence




"""
In general, we don't want the temperature setting to be too large/high
because this will cause our model's generated sequences of text to have
more variation, however, the quality of these sequences will be lower.
This is because increasing temperature increases the model's sensitivity
to vocabulary with lower probability.

"""
#testing different temps
hidden = model.encode(input_seq) 
temps = [1.5,4,10]
for i in temps:
  for j in range(5):
    print(sample_sequence(model,hidden,max_len=20,temperature=i))



def tokenize_and_randomize(headline,
                           drop_prob=0.1,  # probability of dropping a word
                           blank_prob=0.1, # probability of "blanking" out a word
                           sub_prob=0.1,   # probability of substituting a word with a random one
                           shuffle_dist=3): # maximum distance to shuffle a word
    """
    Adding 'noise' to a headline by slightly shuffling the word order,
    dropping some words, blanking out some words (replacing with the <pad> token)
    and substituting some words with random ones.
    """
    headline = [vocab.stoi[w] for w in headline.split()]
    n = len(headline)
    # shuffle
    headline = [headline[i] for i in get_shuffle_index(n, shuffle_dist)]

    new_headline = [vocab.stoi['<bos>']]
    for w in headline:
        if random.random() < drop_prob:
            # drop the word
            pass
        elif random.random() < blank_prob:
            # replace with blank word
            new_headline.append(vocab.stoi["<pad>"])
        elif random.random() < sub_prob:
            # substitute word with another word
            new_headline.append(random.randint(0, vocab_size - 1))
        else:
            # keep the original word
            new_headline.append(w)
    new_headline.append(vocab.stoi['<eos>'])
    return new_headline

def get_shuffle_index(n, max_shuffle_distance):
    """ This is a helper function used to shuffle a headline with n words,
    where each word is moved at most max_shuffle_distance. The function does
    the following: 
       1. start with the *unshuffled* index of each word, which
          is just the values [0, 1, 2, ..., n]
       2. perturb these "index" values by a random floating-point value between
          [0, max_shuffle_distance]
       3. use the sorted position of these values as our new index
    """
    index = np.arange(n)
    perturbed_index = index + np.random.rand(n) * 3
    new_index = sorted(enumerate(perturbed_index), key=lambda x: x[1])
    return [index for (index, pert) in new_index]




def train_autoencoder(model, batch_size=64, learning_rate=0.001, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for ep in range(num_epochs):
        # We will perform data augmentation by re-reading the input each time
        field = torchtext.legacy.data.Field(sequential=True,
                                     tokenize=tokenize_and_randomize, # <-- data augmentation
                                     include_lengths=True,
                                     batch_first=True,
                                     use_vocab=False, # <-- the tokenization function replaces this
                                     pad_token=vocab.stoi['<pad>'])
        dataset = torchtext.legacy.data.TabularDataset(train_path, "tsv", [('title', field)])

        # This BucketIterator will handle padding of sequences that are not of the same length
        train_iter = torchtext.legacy.data.BucketIterator(dataset,
                                                   batch_size=batch_size,
                                                   sort_key=lambda x: len(x.title), # to minimize padding
                                                   repeat=False)
        for it, ((xs, lengths), _) in enumerate(train_iter):

            # Fill in the training code here
            output = model(xs)
            
            target = xs[:,1:] #target length is now len(sequence) - 1
            
            output2 = list(output)
            batch_size = len(output2)
            sequence_len = len(output2[0])
            
            loss = criterion(output.reshape(batch_size*sequence_len,vocab_size),target.reshape(-1)) #target.reshape(-1) is flattening tensor
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            if (it+1) % 100 == 0:
                print("[Iter %d] Loss %f" % (it+1, float(loss)))

        # Optional: Compute and track validation loss
        #val_loss = 0
        #val_n = 0
        #for it, ((xs, lengths), _) in enumerate(valid_iter):
        #    zs = model(xs)
        #    loss = None # TODO
        #    val_loss += float(loss)

# training loss is trending down
model = AutoEncoder(vocab_size,128,128)
train_autoencoder(model, num_epochs=1)


model = AutoEncoder(10000, 128, 128)
checkpoint_path = '/content/gdrive/My Drive/CSC413/A3/p4model.pk' # Update me
model.load_state_dict(torch.load(checkpoint_path))


new_headline = train_data[10].title
input_sequence = torch.Tensor([vocab.stoi[w] for w in headline]).unsqueeze(0).long()

hidden = model.encode(input_sequence) #input <-- maybe change later.
temps = [0.7,0.9,1.5]
for i in temps:
  for j in range(5):
    print(sample_sequence(model,hidden,max_len=20,temperature=i))



#let's load the **validation** data set:
valid_data = torchtext.legacy.data.TabularDataset(
    path=valid_path,                # data file path
    format="tsv",                   # fields are separated by a tab
    fields=[('title', text_field)]) # list of fields (we have only one)

"""
Computing the embeddings of every item in the validation set. Then, storing the
result in a single PyTorch tensor of shape `[19046, 128]`, since there are
19,046 headlines in the validation set.
"""
embeddings = []
for headline in valid_data:
  headline = headline.title
  input_seq = torch.Tensor([vocab.stoi[w] for w in headline]).long().unsqueeze(0)
  embds = model.encode(input_seq)
  #print("shape of embds = ",embds.shape)
  embeddings.append(embds.reshape(128))

embeddings = torch.stack(embeddings)
print("Shape of embeddings tensor = ",embeddings.shape)

"""
Finding the 5 closest headlines to the headline `valid_data[13]`. Useing cosine similarity
"""

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def five_closest(original_embedding, dataset_embeddings):
  score_dict = {}

  original_embedding = original_embedding.reshape(1,-1)

  for i in range(len(dataset_embeddings)):
    headline = word_embeddings[i].reshape(1, -1)
    cosine = cosine_similarity(original_embedding,headline)
    score_dict[cosine[0][0]] = i

  max_scores = score_dict.keys()
  max_scores = sorted(max_scores,reverse = True)
  top_five = max_scores[1:6]

  for i in range(len(top_five)):
    print(valid_data[score_dict[top_five[i]]].title)


original_headline = valid_data[13].title
input_headline= torch.Tensor([vocab.stoi[w] for w in original_headline]).unsqueeze(0).long()

word_embeddings = embeddings.detach() 

print("Original Headline = ", valid_data[13].title)
five_closest(word_embeddings[13],word_embeddings)

"""

Find the 5 closest headlines to another headline
"""

headline = valid_data[240].title
print("Original Headline = ", headline)

five_closest(word_embeddings[240],word_embeddings)

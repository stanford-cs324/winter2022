---
layout: page
parent: Lectures
title: Modeling
nav_order: 4.2
usemathjax: true
---
$$
\newcommand{\sV}{\mathcal{V}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\x}{x_{1:L}}
\newcommand{\softmax}{\text{softmax}}
\newcommand{\EmbedToken}{\text{EmbedToken}}
\newcommand{\SequenceModel}{\text{SequenceModel}}
\newcommand{\FeedForward}{\text{FeedForward}}
\newcommand{\FeedForwardSequenceModel}{\text{FeedForwardSequenceModel}}
\newcommand{\SequenceRNN}{\text{SequenceRNN}}
\newcommand{\BidirectionalSequenceRNN}{\text{BidirectionalSequenceRNN}}
\newcommand{\RNN}{\text{RNN}}
\newcommand{\SimpleRNN}{\text{SimpleRNN}}
\newcommand{\LSTM}{\text{LSTM}}
\newcommand{\GRU}{\text{GRU}}
\newcommand{\Attention}{\text{Attention}}
\newcommand{\MultiHeadedAttention}{\text{MultiHeadedAttention}}
\newcommand{\SelfAttention}{\text{SelfAttention}}
\newcommand{\TransformerBlock}{\text{TransformerBlock}}
\newcommand{\EmbedTokenWithPosition}{\text{EmbedTokenWithPosition}}
\newcommand{\LayerNorm}{\text{LayerNorm}}
\newcommand{\AddNorm}{\text{AddNorm}}
\newcommand{\nl}[1]{\textsf{#1}}
\newcommand{\generate}[1]{\stackrel{#1}{\rightsquigarrow}}
\newcommand{\embed}{\stackrel{\phi}{\Rightarrow}}
$$
We started this course by analyzing a language model as a black box:

$$p(x_1, \dots, x_L) \quad \text{or} \quad \text{prompt} \generate{} \text{completion}$$

Then we looked at the training data of large language models (e.g., The Pile):

$$\text{training data} \Rightarrow p.$$

In this lecture, we will open up the onion all the way and talk about how large
language models are built.

Today's lecture will focus on two topics, **tokenization** and **model architecture**.

1. [Tokenization](#tokenization): how a string is split into tokens.

2. [Model architecture](#model-architecture): We will discuss mostly the Transformer architecture,
   which is the modeling innovation that really enabled large language models.

## Tokenization

Recall that a language model $$p$$ is a probability distribution over a **sequence of tokens** where each token comes from some vocabulary $$\sV$$:

$$[\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}]$$

However, natural language doesn't come as a sequence of tokens, but as just a string (concretely, sequence of Unicode characters):

$$\nl{the mouse ate the cheese}$$

A **tokenizer** converts any string into a sequence of tokens.

$$\nl{the mouse ate the cheese} \Rightarrow [\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}]$$

This is not necessarily the most glamorous part of language modeling,
but plays a really important role in determining how well a model will work.

### Split by spaces

The simplest solution is to do:

> `text.split(' ')`

- This doesn't work for languages such as Chinese, where sentences are written without spaces between words:

> *我今天去了商店。* [gloss: *I went to the store.*]

- Then there are languages like German that have long compound words (e.g., *Abwasserbehandlungsanlange*).

- Even in English, there are hyphenated words (e.g., *father-in-law*) and contractions (e.g., *don't*), which should get split up.
  For example, the Penn Treebank splits *don't* into *do* and *n't*, a linguistically-informed but not obvious choice.

Therefore, splitting by spaces by spaces to identify words is quite problematic.

What makes a good tokenization?
- We don't want too **many** tokens (extreme: characters or bytes), or else the sequence becomes difficult to model.
- We don't want too **few** tokens, or else there won't be parameter sharing between words (e.g., should *mother-in-law* and *father-in-law* be completely different)?
  This is especially problematic for morphologically rich languages (e.g., Arabic, Turkish, etc.).
- Each token should be a linguistically or statistically meaningful unit.

### Byte pair encoding

[Sennrich et al, 2015](https://arxiv.org/pdf/1508.07909.pdf) applied the [byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) (BPE) algorithm,
originally developed for data compression, to produce one of the most commonly used tokenizers.

**Learning the tokenizer**.
Intuition: start with each character as its own token and combine tokens that co-occur a lot.

- Input: a training corpus (sequence of characters).
- Initialize the vocabulary $$\sV$$ be the set of characters.
- While we want to still grow $$\sV$$:
  * Find the pair of elements $$x,x' \in \sV$$ that co-occur the most number of times.
  * Replace all occurrences of $$x, x'$$ with a new symbol $$x x'$$.
  * Add $$x x'$$ to $$\sV$$.

Example:
1. [t, h, e, ␣, c, a, r], [t, h, e, ␣, c, a, t], [t, h, e, ␣, r, a, t]
1. [th, e, ␣, c, a, r], [th, e, ␣, c, a, t], [th, e, ␣, r, a, t]  (*th* occurs 3x)
1. [the, ␣, c, a, r], [the, ␣, c, a, t], [the, ␣, r, a, t]  (*the* occurs 3x)
1. [the, ␣, ca, r], [the, ␣, ca, t], [the, ␣, r, a, t]  (*ca* occurs 2x)

The output of learning is:
- Updated vocabulary $$\sV$$: [a, c, e, h, t, r, ca, th, the]
- The merges that we made (important for applying the tokenizer):
  * *t, h* $$\Rightarrow$$ *th*
  * *th, e* $$\Rightarrow$$ *the*
  * *c, a* $$\Rightarrow$$ *ca*

**Applying the tokenizer**.
To tokenize a new string, apply the merges in the same order:
- [t, h, e, ␣, o, x]
- [th, e, ␣, o, x]
- [the, ␣, o, x]

**Unicode**.
- One problem is that (especially in the multilingual setting), there are a lot (144,697) of Unicode characters.
- We certainly will not see all characters in the training data.
- In order to reduce data sparsity even further, we can run BPE on bytes instead of Unicode characters ([Wang et al. 2019](https://arxiv.org/pdf/1909.03341.pdf)).
- Example in Chinese: 

> *今天* [gloss: *today*]<br>
> [x62, x11, 4e, ca]<br>

### Unigram model (SentencePiece)

Rather than just splitting by frequency, a more "principled" approach is to define an objective function
that captures what a good tokenization looks like.
We now describe the **unigram model** ([Kudo 2018](https://arxiv.org/pdf/1804.10959.pdf)).
- It was of the tokenizations supported in the **SentencePiece** tool ([Kudo & Richardson, 2018](https://aclanthology.org/D18-2012.pdf)), along with BPE.
- It was used to train T5 and Gopher.

Given a sequence $$x_{1:L}$$, a tokenization $$T$$ is a set of 

$$p(x_{1:L}) = \prod_{(i, j) \in T} p(x_{i:j}).$$

Example: 
- Training data (string): $$\nl{ababc}$$
- Tokenization $$T = \{ (1, 2), (3, 4), (5, 5) \}$$ ($$\sV = \{ \nl{ab}, \nl{c} \}$$)
- Likelihood: $$p(x_{1:L}) = \frac{2}{3} \cdot \frac{2}{3} \cdot \frac{1}{3} = \frac{4}{9}$$.

**Algorithm**:
- Start with a "reasonably big" seed vocabulary $$\sV$$.
- Repeat:
  * Given $$\sV$$, optimize $$p(x)$$ and $$T$$ using the EM algorithm.
  * Compute $$\text{loss}(x)$$ for each token $$x \in \sV$$ capturing how much the likelihood would be reduced if $$x$$ were removed from $$\sV$$.
  * Sort by loss and keep the top 80% tokens in $$\sV$$.

### Comparing tokenizers

- GPT-2 and GPT-3 used BPE, vocabulary size of 50K
- [Jurassic](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) used SentencePiece with vocabulary size of 256K

Impact:
- Given the same string, Jurassic requires 28% fewer tokens than GPT-3, so it is 1.4x faster
- Both Jurassic and GPT-3 use the same context size (2048), so one can feed in 39% more text into the prompt.

Examples of tokenizations for both GPT-3 and Jurassic ([demo](https://crfm-models.stanford.edu/static/index.html?prompt=Abraham%20Lincoln%20lived%20at%20the%20White%20House.&settings=echo_prompt%3A%20true%0Amax_tokens%3A%200%0Atop_k_per_token%3A%205%0Amodel%3A%20%24%7Bmodel%7D&environments=model%3A%20%5Bopenai%2Fdavinci%2C%20ai21%2Fj1-jumbo%5D)):
- GPT-3: [Ab, raham, ␣Lincoln, ␣lived, ␣at, ␣the, ␣White, ␣House, .]
- Jurassic: [Abraham␣Lincoln, ␣lived, ␣at␣the␣White␣House, .]

## Models

Thus far, we have defined language models as a probability distribution over sequences of tokens $$p(x_1, \dots, x_L)$$,
which as we saw was very elegant and powerful (via prompting, a language model can in principle do anything, as GPT-3 hints at).
In practice, however, it can be more efficient for specialized tasks to avoid having to generatively model the entire sequence.

**Contextual embeddings**.
As a prerequisite, the main key development is to associate a sequence of tokens with a corresponding sequence of contextual embeddings:

$$[\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}] \embed \left[\binom{1}{0.1}, \binom{0}{1}, \binom{1}{1}, \binom{1}{-0.1}, \binom{0}{-1} \right].$$

- As the name suggests, the contextual embedding of a token depends on its context (surrounding words); for example, consider $$\nl{the}$$.
- Notation: We will $$\phi : \sV^L \to \R^{d \times L}$$ to be the embedding function (analogous to a feature map for sequences).
- For a token sequence $$x_{1:L} = [x_1, \dots, x_L]$$, $$\phi$$ produces contextual embeddings $$\phi(x_{1:L})$$.

### Types of language models

We will broaden our notion of language models to three types of models.

**Encoder-only** (BERT, RoBERTa, etc.).
These language models produce contextual embeddings but cannot be used directly to generate text.

$$x_{1:L} \Rightarrow \phi(x_{1:L}).$$

These contextual embeddings are generally used for classification tasks
(sometimes boldly called natural language understanding tasks).
- Example: sentiment classification

$$[\nl{[CLS]}, \nl{the}, \nl{movie}, \nl{was}, \nl{great}] \Rightarrow \text{positive}.$$

- Example: natural language inference

$$[\nl{[CLS]}, \nl{all}, \nl{animals}, \nl{breathe}, \nl{[SEP]}, \nl{cats}, \nl{breathe}] \Rightarrow \text{entailment}.$$

- Pro: contextual embedding for $$x_i$$ can depend **bidirectionally** on both the left context ($$x_{1:i-1}$$) and the right context ($$x_{i+1:L}$$).
- Con: cannot naturally **generate** completions.
- Con: requires more **ad-hoc training** objectives (masked language modeling).

**Decoder-only** (GPT-2, GPT-3, etc.). These are our standard autoregressive language models,
  which given a prompt $$x_{1:i}$$ produces both contextual embeddings and a
  distribution over next tokens $$x_{i+1}$$ (and recursively, over the entire
  completion $$x_{i+1:L}$$).

$$x_{1:i} \Rightarrow \phi(x_{1:i}), p(x_{i+1} \mid x_{1:i}).$$

- Example: text autocomplete

$$[\nl{[CLS]}, \nl{the}, \nl{movie}, \nl{was}] \Rightarrow \nl{great}$$

- Con: contextual embedding for $$x_i$$ can only depend **unidirectionally** on both the left context ($$x_{1:i-1}$$).
- Pro: can naturally **generate** completions.
- Pro: **simple training** objective (maximum likelihood).

**Encoder-decoder** (BART, T5, etc.).
These models in some ways can the best of both worlds:
they can use bidirectional contextual embeddings for the input $$x_{1:L}$$ and can generate the output $$y_{1:L}$$.

$$x_{1:L} \Rightarrow \phi(x_{1:L}), p(y_{1:L} \mid \phi(x_{1:L})).$$

- Example: table-to-text generation

$$[\nl{name}, \nl{:}, \nl{Clowns}, \nl{|}, \nl{eatType}, \nl{:}, \nl{coffee}, \nl{shop}] \Rightarrow [\nl{Clowns}, \nl{is}, \nl{a}, \nl{coffee}, \nl{shop}].$$

- Pro: contextual embedding for $$x_i$$ can depend **bidirectionally** on both the left context ($$x_{1:i-1}$$) and the right context ($$x_{i+1:L}$$).
- Pro: can naturally **generate** outputs.
- Con: requires more **ad-hoc training** objectives.

### Preliminaries

We now describe the innards of the embedding function $$\phi : \sV^L \to \R^{d \times L}$$:

$$[\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}] \embed \left[\binom{1}{0.1}, \binom{0}{1}, \binom{1}{1}, \binom{1}{-0.1}, \binom{0}{-1} \right].$$

We now introduce the model architectures for language model, with an emphasis on the ubiquitous Transformer architecture.
Our exposition of the Transformer architecture will be based on these [slides from CS221 on differentiable programming](https://stanford-cs221.github.io/autumn2021/modules/module.html#include=machine-learning%2Fdifferentiable-programming.js&slideId=embedding-tokens&level=0),
and will depart a bit from the standard presentation.

The beauty of deep learning is being able to create building blocks, just like we build whole programs out of functions.
So we want to be able to functions like the following to encapsulate the complexity:

$$\TransformerBlock(x_{1:L}).$$

This function will have parameters which we will include in the body but elide in the function signature for simplicity.

In what follows, we will define a library of building blocks until we get to the full Transformer.

### Preliminaries

First, we have to convert sequences of tokens into sequences of vectors.
$$\EmbedToken$$ does exactly this by looking up each token in an embedding matrix $$E \in \R^{|\sV| \times d}$$ (a parameter that will be learned from data):

$$[\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}]$$

def $$\EmbedToken(\x: \sV^L) \to \R^{d \times L}$$:
- *Turns each token $$x_i$$ in the sequence $$\x$$ into a vector*.
- Return $$[E_{x_1}, \dots, E_{x_L}]$$.

These are exactly the (context-independent) word embeddings of yore.
We define an abstract $$\SequenceModel$$ function that takes these **context-independent embeddings**
and maps them into **contextual embeddings**.

def $$\SequenceModel(\x: \R^{d \times L}) \to \R^{d \times L}$$:
- *Process each element $$x_i$$ in the sequence $$\x$$ with respect to other elements.*
- [abstract implementation (e.g., $$\FeedForwardSequenceModel$$, $$\SequenceRNN$$, $$\TransformerBlock$$)]

The simplest type of sequence model is based on feedforward networks
([Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf))
applied to a **fixed length** context, just as in an n-gram model:

def $$\FeedForwardSequenceModel(\x: \R^{d \times L}) \to \R^{d \times L}$$:
- *Process each element $$x_i$$ in the sequence $$\x$$ by looking at the last $$n$$ elements.*.
- For each $$i = 1, \dots, L$$:
  * Compute $$h_i = \FeedForward(x_{i-n+1}, \dots, x_i)$$.
- Return $$[h_1, \dots, h_L]$$.

### Recurrent neural networks

The first "real" sequence model is a recurrent neural network (RNN),
which is a family of models that include simple RNNs, LSTMs, and GRUs.
The basic form of an RNN simply computes a sequence of **hidden states**
recursively.

def $$\SequenceRNN(x: \R^{d \times L}) \to \R^{d \times L}$$:
- *Process the sequence $$x_1, \dots, x_L$$ left-to-right and recursively compute vectors $$h_1, \dots, h_L$$.*
- For $$i = 1, \dots, L$$:
  * Compute $$h_i = \RNN(h_{i-1}, x_i)$$.
- Return $$[h_1, \dots, h_L]$$.

The actual module that does the hard work is the $$\RNN$$,
which analogous to a finite state machine, takes the current state $$h$$, a new observation $$x$$,
and returns the updated state:

def $$\RNN(h: \R^d, x: \R^d) \to \R^d$$:
- *Updates the hidden state $$h$$ based on a new observation $$x$$.*
- [abstract implementation (e.g., $$\SimpleRNN$$, $$\LSTM$$, $$\GRU$$)]

There are three ways to implement the $$\RNN$$.
The earliest RNN is a simple RNN [Elman, 1990](https://onlinelibrary.wiley.com/doi/epdf/10.1207/s15516709cog1402_1),
which takes a linear combination of $$h$$ and $$x$$
and pushes it through an elementwise non-linear function $$\sigma$$
(e.g., logistic $$\sigma(z) = (1 + e^{-z})^{-1}$$ or more the modern ReLU $$\sigma(z) = \max(0, z)$$).

def $$\SimpleRNN(h: \R^d, x: \R^d) \to \R^d$$:
- *Updates the hidden state $$h$$ based on a new observation $$x$$ by simple linear transformation and non-linearity.*
- Return $$\sigma(U h + V x + b)$$.

As defined RNNs only depend on the past, but we can them depend on the future two by running another RNN backwards.
These models were used by [ELMo](https://arxiv.org/pdf/1802.05365.pdf) and [ULMFiT](https://arxiv.org/pdf/1801.06146.pdf).

def $$\BidirectionalSequenceRNN(\x: \R^{d \times L}) \to \R^{2d \times L}$$:
- *Process the sequence both left-to-right and right-to-left.*
- Compute left-to-right: $$[h_1^\rightarrow, \dots, \vec{h}_L^\rightarrow] \leftarrow \SequenceRNN(x_1, \dots, x_L)$$.
- Compute right-to-left: $$[h_L^\leftarrow, \dots, h_1^\leftarrow] \leftarrow \SequenceRNN(x_L, \dots, x_1)$$.
- Return $$[h_1^\rightarrow h_1^\leftarrow, \dots, h_L^\rightarrow h_L^\leftarrow]$$.

Notes:
- The simple RNN is difficult to train due to vanishing gradients.
- The Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) (both of $$\RNN$$) have been developed to address these.
- Still, even though the embedding $$h_{200}$$ can depend arbitrarily far back (e.g., on $$x_1$$), it is unlikely to depend on it in a "crisp" way
  (see [Khandelwal et al., 2018](https://arxiv.org/pdf/1805.04623.pdf) for more discussion).
- LSTMs in some sense were really what brought deep learning into full swing within NLP.

We will not discuss these models in the interest of time.

### Transformers

Now, we will discuss Transformers ([Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762.pdf)),
the sequence model that is really responsible for the takeoff of large language models;
they are the building blocks of decoder-only (GPT-2, GPT-3), encoder-only (BERT, RoBERTa), and decoder-encoder (BART, T5) models.

There are great resources for learning about the Transformer:
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/):
  very nice visual description of the Transformer.
- [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html):
  Pytorch implementation of the Transformer.

You are highly encouraged to read these references.
In this lecture, I will strive to take a middle path which emphasizes
pseudocode functions and interfaces.

The crux of the Transformers are the **attention mechanism**,
which was developed earlier for machine translation ([Bahdananu et al. 2017](https://arxiv.org/pdf/1409.0473.pdf)).

One can think of attention as a "soft" lookup table, where we have a query $$y$$
that we want to match against each element in a sequence $$x_{1:L} = [x_1, \dots, x_L]$$:

$$[x_1, \dots, x_L] \quad\quad\quad y$$

We can think of each $$x_i$$ as representing a key-value pair via linear transformations:

$$(W_\text{key} x_i): (W_\text{value} x_i)$$

and forming the query via another linear transformation:

$$W_\text{query} y.$$

The key and the query can be compared to give a score:

$$\text{score}_i = x_i^\top W_\text{key}^\top W_\text{query} y.$$

These scores can be exponentiated and normalized to form a probability distribution over the token positions $$\{ 1, \dots, L \}$$:

$$[\alpha_1, \dots, \alpha_L] = \softmax([\text{score}_1, \dots, \text{score}_L]).$$

Then the final output is a weighted combination over the values:

$$\sum_{i=1}^L \alpha_i (W_\text{value} x_i).$$

We can write this all succinctly in matrix form:

def $$\Attention(\x: \R^{d \times L}, y: \R^d) \to \R^d$$:
- *Process $$y$$ by comparing it to each $$x_i$$.*
- Return $$W_\text{value} \, \x \, \softmax(\x^\top W_\text{key}^\top W_\text{query} y / \sqrt{d})$$.

We can think of there as being multiple aspects (e.g., syntax, semantics) that we would want to match on.
To accommodate this, we can simultaneously have multiple **attention heads** and simply combine their outputs.

def $$\MultiHeadedAttention(\x: \R^{d \times L}, y: \R^d) \to \R^d:$$
- *Process $$y$$ by comparing it to each $$x_i$$ with respect to $$n_\text{heads}$$ aspects.*
- Return $$W_\text{output} \underbrace{[\Attention(\x, y), \dots, \Attention(\x, y)]}_{n_\text{heads} \text{times}}$$.

**Self-attention layer**.
Now we will substitute each $$x_i$$ in for $$y$$ as the query argument to produce:

def $$\SelfAttention(\x: \R^{d \times L}) \to \R^{d \times L})$$:
- *Compare each element $$x_i$$ to each other element.*
- Return $$[\Attention(\x, x_1), \dots, \Attention(\x, x_L)]$$.


**Feedforward layer**.
Self-attention allows all the tokens to "talk" to each other, whereas feedforward connections provide:

def $$\FeedForward(\x: \R^{d \times L}) \to \R^{d \times L}$$:
- *Process each token independently.*
- For $$i = 1, \dots, L$$:
  * Compute $$y_i = W_2 \max(W_1 x_i + b_1, 0) + b_2$$.
- Return $$[y_1, \dots, y_L]$$.

**Improving trainability**.
We're almost done.
We could in principle just take the $$\FeedForward \circ \SelfAttention$$ sequence model and iterate it 96 times to make GPT-3,
but that network would be hard to optimize (for the same vanishing gradients problems that afflicted RNNs, now just along the depth direction).
So we have to do two shenanigans to make sure that the network is trainable.

**Residual connections**.
One trick from computer vision is residual connections (ResNet).
Instead of applying some function $$f$$:

$$f(\x),$$

we add a residual (skip) connection so that if $$f$$'s gradients vanish, gradients can still flow through $$\x$$:

$$\x + f(\x).$$

**Layer normalization**.
Another trick is [layer normalization](https://arxiv.org/pdf/1607.06450.pdf),
which takes a takes a vector and makes sure its elements are too big:

def $$\LayerNorm(\x: \R^{d \times L}) \to \R^{d \times L}$$:
- *Make each $$x_i$$ not too big or small*.

We first define an adapter function that takes a sequence model $$f$$ and makes it "robust":

def $$\AddNorm(f: (\R^{d \times L} \to \R^{d \times L}), \x: \R^{d \times L}) \to \R^{d \times L}$$:
- *Safely apply $$f$$ to $$\x$$*.
- Return $$\LayerNorm(\x + f(\x))$$.

Finally, we can define the Transformer block succinctly as follows:

def $$\TransformerBlock(\x: \R^{d \times L}) \to \R^{d \times L}$$:
- *Process each element $$x_i$$ in context.*
- Return $$\AddNorm(\FeedForward, \AddNorm(\SelfAttention, \x))$$.

**Positional embeddings**.
You might have noticed that as defined,
the embedding of a token doesn't depend on where it occurs in the sequence,
so $$\nl{mouse}$$ in both sentences would have the same embedding,
which is not sensible.

$$[\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}]$$

$$[\nl{the}, \nl{cheese}, \nl{ate}, \nl{the}, \nl{mouse}]$$

To fix this, we add **positional information** into the embedding:

def $$\EmbedTokenWithPosition(\x: \R^{d \times L})$$:
- *Add in positional information*.
- Define positional embeddings:
  * Even dimensions: $$P_{i,2j} = \sin(i / 10000^{2j/d_\text{model}})$$
  * Odd dimensions: $$P_{i,2j+1} = \cos(i / 10000^{2j/d_\text{model}})$$
- Return $$[x_1 + P_1, \dots, x_L + P_L]$$.

**GPT-3**.
With all the pieces in place, we can now define roughly GPT-3 architecture in one line,
just by stacking the Transformer block 96 times:

$$\text{GPT-3}(x_{1:L}) = \TransformerBlock^{96}(\EmbedTokenWithPosition(x_{1:L}))$$

Shape of the architecture (how the 175 billion parameters are allocated):
- Dimension of hidden state: $$d_\text{model} = 12288$$
- Dimension of the intermediate feed-forward layer: $$d_\text{ff} = 4 d_\text{model}$$
- Number of heads: $$n_\text{heads} = 96$$
- Context length: $$L = 2048$$

These decisions are not necessarily optimal.
[Levine et al. 2020](https://arxiv.org/pdf/2006.12467.pdf) provide some theoretical justification,
showing that the GPT-3 is too deep, which motivated the training of a deeper but wider Jurassic architecture.

There are important but detailed differences between different versions of Transformers:
- Layer normalization "post-norm" (original Transformers paper) versus pre-norm (GPT-2), which impacts training stability
  ([Davis et al. 2021](http://proceedings.mlr.press/v139/davis21a/davis21a.pdf)).
- Dropout is applied throughout to prevent overfitting.
- GPT-3 uses a [sparse Transformer](https://arxiv.org/pdf/1904.10509.pdf) to reduce the number of parameters,
  interleaving it with dense layers.
- Depending on the type of Transformer (encoder-only, decoder-only, encoder-decoder), different masking operations are used.
- And of course there are many more details involved in the training of Transformer models which we will discuss next time.

## Further reading

Tokenization:
- [Between words and characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP](https://arxiv.org/pdf/2112.10508.pdf). *Sabrina J. Mielke, Zaid Alyafeai, Elizabeth Salesky, Colin Raffel, Manan Dey, Matthias Gallé, Arun Raja, Chenglei Si, Wilson Y. Lee, Benoît Sagot, Samson Tan*. 2021.
  Comprehensive survey of tokenization.
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf). *Rico Sennrich, B. Haddow, Alexandra Birch*. ACL 2015.
  Introduces **byte pair encoding** into NLP.  Used by GPT-2, GPT-3.
- [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf). *Yonghui Wu, M. Schuster, Z. Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, M. Krikun, Yuan Cao, Qin Gao, Klaus Macherey, J. Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser, Stephan Gouws, Y. Kato, Taku Kudo, H. Kazawa, K. Stevens, George Kurian, Nishant Patil, W. Wang, C. Young, Jason R. Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, G. Corrado, Macduff Hughes, J. Dean*. 2016.
  Introduces **WordPiece**.  Used by BERT.
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf). *Taku Kudo, John Richardson*. EMNLP 2018.
  Introduces **SentencePiece**.

Modeling:
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
  Introduces GPT-2.
- [Attention is All you Need](https://arxiv.org/pdf/1706.03762.pdf). *Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*. NIPS 2017.
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [CS224N slides on RNNs](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture06-fancy-rnn.pdf)
- [CS224N slides on Transformers](http://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture09-transformers.pdf)

- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/pdf/2108.12409.pdf). *Ofir Press, Noah A. Smith, M. Lewis*. 2021.
  Introduces **Alibi embeddings**.

- [Transformer-XL: Attentive Language Models beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf). *Zihang Dai, Zhilin Yang, Yiming Yang, J. Carbonell, Quoc V. Le, R. Salakhutdinov*. ACL 2019.
  Introduces recurrence on Transformers, relative position encoding scheme.

- [Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf). *R. Child, Scott Gray, Alec Radford, Ilya Sutskever*. 2019.
  Introduces **Sparse Transformers**.
- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/pdf/2006.04768.pdf). *Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma*. 2020.
  Introduces **Linformers**.
- [Rethinking Attention with Performers](https://arxiv.org/pdf/2009.14794.pdf). *K. Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy J. Colwell, Adrian Weller*. ICLR 2020.
  Introduces **Performers**.
- [Efficient Transformers: A Survey](https://arxiv.org/pdf/2009.06732.pdf). *Yi Tay, M. Dehghani, Dara Bahri, Donald Metzler*. 2020.

Decoder-only architectures:
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). *Alec Radford, Jeff Wu, R. Child, D. Luan, Dario Amodei, Ilya Sutskever*. 2019.
  Introduces **GPT-2** from OpenAI.
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf). *Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, J. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. Henighan, R. Child, A. Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei*. NeurIPS 2020.
  Introduces **GPT-3** from OpenAI.
- [Scaling Language Models: Methods, Analysis&Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf). *Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, J. Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, G. V. D. Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John F. J. Mellor, I. Higgins, Antonia Creswell, Nathan McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, D. Budden, Esme Sutherland, K. Simonyan, Michela Paganini, L. Sifre, Lena Martens, Xiang Lorraine Li, A. Kuncoro, Aida Nematzadeh, E. Gribovskaya, Domenic Donato, Angeliki Lazaridou, A. Mensch, J. Lespiau, Maria Tsimpoukelli, N. Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Tobias Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, I. Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William S. Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem W. Ayoub, Jeff Stanway, L. Bennett, D. Hassabis, K. Kavukcuoglu, Geoffrey Irving*. 2021.
  Introduces **Gopher** from DeepMind.
- [Jurassic-1: Technical details and evaluation](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf). *Opher Lieber, Or Sharir, Barak Lenz, Yoav Shoham*. 2021.
  Introduces **Jurassic** from AI21 Labs.

Encoder-only architectures:
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf). *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*. NAACL 2019.
  Introduces **BERT** from Google.
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf). *Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, M. Lewis, Luke Zettlemoyer, Veselin Stoyanov*. 2019.
  Introduces **RoBERTa** from Facebook.

Encoder-decoder architectures:
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf). *M. Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, Luke Zettlemoyer*. ACL 2019.
  Introduces **BART** from Facebook.
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf). *Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, W. Li, Peter J. Liu*. J. Mach. Learn. Res. 2019.
  Introduces **T5** from Google.

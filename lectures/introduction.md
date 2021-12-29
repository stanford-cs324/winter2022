---
layout: page
parent: Lectures
title: Introduction
nav_order: 1
usemathjax: true
---
$$
\newcommand{\sV}{\mathcal{V}}
\newcommand{\nl}[1]{\textsf{#1}}
$$

*UNDER CONSTRUCTION*

Hello, welcome to CS324!  This is a new course on understanding and developing **large language models**.

TODO: language is important

TODO: Shannon

TODO: Jelinek

## Why does this course exist?

Language models are usually covered in a standard NLP course (such as CS224N),
so why are *large* language models any different?

TODO: table with numbers

We see that the model sizes have increased by an order of 5000x over the last 4 years.
Training models such as GPT-3 are **expensive** (cost millions of dollars of compute).
Whereas smaller models such as BERT are publicly released, more recent models
such as GPT-3 are **closed** and only available through API access (if at all).
And of course, they are interesting because they represent a step function increase in **capabilities**.
For example, TODO.

TODO(generated text).

TODO(in-context learning)

TODO(stochastic parrots)

## Structure of this course

1. [Behavior](/lectures/behavior)
1. [Data](/lectures/data)
1. [Building](/lectures/building)

## What is a language model?

The term **language model** is used fairly loosely to mean any machine learning model that involves language.

The most classic definition of a language model is a **probability distribution over sequences of tokens**.
Suppose we have a **vocabulary** $$\sV$$ of a set of tokens.
For each sequence of tokens $$x_1, \dots, x_L \in \sV$$, the language model assigns a probability (a number between 0 and 1):

$$p(x_1, \dots, x_L).$$

For example, if $$\sV = \{ \nl{the}, \nl{mouse}, \nl{cheese}, \nl{ate} \}$$, the language model might assign:

$$p(\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}) = 0.002.$$

$$p(\nl{mouse}, \nl{the}, \nl{the}, \nl{cheese}, \nl{ate}) = 0.00001.$$

$$p(\nl{the}, \nl{cheese}, \nl{ate}, \nl{the}, \nl{mouse}) = 0.001.$$

TODO: get real probabilities from GPT-3

Note that the probability tells us whether a sequence of tokens is "good" or not.

As defined, a language model takes a sequence and returns a probability.
Mathematically, a language model can be used to **generate** a sequence by
sampling a sequence $$x_{1:L}$$ from the language model with probability equal to $$p(x_{1:L})$$, denoted:

$$x_1, \dots, x_L \sim p.$$

### Autoregressive language models

A **autoregressive language model** is specified by a **conditional probability distribution of the next token given the previous tokens**:

$$p(x_i \mid x_{1:i-1}).$$

For example, $$p(x_3 \mid \nl{the}, \nl{mouse})$$ specifies a distribution over the third word given that the first two words are $$\nl{the}$$ and $$\nl{mouse}$$.

Given this conditional probability distribution,
we can define the joint distribution over the entire sequence:

$$p(x_1, \dots, x_L) = \prod_{i=1}^L p(x_i \mid x_{1:i-1}) = p(x_1) p(x_2 \mid x_1) \cdots p(x_L \mid x_1, \dots, x_{L-1})$$

For example,

$$
\begin{align*}
p(\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}) = \,
& p(\nl{the}) \\
& p(\nl{mouse} \mid \nl{the}) \\
& p(\nl{ate} \mid \nl{the}, \nl{mouse}) \\
& p(\nl{the} \mid \nl{the}, \nl{mouse}, \nl{ate}) \\
& p(\nl{cheese} \mid \nl{the}, \nl{mouse}, \nl{ate}, \nl{the}).
\end{align*}
$$

Of course, any probability distribution can be written this way using the chain rule,
but the important aspect is that each conditional distribution $$p(x_i \mid x_{1:i-1})$$ can be **computed efficiently**.

Now to **generate** from an autoregressive language model, we can generate sequentially:

$$
\text{for } i = 1, \dots, L: \\
\hspace{0.5in} x_i \sim p(x_i \mid x_{1:i-1}).
$$

(We have assumed that sentences are [tokenized](../tokenization) into sequence of tokens.)

## Further reading

- [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922). *Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell*. FAccT 2021.
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf). *Rishi Bommasani, Drew A. Hudson, E. Adeli, R. Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, E. Brynjolfsson, S. Buch, D. Card, Rodrigo Castellon, Niladri S. Chatterji, Annie Chen, Kathleen Creel, Jared Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, S. Ermon, J. Etchemendy, Kawin Ethayarajh, L. Fei-Fei, Chelsea Finn, Trevor Gale, Lauren E. Gillespie, Karan Goel, Noah D. Goodman, S. Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas F. Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, G. Keeling, Fereshte Khani, O. Khattab, Pang Wei Koh, M. Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, J. Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir P. Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, A. Narayan, D. Narayanan, Benjamin Newman, Allen Nie, Juan Carlos Niebles, H. Nilforoshan, J. Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, J. Park, C. Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Robert Reich, Hongyu Ren, Frieda Rong, Yusuf H. Roohani, Camilo Ruiz, Jackson K. Ryan, Christopher R'e, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, K. Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramèr, Rose E. Wang, William Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, M. Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, Percy Liang*. 2021.

- [CS224N lecture notes on language models](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)

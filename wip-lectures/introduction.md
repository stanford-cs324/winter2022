---
layout: page
parent: WIP Lectures
title: WIP Introduction
nav_order: 1
usemathjax: true
---
$$
\newcommand{\sV}{\mathcal{V}}
\newcommand{\nl}[1]{\textsf{#1}}
$$

Welcome to CS324!  This is a new course on understanding and developing **large language models**.
What is a large language model and why is there a new course on it?

<!--Language models are usually covered in a standard NLP course (such as CS224N),
so why are *large* language models any different?-->

First, as the name suggests, large language models are **large**.
The following table shows what is meant by "large":

| Model               | Organization      | Date     | Size (# params) |
|---------------------|-------------------|----------|----------------:|
| ELMo                | AI2               | Feb 2018 | 94,000,000      |
| GPT                 | OpenAI            | Jun 2018 | 110,000,000     |
| BERT                | Google            | Oct 2018 | 340,000,000     |
| XLM                 | Facebook          | Jan 2019 | 655,000,000     |
| GPT-2               | OpenAI            | Mar 2019 | 1,500,000,000   |
| RoBERTa             | Facebook          | Jul 2019 | 355,000,000     |
| Megatron-LM         | NVIDIA            | Sep 2019 | 8,300,000,000   |
| T5                  | Google            | Oct 2019 | 11,000,000,000  |
| Turing-NLG          | Microsoft         | Feb 2020 | 17,000,000,000  |
| GPT-3               | OpenAI            | May 2020 | 175,000,000,000 |
| Megatron-Turing NLG | Microsoft, NVIDIA | Oct 2021 | 530,000,000,000 |
| Gopher              | DeepMind          | Dec 2021 | 280,000,000,000 |

Adjectives such as enormous, huge, and massive, are probably better descriptors.
We see that the model sizes have increased by an order of 5000x over just the last 4 years.

Second, this scaling has enabled a step function increase in **capabilities**.
They are capable of generating text.  Here is an example of an article that
GPT-3 fabricated (everything after the bolded text):

> **Title: United Methodists Agree to Historic Split
> Subtitle: Those who oppose gay marriage will form their own denomination
> Article:** After two days of intense debate, the United Methodist Church
> has agreed to a historic split - one that is expected to end in the
> creation of a new denomination, one that will be "theologically and
> socially conservative," according to The Washington Post. The majority of
> delegates attending the church's annual General Conference in May voted to
> strengthen a ban on the ordination of LGBTQ clergy and to write new rules
> that will "discipline" clergy who officiate at same-sex weddings. But
> those who opposed these measures have a new plan: They say they will form a
> separate denomination by 2020, calling their church the Christian Methodist
> denomination.
> The Post notes that the denomination, which claims 12.5 million members, was
> in the early 20th century the "largest Protestant denomination in the U.S.,"
> but that it has been shrinking in recent decades. The new split will be the
> second in the church's history. The first occurred in 1968, when roughly
> 10 percent of the denomination left to form the Evangelical United Brethren
> Church. The Post notes that the proposed split "comes at a critical time
> for the church, which has been losing members for years," which has been
> "pushed toward the brink of a schism over the role of LGBTQ people in the
> church." Gay marriage is not the only issue that has divided the church. In
> 2016, the denomination was split over ordination of transgender clergy, with
> the North Pacific regional conference voting to ban them from serving as
> clergy, and the South Pacific regional conference voting to allow them.

Not only can they generate text, as one expects from a language model (as ),
what is really surprising is that they can perform **in-context learning**,
whereby examples are given directly to the language model.

> **Translate English to French:
> sea otter => loutre de mer
> cheese =>** fromage

TODO(stochastic parrots)

Training models such as GPT-3 are **expensive** (cost millions of dollars of compute).
Whereas smaller models such as BERT are publicly released, more recent models
such as GPT-3 are **closed** and only available through API access (if at all).

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

## A brief history

TODO: language is important
language models are canonically for 
code, vision.

TODO: Shannon

TODO: Jelinek

## Further reading

- [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922). *Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell*. FAccT 2021.
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf). *Rishi Bommasani, Drew A. Hudson, E. Adeli, R. Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, E. Brynjolfsson, S. Buch, D. Card, Rodrigo Castellon, Niladri S. Chatterji, Annie Chen, Kathleen Creel, Jared Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, S. Ermon, J. Etchemendy, Kawin Ethayarajh, L. Fei-Fei, Chelsea Finn, Trevor Gale, Lauren E. Gillespie, Karan Goel, Noah D. Goodman, S. Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas F. Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, G. Keeling, Fereshte Khani, O. Khattab, Pang Wei Koh, M. Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, J. Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir P. Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, A. Narayan, D. Narayanan, Benjamin Newman, Allen Nie, Juan Carlos Niebles, H. Nilforoshan, J. Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, J. Park, C. Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Robert Reich, Hongyu Ren, Frieda Rong, Yusuf H. Roohani, Camilo Ruiz, Jackson K. Ryan, Christopher R'e, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, K. Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramèr, Rose E. Wang, William Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, M. Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, Percy Liang*. 2021.

- [CS224N lecture notes on language models](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)

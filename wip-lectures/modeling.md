---
layout: page
grand_parent: WIP Lectures
parent: WIP Building
title: WIP Modeling
nav_order: 4.2
usemathjax: true
---
## Tokenization

### Byte pair encoding

GPT-2 operates on bytes, not unicode

BERT
- WordPiece embeddings
- [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf). *Yonghui Wu, M. Schuster, Z. Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, M. Krikun, Yuan Cao, Qin Gao, Klaus Macherey, J. Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser, Stephan Gouws, Y. Kato, Taku Kudo, H. Kazawa, K. Stevens, George Kurian, Nishant Patil, W. Wang, C. Young, Jason R. Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, G. Corrado, Macduff Hughes, J. Dean*. 2016.

## Transformer

## Encoder-only, decoder-only, encoder-decoder

- GPT-3
- Sparse Transformer

## Layer normalization

## Positional embeddings

## Modeling long sequences

Fixed patterns [Child et al. 2019]

Learned attention

Low-rank methods

GPT-3
- Sparse Transformer

Gopher
- SentencePiece

## Further reading

Tokenization:
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf). *Rico Sennrich, B. Haddow, Alexandra Birch*. ACL 2015.
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf). *Taku Kudo, John Richardson*. EMNLP 2018.

Modeling:
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
  Introduces GPT-2.
- [Attention is All you Need](https://arxiv.org/pdf/1706.03762.pdf). *Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*. NIPS 2017.
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
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

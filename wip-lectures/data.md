---
layout: page
parent: WIP Lectures
title: WIP Data
nav_order: 3
has_children: true
usemathjax: true
---
## Data behind large language models

GPT-3 - CommonCrawl

The Google search index is 100,000,000 GB ([reference](https://www.google.com/search/howsearchworks/how-search-works/organizing-information/)),
and the actual web is likely even larger, and the [Deep Web](https://en.wikipedia.org/wiki/Deep_web) is even larger.

## Documentation

## Data governance

## Data privacy

## Data dignity

## Further reading

Documentation:
- [Datasheets for datasets](https://arxiv.org/pdf/1803.09010.pdf). *Timnit Gebru, Jamie H. Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, H. Wallach, Hal Daumé, Kate Crawford*. Communications of the ACM 2018.
- [Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science](https://aclanthology.org/Q18-1041.pdf). *Emily M. Bender and Batya Friedman*. ACL 2018.
- [Model Cards for Model Reporting](https://arxiv.org/pdf/1810.03993.pdf). *Margaret Mitchell, Simone Wu, Andrew Zaldivar, P. Barnes, Lucy Vasserman, B. Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru*. FAT 2018.

Web datasets:
- [CommonCrawl](http://commoncrawl.org/)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf). *Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, W. Li, Peter J. Liu*. J. Mach. Learn. Res. 2019.
  Introduces **Clossal Clean Crawled Corpus (C4)** and the T5 model.
- [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/pdf/1911.00359.pdf). *Guillaume Wenzek, Marie-Anne Lachaux, A. Conneau, Vishrav Chaudhary, Francisco Guzm'an, Armand Joulin, Edouard Grave*. LREC 2019.
  Introduces **CCNet**.
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/pdf/2101.00027.pdf). *Leo Gao, Stella Rose Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy*. 2020.  Introduces **The Pile**.
  Introduces **The Pile**, used to train GPT-J.
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/)
  Similar to WebText, used to train GPT-2.
- [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf). *A. Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov*. ACL 2019.
  Introduces cleaned versions of CommonCrawl corpus on 100 datasets, used to train XLM-R.

Analysis of datasets:
- [An Empirical Exploration in Quality Filtering of Text Data](https://arxiv.org/pdf/2109.00698.pdf). *Leo Gao*. 2021.
- [Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus](https://arxiv.org/pdf/2104.08758.pdf). *Jesse Dodge, Ana Marasović, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, Matt Gardner*. EMNLP 2021.
- [Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](https://arxiv.org/pdf/2103.12028.pdf). *Isaac Caswell, Julia Kreutzer, Lisa Wang, Ahsan Wahab, D. Esch, Nasanbayar Ulzii-Orshikh, A. Tapo, Nishant Subramani, A. Sokolov, Claytone Sikasote, Monang Setyawan, S. Sarin, Sokhar Samb, B. Sagot, Clara Rivera, Annette Rios Gonzales, Isabel Papadimitriou, Salomey Osei, Pedro Ortiz Suarez, Iroro Orife, Kelechi Ogueji, Rubungo Andre Niyongabo, Toan Q. Nguyen, Mathias Muller, A. Muller, S. Muhammad, N. Muhammad, Ayanda Mnyakeni, Jamshidbek Mirzakhalov, Tapiwanashe Matangira, Colin Leong, Nze Lawson, Sneha Kudugunta, Yacine Jernite, M. Jenny, Orhan Firat, Bonaventure F. P. Dossou, Sakhile Dlamini, N. D. Silva, Sakine cCabuk Balli, Stella Rose Biderman, A. Battisti, Ahmed Baruwa, Ankur Bapna, P. Baljekar, Israel Abebe Azime, A. Awokoya, Duygu Ataman, Orevaoghene Ahia, Oghenefego Ahia, Sweta Agrawal, Mofetoluwa Adeyemi*. 2021.

- [Data Freedom Act](https://www.radicalxchange.org/media/papers/data-freedom-act.pdf)

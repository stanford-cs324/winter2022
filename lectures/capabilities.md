---
layout: page
parent: Lectures
title: Capabilities
nav_order: 2.1
usemathjax: true
---
$$
\newcommand{\nl}[1]{\textsf{#1}}
\newcommand{\generate}[1]{\stackrel{#1}{\rightsquigarrow}}
\newcommand{\perplexity}{\text{perplexity}}
$$

In this lecture, we will explore the capabilities of GPT-3, the canonical large language model.
We will closely follow the benchmarks from the [GPT-3
paper](https://arxiv.org/pdf/2005.14165.pdf), which include:
- standard NLP benchmarks (e.g., question answering), as well as
- quirky one-off demos (e.g., using a new word in a sentence).

In comparison with the state-of-the-art-result for each task, the results are **mixed**:
- On some tasks such as language modeling, GPT-3 exceeds the state-of-the-art by a **huge margin**.
- On others, where GPT-3 is competing against systems that are trained with large amounts of labeled data, it **lags far behind**.

The way to think about these results is as follows:
- GPT-3 was **not trained on these tasks** explicitly; it was just trained as a language model to predict the next word.
- Nonetheless, **even without "trying"**, GPT-3 does a passable job on average at a broad range of NLP tasks.
- Because GPT-3 was not trained on any of these tasks, it hasn't overfit, which
  means it has a **good chance of doing well at many many other tasks** (as
  seen by the passable performance on one-off tasks).
- Moreover, if you wanted to do well on any particular task (e.g., question answering),
  you should in principle be able to **adapt GPT-3 using the large amounts of labeled data**
  to exceed state-of-the-art.

**Adaptation**.
Recall that a **language model** $$p$$ is a distribution over sequences of tokens $$x_{1:L}$$
and thus can be used to score sequences:

$$p(\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}).$$

It can also be used to perform conditional generation of a completion given a prompt:

$$\nl{the mouse ate} \generate{} \nl{the cheese}.$$

A **task** is a mapping from inputs to outputs.  For example, for question answering, we might have:

> Input: What school did burne hogarth establish?<br>
> Output: School of Visual Arts

We use the term **adaptation** to refer to the process of taking a language
model and turning it into a task model, given:
- a natural language **description** of the task, and
- a set of **training instances** (input-output pairs).

There are two primary ways to perform adaptation:
1. **Training** (standard supervised learning): train a new model that maps inputs to outputs, either by
    1. creating a new model that uses the language model as features (probing), or
    1. starting with the language model and updating it based on the training instances (fine-tuning), or
    1. something in between (lightweight fine-tuning).
1. **Prompting** (in-context learning): Construct a prompt (a string based on
   the description and training instances) or a set of prompts, feed those into a
   language model to obtain completions.
    1. Zero-shot learning: number of training examples is 0
    1. One-shot learning: number of training examples is 1
    1. Few-shot learning: number of training examples is few

Which adaptation procedure should we go with?
- **Training can be challenging due to overfitting** (just imagine fine-tuning
  a 175 billion parameter model based on 5 examples).  How to do this
  effectively will be the topic of the adaptation lecture.
- For now, we will be content with **adaptation of GPT-3 using prompting**.
  Note that the limitation of prompting is that we can only leverage a only
  small number of training instances (as many as can fit into a prompt).  This
  is due to a limitation of Transformers, where the prompt and the completion
  must fit into 2048 tokens.

The GPT-3 paper evaluated GPT-3 on a large set of tasks.
We will consider a subset of these, and for each task, discuss the following:
1. **Definition**: What is the task and its motivation?
1. **Adaptation**: How do we reduce the task to language modeling (via prompting)?
1. **Results**: What are the quantitative numbers compared to task-specific state-of-the-art models?

**Size and number of examples matters**.
By default, the results will based on
- the full GPT-3 model (davinci), which has 175 billion parameters
- using in-context learning with as many training instances as you can stuff into the prompt.

Along the way, we will do ablations to see if model size and number of
in-context training instances matters.
Spoiler: it does and more is better.

The tasks are grouped as follows:
1. [Language modeling](#language-modeling)
1. [Question answering](#question-answering)
1. [Translation](#translation)
1. [Arithmetic](#arithmetic)
1. [News article generation](#news-article-generation)
1. [Novel tasks](#novel-tasks)

The goals of this lecture is to provide:
1. an overview of tasks in NLP (independent of large language models),
1. a sense of how well GPT-3 works, and
1. a taste for the art of prompt engineering.

## Language modeling

The most natural starting point for thinking about what a language model can do
is to ask if it can do the thing that language models are supposed to do:
model language.

Recall that a language model $$p$$ is a probability distribution over sequences
of tokens.  Suppose we take a corpus of text $$x_{1:L}$$, for example:

$$\nl{the mouse ate the cheese}$$

We can ask: what is the probability the language model assigns to it?

$$p(\nl{the mouse ate the cheese})$$

Recall that we can break down the the joint probability into the product of the conditional probabilities for each token by the chain rule:

$$p(x_{1:L}) = \prod_{i=1}^L p(x_i \mid x_{1:i-1}).$$

**Perplexity**.
The joint probability of a sequence depends on its length and thus **goes to zero** as the length grows,
which makes it hard to track.  (Just think about trying to get a better
estimate of perplexity on newswire by getting more newswire.)

Intuitively we want to average the per token probabilities $$p(x_i \mid x_{1:i-1})$$.
We don't want to take the arithmetic average because assigning a token
probability 0 is really bad (think about coding: your code length would be
infinite), but the arithmetic average doesn't penalize you for that.
Instead, we want the **geometric average**, which is exactly what perplexity does:

$$\perplexity_p(x_{1:L}) = \exp\left(\frac{1}{L} \sum_{i=1}^L \log \frac{1}{p(x_i \mid x_{1:i-1})}\right).$$

Perplexity can be interpreted as the **average "branching factor"** per token.
Recall that $$\log \frac{1}{p(x_i \mid x_{1:i-1})}$$ is the code length.
We are taking the average code length; exponentiating provides the number of possibilities.
For intuition, take uniform distribution:
a bitstring of length of 3 can encode $$2^3$$ possible strings.

**Tale of two errors**.
There are two types of errors a language model can make, and perplexity treats them asymmetrically:
- **Recall error**: The language model fails to place probability mass on some token.  Perplexity has no mercy:

$$p(\nl{ate} \mid \nl{the}, \nl{mouse}) \to 0 \quad\Rightarrow\quad \perplexity_p(\nl{the}, \nl{mouse}, \nl{ate}, \nl{the}, \nl{cheese}) \to \infty.$$

- **Precision error**: The language model places extra probability mass on some bad sequences.  Perplexity provides a slap on the wrist.
  Given a language model $$p$$, suppose we mix in some garbage distribution $$r$$ with probability $$\epsilon$$:

$$q(x_i \mid x_{1:i-1}) = (1-\epsilon) p(x_i \mid x_{1:i-1}) + \epsilon r(x_i \mid x_{1:i-1}).$$

Then we can compute the perplexity of $$x_{1:L}$$ under $$q$$:

$$\perplexity_q(x_{1:L}) \le \frac{1}{1 - \epsilon} \perplexity_p(x_{1:L}) \approxeq (1 + \epsilon) \perplexity_p(x_{1:L}),$$

where the last approximate equality holds for small values of $$\epsilon$$.
If we mix in 5% junk, then perplexity only by 5%.
Note that the resulting language is horrible for generation,
since every 20 tokens on average it's just going to generate a gibberish token.

Now let's get on with evaluating perplexity on an actual dataset.

### Penn Tree Bank

The [Penn Tree Bank](https://catalog.ldc.upenn.edu/LDC99T42) is a classic dataset
in NLP, originally annotated for syntactic parsing.
Beginning with [Emami and Jelinek (2004)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1325968) and [Mikolov and Zweig (2012)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6424228), 
a version of the dataset that only contained Wall Street Journal articles was used as a language modeling evaluation. 
Note that the PTB language modeling benchmark involved some significant preprocessing of the original dataset (h/t to [John Hewitt](https://nlp.stanford.edu/~johnhew/) for pointing this out).

**Adaptation**.
Feed the entire text as a prompt into GPT-3 and evaluate the perplexity
([demo](http://crfm-models.stanford.edu/static/index.html?prompt=Pierre%20Vinken%2C%2061%20years%20old%2C%20will%20join%20the%20board%20as%20a%20nonexecutive%20director%20Nov.%2029.%20%20Mr.%20Vinken%20is%20chairman%20of%20Elsevier%20N.V.%2C%20the%20Dutch%20publishing%20group.&settings=echo_prompt%3A%20true%0Amax_tokens%3A%200%0Atop_k_per_token%3A%205%0Amodel%3A%20%24%7Bmodel%7D&environments=model%3A%20%5Bopenai%2Fdavinci%2C%20openai%2Fcurie%2C%20ai21%2Fj1-jumbo%5D)):

> *Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.  Mr. Vinken is chairman of Elsevier N.V., the Dutch publishing group.*

**Results**.
GPT-3 vastly outperforms the existing state-of-the-art:

|----------------------|-------------|
| Model                | Perplexity  |
| GPT-3                | **20.5**    |
| BERT-Large-CAs1      | 31.3        |

See the [leaderboard](https://paperswithcode.com/sota/language-modelling-on-penn-treebank-word) for the latest results.

**Train/test leakage**.  The authors did not evaluate on some datasets such as [WikiText-103](https://paperswithcode.com/dataset/wikitext-103)
because GPT-3 was trained on Wikipedia.
PTB had the advance of predating the Internet,
and is only available through a paid license.
This is another complication with large datasets:
it is difficult to check that your test data did not appear in your training data and was memorized.

### LAMBADA ([Paperno et al. 2016](https://arxiv.org/pdf/1606.06031.pdf))

- Task: predict the last word of a sentence.
- Motivation: Solving the task requires modeling **long-range dependencies**.

**Adaptation**.
- LAMBADA is natively already a language modeling task, so we could just ask a
  language model to complete the final word of the sentence.
- Problem: language model doesn't know it should be producing the final word of the sentence.
- Solution: frame it more explicitly as a input-output mapping and use in-context learning with additional examples
([demo](http://crfm-models.stanford.edu/static/index.html?prompt=Fill%20in%20blank%3A%0A%0AAlice%20was%20friends%20with%20Bob.%20Alice%20went%20to%20visit%20her%20friend%20___.%20-%3E%20Bob%0A%0AShe%20held%20the%20torch%20in%20front%20of%20her.%0AShe%20caught%20her%20breath.%0A%22Chris%3F%20%20There%E2%80%99s%20a%20step.%22%0A%22What%3F%22%0A%22A%20step.%20Cut%20in%20the%20rock.%20About%20fifty%20feet%20ahead.%22%20She%20moved%20faster.%20They%20both%20moved%20faster.%20%22In%20fact%2C%22%20she%20said%2C%20raising%20the%20torch%20higher%2C%20%22there%E2%80%99s%20more%20than%20a%20___.%20-%3E&settings=temperature%3A%200%0Amax_tokens%3A%201%0Atop_k_per_token%3A%2010%0Amodel%3A%20%24%7Bmodel%7D&environments=model%3A%20%5Bopenai%2Fdavinci%2C%20openai%2Fcurie%2C%20ai21%2Fj1-jumbo%5D)):

> *Fill in blank:<br>
> <br>
> Alice was friends with Bob. Alice went to visit her friend \_\_\_. -> Bob<br>
> <br>
> She held the torch in front of her.<br>
> She caught her breath.<br>
> "Chris?  There’s a step."<br>
> "What?"<br>
> "A step. Cut in the rock. About fifty feet ahead." She moved faster. They both moved faster. "In fact," she said, raising the torch higher, "there’s more than a \_\_\_. -> <span style="color:orange">step</span>*


**Results**.
GPT-3 does *much better* on this task than the previous state-of-the-art (based on GPT-2):

|----------------------|-------------|
| Model                | Perplexity  |
| GPT-3 (few-shot)     | **1.92**    |
| SOTA                 | 8.63        |

See the [leaderboard](https://paperswithcode.com/sota/language-modelling-on-lambada) for the latest results.

### HellaSwag ([Zellers et al. 2019](https://arxiv.org/pdf/1905.07830.pdf))

- Motivation: evaluate a model's ability to perform commonsense reasoning
- Task: choose the most appropriate completion for a sentence from a list of choices

**Adaptation**.
This is a **multiple-choice task**,
so the most natural thing to do is to **score** each candidate answer with the language model and predict the "best" one
([demo](http://crfm-models.stanford.edu/static/index.html?prompt=Making%20a%20cake%3A%20Several%20cake%20pops%20are%20shown%20on%20a%20display.%20%20A%20woman%20and%20girl%20are%20shown%20making%20the%20cake%20pops%20in%20a%20kitchen.%20%0A%20They%20%24%7Banswer%7D&settings=temperature%3A%200%0Amax_tokens%3A%200%0Atop_k_per_token%3A%205%0Aecho_prompt%3A%20true&environments=answer%3A%20%5B%22bake%20them%2C%20then%20frost%20and%20decorate.%22%2C%20%22taste%20them%20as%20they%20place%20them%20on%20plates.%22%2C%20%22put%20the%20frosting%20on%20the%20cake%20as%20they%20pan%20it.%22%2C%20%22come%20out%20and%20begin%20decorating%20the%20cake%20as%20well.%22%5D)):

> *Making a cake: Several cake pops are shown on a display.  A woman and girl are shown making the cake pops in a kitchen.  They ${answer}*<br>

where ${answer} is one of:

1. *bake them, then frost and decorate.*
1. *taste them as they place them on plates.*
1. *put the frosting on the cake as they pan it.*
1. *come out and begin decorating the cake as well.*

How do you score a candidate answer $$y$$ given a question $$x$$?  There's no principled answer,
but here are some **heuristics**:
1. Unnormalized probability: $$\text{score}(x, y) = p(x, y)$$.  The problem with the unnormalized probability is that
   it has a bias towards short answers ([demo](http://localhost:1959/static/index.html?prompt=Question%3A%20Why%20is%20the%20sky%20blue%3F%0AAnswer%3A%20%24%7Banswer%7D&settings=temperature%3A%200%0Amax_tokens%3A%200%0Atop_k_per_token%3A%2010%0Aecho_prompt%3A%20true&environments=answer%3A%20%5BIt%27s%20due%20to%20a%20phenomenon%20called%20Raleigh%20scattering%2C%20Because%20Mars%20is%20red%5D)).

1. Length-normalized probability: $$\text{score}(x, y) = \frac{p(x, y)}{\text{num-tokens}(y)}$$.
   This fixes the length bias.
   However, given two answers of the same length, the model still might prefer the more popular entity.

1. Frequency-normalized probability: $$\text{score}(x, y) = \frac{p(y \mid x)}{p(y \mid x_0)}$$, where $$x_0$$ is a neutral string like $$\nl{Answer:}$$.
   This lowers the score for answers that happen to just be common (e.g., \nl{John}).
   Compare [demo](http://localhost:1959/static/index.html?prompt=John%20pushed%20Khaleesi.%0AQuestion%3A%20Who%20was%20upset%3F%0AAnswer%3A%20%24%7Banswer%7D&settings=temperature%3A%200%0Amax_tokens%3A%200%0Atop_k_per_token%3A%2010%0Aecho_prompt%3A%20true&environments=answer%3A%20%5BJohn%2C%20Khaleesi%5D)
   versus [demo](http://localhost:1959/static/index.html?prompt=John%20pushed%20Bob.%0AQuestion%3A%20Who%20was%20upset%3F%0AAnswer%3A%20%24%7Banswer%7D&settings=temperature%3A%200%0Amax_tokens%3A%200%0Atop_k_per_token%3A%2010%0Aecho_prompt%3A%20true&environments=answer%3A%20%5BJohn%2C%20Bob%5D).

**Results**.
GPT-3 got close but did not exceed the state-of-the-art:

|----------------------|-------------|
| Model                | Accuracy    |
| SOTA                 | **85.6**    |
| GPT-3                | 79.3        |

However, the SOTA used fine-tuning on the HellaSwag training set, so it is
pretty impressive that GPT-3 can get close without any task-specific training data!

See the [leaderboard](https://paperswithcode.com/sota/sentence-completion-on-hellaswag) for the latest results.

## Question answering

Now we consider (closed-book) question answering, where the input is a question
and the output is an answer.  The **language model has to somehow "know" the answer**
without looking up information in a database or a set of documents (we'll
consider reading comprehension later, where the information is provided).

> Input: What school did burne hogarth establish?<br>
> Output: School of Visual Arts

### TriviaQA ([Joshi et al. 2017](https://arxiv.org/pdf/1705.03551.pdf))

- Task: given a trivia question, generate the answer
- The original dataset was collected from trivial enthusiasts and was presented
  as a challenge used for (open book) reading comprehension, but we use it for
  (closed-book) question answering.

**Adaptation**.
We define a prompt based on the training instances (if any) and the question,
and take the completion as the predicted answer
([demo](http://crfm-models.stanford.edu/static/index.html?prompt=Q%3A%20%E2%80%98Nude%20Descending%20A%20Staircase%E2%80%99%20is%20perhaps%20the%20most%20famous%20painting%20by%20which%0A20th%20century%20artist%3F%0AA%3A&settings=temperature%3A%200%0Astop_sequences%3A%20%5B%5Cn%5D&environments=)):

> *Q: ‘Nude Descending A Staircase’ is perhaps the most famous painting by which<br>
> 20th century artist?<br>
> A: <font style="color:orange">Marcel Duchamp</font>*

**Results**.

|----------------------|-------------|
| Model                | Accuracy    |
| RAG                  | 68.0        |
| GPT-3 (zero-shot)    | 64.3        |
| GPT-3 (few-shot)     | **71.2**    |

We also see that both increasing the model size and the number of in-context training instances helps:

![GPT-3 performance on TriviaQA](../images/gpt3-trivia-qa-performance.png)

### WebQuestions ([Berant et al. 2013](https://aclanthology.org/D13-1160.pdf))

- Task: answer questions
- Dataset collected from Google search queries, initially created for question answering on knowledge bases

**Adaptation**.

We define a prompt the same as above
([demo](http://crfm-models.stanford.edu/static/index.html?prompt=Q%3A%20What%20school%20did%20burne%20hogarth%20establish%3F%0AA%3A&settings=temperature%3A%200%0Astop_sequences%3A%20%5B%5Cn%5D&environments=)):

> *Q: What school did burne hogarth establish?<br>
> A: <font style="color:orange">School of Visual Arts</font>*

**Results**.

|----------------------|-------------|
| Model                | Accuracy    |
| RAG                  | **45.5**    |
| GPT-3 (zero-shot)    | 14.4        |
| GPT-3 (few-shot)     | 41.5        |

### NaturalQuestions

- Task: answer questions
- Dataset collected from Google search queries (with long-form answers)

**Adaptation**.
We define a prompt the same as above
([demo](http://crfm-models.stanford.edu/static/index.html?prompt=Q%3A%20Who%20played%20tess%20on%20touched%20by%20an%20angel%3F%0AA%3A&settings=temperature%3A%200%0Astop_sequences%3A%20%5B%5Cn%5D&environments=)):

> *Q: Who played tess on touched by an angel?<br>
> A: <font style="color:orange">Delloreese Patricia Early (July 6, 1931 - November 19, 2017), known professionally as Della Reese.</font>*

**Results**.

|----------------------|-------------|
| Model                | Accuracy    |
| RAG                  | **44.5**    |
| GPT-3 (zero-shot)    | 14.6        |
| GPT-3 (few-shot)     | 29.9        |

## Translation

- Task: translate a sentence in a source language (e.g., German) to sentence in a target language (e.g., English)
- Machine translation has been a long standing NLP task since the 1960s,
  and statistical machine translation took off within NLP (with its own distinct subcommunity) in the 2000s,
  followed by neural machine translation in the mid-2010s.
  It has always been a data-rich field due to the existence of human translators.
- The standard evaluation dataset is the
  [WMT'14](https://paperswithcode.com/dataset/wmt-2014) and
  [WMT'16](https://paperswithcode.com/dataset/wmt-2016) datasets.
- Since there are multiple possible translations, the (automatic) evaluation
  metric is BLEU (which captures a notion of n-gram overlap).

**Adaptation**.
For the few-shot setting, we construct a prompt containing input-output
training instances along with the input
([demo](http://crfm-models.stanford.edu/static/index.html?prompt=Mein%20Haus%20liegt%20auf%20dem%20H%C3%BCgel.%20%3D%20My%20house%20is%20on%20the%20hill.%0AKeinesfalls%20d%C3%BCrfen%20diese%20f%C3%BCr%20den%20kommerziellen%20Gebrauch%20verwendet%20werden.%20%3D&settings=temperature%3A%200%0Astop_sequences%3A%20%5B%5Cn%5D&environments=)):

> *Mein Haus liegt auf dem Hügel. = My house is on the hill.<br>
> Keinesfalls dürfen diese für den kommerziellen Gebrauch verwendet werden. = <font style="color:orange">In no case may they be used for commercial purposes.</font>*

<!--Under no circumstances may these be used for commercial purposes.-->

**Results**.
Here are the results from German to English:

|----------------------|-------------|
| Model                | Accuracy    |
| SOTA (supervised)    | 40.2        |
| GPT-3 (zero-shot)    | 27.2        |
| GPT-3 (few-shot)     | 40.6        |

- Even without supervised training data, GPT-3 matches the state-of-the-art of
  a fully-supervised system!
- This presents a lower bound on how well one can do in machine translation;
  you would definitely want to leverage the large amount of parallel corpora
  (aligned input-output pairs).
- Results from French and Romanian are similar.
- Results from English to a foreign language is much worse, which is
  expected since GPT-3 is primarily an English language model.

## Arithmetic

GPT-3 is a language model (primarily on English), but we can evaluate it on a
range of more "abstract reasoning" tasks,
to evaluate GPT-3 as more of a general-purpose model.

- Task: do arithmetic (2-5 digit addition, subtraction, multiplication)
- There's no practical reason you would want to solve this task; it's just a
  diagnostic task to satisfy our scientific curiosity.

**Adaptation**.
Pose the problem as question answering ([demo](http://crfm-models.stanford.edu/static/index.html?prompt=Q%3A%20What%20is%20556%20plus%20497%3F%0AA%3A&settings=temperature%3A%200%0Astop_sequences%3A%20%5B%5Cn%5D&environments=)):

> *Q: What is 556 plus 497?<br>
> A: <font style="color:orange">1053</font>*

**Results**.

![GPT-3 performance on arithmetic tasks](../images/gpt3-arithmetic-performance.png)

It doesn't work perfectly and can hardly be said to "understand arithmetic"
fully, but it works surprisingly well.

## News article generation

- Task: given title and subtitle, generate a news article
- Dataset: title/subtitles taken from [newser.com](newser.com)
- Evaluation: humans rated articles based on how likely the article was likely to be written by a machine

**Adaptation**.
Note: in-context learning was needed to give the model an idea of what a prompt looks like.

> *Title: United Methodists Agree to Historic Split<br>
> Subtitle: Those who oppose gay marriage will form their own denomination<br>
> Article: <font style="color:orange">After two days of intense debate, the
> United Methodist Church has agreed to a historic split - one that is expected
> to end in the creation of a new denomination, one that will be "theologically
> and socially conservative," according to The Washington Post. The majority of
> delegates attending the church's annual General Conference in May voted to
> strengthen a ban on the ordination of LGBTQ clergy and to write new rules
> that will "discipline" clergy who officiate at same-sex weddings. But
> those who opposed these measures have a new plan: They say they will form a
> separate denomination by 2020, calling their church the Christian Methodist
> denomination...</font>*

**Results**.
Humans were able to able to detect classify "human" versus "machine" only 52% of
the time (barely above random chance).

For the article above, humans guessed "machine" correctly only 12% of the time.

## Novel tasks

### Using new words

- Task: given a new made-up word and a definition, generate a sentence that uses the word.

**Adaptation**.
Just describe the task in the prompt ([demo](http://crfm-models.stanford.edu/static/index.html?prompt=To%20%22screeg%22%20something%20is%20to%20swing%20a%20sword%20at%20it.%20%20An%20example%20of%20a%20sentence%20that%20uses%20the%20word%20screeg%20is%3A%0AWe&settings=stop_sequences%3A%20%5B%5Cn%5D%0Anum_completions%3A%2010%0Atemperature%3A%200.5&environments=)):

> *To "screeg" something is to swing a sword at it.  An example of a sentence that uses the word screeg is:
> We <font style="color:orange">screeged the tree with our swords.</font>*

### Correcting English grammar

- Task: given an ungrammatical sentence, generate its grammatical version.

**Adaptation**.
The prompt consists of input-output pairs ([demo](http://crfm-models.stanford.edu/static/index.html?prompt=Poor%20English%20input%3A%20I%20eated%20the%20purple%20berries.%0AGood%20English%20output%3A%20I%20ate%20the%20purple%20berries.%0APoor%20English%20input%3A%20Thank%20you%20for%20picking%20me%20as%20your%20designer.%20I%E2%80%99d%20appreciate%20it.%0AGood%20English%20output%3A%20Thank%20you%20for%20choosing%20me%20as%20your%20designer.%20I%20appreciate%20it.%0APoor%20English%20input%3A%20The%20mentioned%20changes%20have%20done.%20or%20I%20did%20the%20alteration%20that%20you%0Arequested.%20or%20I%20changed%20things%20you%20wanted%20and%20did%20the%20modifications.%0AGood%20English%20output%3A%20The%20requested%20changes%20have%20been%20made.%20or%20I%20made%20the%20alteration%20that%20you%0Arequested.%20or%20I%20changed%20things%20you%20wanted%20and%20made%20the%20modifications.%0APoor%20English%20input%3A%20I%E2%80%99d%20be%20more%20than%20happy%20to%20work%20with%20you%20in%20another%20project.%0AGood%20English%20output%3A&settings=stop_sequences%3A%20%5B%5Cn%5D%0Atemperature%3A%200%0Atop_k_per_token%3A%205&environments=)):

> *Poor English input: I eated the purple berries.<br>
> Good English output: I ate the purple berries.<br>
> Poor English input: Thank you for picking me as your designer. I’d appreciate it.<br>
> Good English output: Thank you for choosing me as your designer. I appreciate it.<br>
> Poor English input: The mentioned changes have done. or I did the alteration that you<br>
> requested. or I changed things you wanted and did the modifications.<br>
> Good English output: The requested changes have been made. or I made the alteration that you<br>
> requested. or I changed things you wanted and made the modifications.<br>
> Poor English input: I’d be more than happy to work with you in another project.<br>
> Good English output: <font style="color:orange">I would be happy to work with you on another project.</font>*

## Other tasks

Since the original paper, GPT-3 has been applied to many more tasks, including
benchmark datasets and one-off demos.  Here is an non-exhaustive list.

**Benchmarks**.
- [SWORDS](https://arxiv.org/pdf/2106.04102.pdf): lexical substitution, where the goal is to predict synonyms in the context of a sentence.
- [Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300.pdf): 57 multiple-choice problems spanning mathematics, US history, computer science, law, etc.
- [TruthfulQA](https://arxiv.org/pdf/2109.07958.pdf): question answering dataset that humans would answer falsely due to misconceptions.

The performance on these benchmarks is still mediocre, but it's perhaps not bad
given that we're doing few-shot learning!

**Demos**.
- [Examples from the OpenAI website](https://beta.openai.com/examples/)
- [Examples from gpt3demo.com](https://gpt3demo.com/)

The demos are creative and interesting, but it's hard to tell how reliably they work.

### Summary

- GPT-3 was evaluated on a wide range of standard NLP benchmarks and on quirky one-off tasks.
- GPT-3 can perform extremely well or be very medicore.
- Both increasing the size of the model and the number of examples helps performance.
- There are a few heuristic ways of adapting the language model to the task of interest.
- Why does this work?  No one knows.

## Further reading

- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf). *Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, J. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. Henighan, R. Child, A. Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei*. NeurIPS 2020.
- [Blog post explaining perplexity](https://towardsdatascience.com/perplexity-in-language-models-87a196019a94)

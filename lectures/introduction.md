---
layout: page
parent: Lectures
title: Introduction
nav_order: 1
usemathjax: true
---
$$
\newcommand{\nl}[1]{\text{[#1]}}
$$

*UNDER CONSTRUCTION*

Hello, welcome to CS324!  This is a new course on understanding and developing **large language models**.

## Why does this course exist?

## Structure of this course

1. [Behavior](/lectures/behavior)
1. [Data](/lectures/data)
1. [Building](/lectures/building)

## What is a language model?

The most classic definition of a language model is a *probability distribution over sequences of tokens*.
For each sequence of tokens $$w_1, \dots, w_L$$, the language model assigns a probability (a number between 0 and 1):

$$p(w_1, \dots, w_L).$$

For example, the language model might assign:

$$p(\nl{The}, \nl{fox}, \nl{jumps}, \nl{.}) = 0.001.$$

(We have assumed that sentences are [tokenized](../tokenization) into sequence of tokens.)

## Further reading

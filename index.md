---
layout: default
title: Home
nav_index: 1
seo:
  type: Course
  name: {{ site.title }}
---

# {{ site.tagline }}

<!--{% if site.announcements %}
{{ site.announcements.last }}
[Announcements](announcements.md){: .btn .btn-outline .fs-3 }
{% endif %}-->

WARNING: this website is still under construction!  Nothing is final!

## Description

The field of natural language processing (NLP) has been transformed by massive
pre-trained language models.  They form the basis of all state-of-the-art
systems across a wide range of tasks and have shown an impressive ability to
generate fluent text and perform few-shot learning.  At the same time, these
models are hard to understand and give rise to new ethical and scalability
challenges.  In this course, students will learn the fundamentals about the
modeling, theory, ethics, and systems aspects of massive language models, as
well as gain hands-on experience working with them.

## Logistics

TODO: Where/when

TODO: policies

## Grading

Two assignments + final project

## Teaching team

{% assign instructors = site.staffers | sort: 'index' | where: 'role', 'Instructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

{% assign teaching_assistants = site.staffers | sort: 'index' | where: 'role', 'Course Assistant' %}
{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}

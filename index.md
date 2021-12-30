---
layout: default
title: Home
seo:
  type: Course
  name: {{ site.title }}
nav_order: 1
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

**Where**: In-person lectures and discussion will be at 200-002 (History Corner). Remote lectures will be on Zoom (Zoom link in the Google Calendar event below). The first two weeks of class will be remote (subject to change according to University policies).

**When**: Mondays and Wednesdays 3:15-4:45pm PST [\[Google Calendar\]](https://calendar.google.com/event?action=TEMPLATE&tmeid=MGUwdHFqdDAycmhicmZhZTBvdDh1bWg0YnNfMjAyMjAxMDNUMjMxNTAwWiBzeGllQHN0YW5mb3JkLmVkdQ&tmsrc=sxie%40stanford.edu&scp=ALL)

**Links**:
- [Gradescope](https://www.gradescope.com/courses/342794)
- [Canvas](https://canvas.stanford.edu/courses/149841)
- [Ed](https://canvas.stanford.edu/courses/149841/external_tools/24287?display=borderless)
- [Slack](https://canvas.stanford.edu/courses/149841/external_tools/11232)


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

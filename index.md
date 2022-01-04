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

The field of natural language processing (NLP) has been transformed by massive
pre-trained language models.  They form the basis of all state-of-the-art
systems across a wide range of tasks and have shown an impressive ability to
generate fluent text and perform few-shot learning.  At the same time, these
models are hard to understand and give rise to new ethical and scalability
challenges.  In this course, students will learn the fundamentals about the
modeling, theory, ethics, and systems aspects of large language models, as
well as gain hands-on experience working with them.

## Teaching team

{% assign instructors = site.staffers | sort: 'index' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

## Logistics

**Where**: Class will by default be in person at
[200-002](https://goo.gl/maps/8ADRSg7nJ9xZC2Zd7) (History Corner).  The first
two weeks will be remote (in accordance with University policies);
[Zoom information](https://canvas.stanford.edu/courses/149841/external_tools/5384) is posted on Canvas.

**When**: Class is Mondays and Wednesdays 3:15-4:45pm PST.

**Links**:
- [Ed](https://canvas.stanford.edu/courses/149841/external_tools/24287?display=borderless):
  This is the main way that you and the teaching team should communicate:
  we will post all important announcements here, and you should ask
  all course-related questions here.
  For personal matters that you don't wish to put in a private Ed post, you can
  email the teaching staff at [cs324-win2122-staff@lists.stanford.edu](mailto:cs324-win2122-staff@lists.stanford.edu).
- [Canvas](https://canvas.stanford.edu/courses/149841): The course Canvas page
  contains links and resources only accessible to students ([Zoom link](https://canvas.stanford.edu/courses/149841/external_tools/5384) for
  remote classes).
- [Gradescope](https://www.gradescope.com/courses/342794): We use Gradescope
  for managing coursework (turning in, returning grades).  Please use your
  @stanford.edu email address to sign up for a Gradescope account.

**Video access disclaimer**: A portion of class activities will be given and
recorded in Zoom. For your convenience, you can access these recordings by
logging into the course Canvas site. These recordings might be reused in other
Stanford courses, viewed by other Stanford students, faculty, or staff, or used
for other education and research purposes. If you have questions, please
contact a member of the teaching team at [cs324-win2122-staff@lists.stanford.edu](mailto:cs324-win2122-staff@lists.stanford.edu).

## Class

Each class is divided into two parts:

1. **Lecture** (45 minutes): an instructor gives a standard lecture on a topic
   (see the [calendar](/calendar) for the list of topics).  Lectures are be
   based on [these lecture notes](/lectures).

1. **Discussion** (45 minutes): there is a student panel discussion on the
   required readings posted on the [calendar](/calendar).

## Coursework

Your grade is based on two activities:

1. Paper reviews and discussion (20%)
1. Projects (2 x 40% = 80%)

### 1. Paper reviews and discussions

[**Paper reviews**](paper-reviews).  Before each class, you will be assigned 1-2 papers.  You
should read these papers carefully and write a review of the paper(s).
Your review should be a few paragraphs (in the style of a conference review,
say for ACL or NeurIPS).

Paper reviews are due at **11:00 AM PST on
[Gradescope](https://www.gradescope.com/courses/342794)** on the day of the
lecture.

[**Paper discussions**](paper-discussions).  During
each class discussion, there is a panel of 4-5 students (you are expected to
sign up for at least two panels).  The student panelists lead a discussion
moderated by the instructors.  Everyone else is expected to participate by
asking the panel questions.

### 2. Projects

There are two [projects](projects), which allow you to get hands-on experience with large
language models.

- [**Project 1**](projects/project1) is on **evaluating** language models.  You will be
  provided with access to models such as GPT-3 and asked to think critically
  about their capabilities and risks.  You will identify a **focal property**
  of language models that you'd like to explore more deeply.

- [**Project 2**](projects/project2) is on **building** language models.  You will be
  provided with a compute budget that will allow you to train models such as
  BERT-base to more systematically evaluate, understand, and improve language
  models along the focal property you identified in project 1.

Projects should be done in **groups of 1-2 students**.
Each project should be written up clearly and succinctly; you may lose points if
your writing is unclear or unnecessarily complicated. Projects must be typeset
using LaTeX, Microsoft Word, Pages for Mac, or an equivalent program, and
submitted as a PDF. We strongly encourage you to use LaTeX: there are
user-friendly web interfaces like [Overleaf](https://www.overleaf.com/).

Projects are due at **11:00 PM (not 11:59 PM) PST on
[Gradescope](https://www.gradescope.com/courses/342794)** on the due date.

### Submitting coursework

**Submissions**: All coursework are submitted via
[Gradescope](https://www.gradescope.com/courses/342794) by the deadline.
Do not submit your coursework via email. If anything goes wrong, please ask
a question on Ed or contact a course assistant. If you need to sign up for a
Gradescope account, please use your @stanford.edu email address. You can submit
as many times as you'd like until the deadline: we will only grade the last
submission. Partial work is better than not submitting any work. If you are
working in a group for a homework, **please make sure all group members are
selected as part of the submission on Gradescope.**

**Late days**: A homework is ⌈d⌉ days late if it is turned in d days past the
due date (note that this means if you are 1 second late, ⌈d⌉=1 and it is 1 day
late). **You have 3 late days in total that can be distributed among the
homeworks without penalty.** After that, the maximum possible grade is
decreased by 25% each day (so the best you can do with d=1 is 75%; paper
reviews will be graded as fails if they are late and you have no late days). As
an example, if you are out of late days and submit one day late, a 90 will be
capped at a 75, but a 72 will not be changed. **Note that we will only allow a
max of d=2 late days per homework.**

**Regrades**: If you believe that the course staff made an objective error in
grading, then you may submit a regrade request. **Regrade requests will only be
accepted via Gradescope for one week after the initial grades were released.**
Note that we may regrade your entire submission, so that depending on your
submission you may actually lose more points than you gain.

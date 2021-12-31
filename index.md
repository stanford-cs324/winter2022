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
modeling, theory, ethics, and systems aspects of large language models, as
well as gain hands-on experience working with them.

## Logistics

**Where**: In-person lectures and discussion will be at [200-002](https://goo.gl/maps/8ADRSg7nJ9xZC2Zd7) (History Corner). Remote lectures will be on Zoom (posted on Canvas). The first two weeks of class will be remote (subject to change according to University policies).

**When**: Mondays and Wednesdays 3:15-4:45pm PST. Office hours for the teaching staff are listed under [Teaching team](#teaching-team).

**Links**:
- [Gradescope](https://www.gradescope.com/courses/342794): We will use Gradescope for turning in assignments. If you need to sign up for a Gradescope account, please use your @stanford.edu email address. 
- [Canvas](https://canvas.stanford.edu/courses/149841): The course Canvas page will contain the course resources and links, as well as the Zoom link for remote lectures.
- [Ed](https://canvas.stanford.edu/courses/149841/external_tools/24287?display=borderless): Students should ask course-related questions in the Ed forum. For external enquiries, emergencies, or personal matters that you don't wish to put in a private Ed post, you can email the teaching staff directly.
- [Slack](https://canvas.stanford.edu/courses/149841/external_tools/11232): We will use Slack as a platform for informal discussions and announcements.

**Video access disclaimer**: A portion of class activities will be given and recorded in Zoom. For your convenience, you can access these recordings by logging into the course Canvas site. These recordings might be reused in other Stanford courses, viewed by other Stanford students, faculty, or staff, or used for other education and research purposes. If you have questions, please contact a member of the teaching team.

## Grading

- **Paper reviews and class participation** (20%): For each lecture, 1-2 papers will be assigned that should be read in advance. For each lecture, a brief review of the papers should be submitted prior to lecture (that will be graded pass/fail) and students should participate in the discussion of the papers.  
- **Assignments** (2 x 40%): There will be two assignments (A1 and A2). Assignments should be done in groups of 2-3 students. Deadlines and details will be released soon.    

## Homeworks

**Paper reviews**: Paper reviews are due at **3:00 PM Pacific Time on [Gradescope](https://www.gradescope.com/courses/342794)** on the day of the lecture. Reviews should address: (i) the main contributions of the paper(s), (ii) the strengths and weaknesses of the paper, and (iii) questions/discussion items you have based on the work. For each lecture, a few students will be designated (in advance) to provide the initial questions/discussion items for the lecture's discussion based on their responses in (iii). 

**Assignments**: Assignments are due at **11:00 PM (not 11:59) Pacific Time on [Gradescope](https://www.gradescope.com/courses/342794)** on the due date. Assignments should be written up clearly and succinctly; you may lose points if your answers are unclear or unnecessarily complicated. Assignments must be typeset using LaTeX, Microsoft Word, Pages for Mac, or an equivalent program, and submitted as a PDF. We strongly encourage you to use LaTeX: there are user-friendly web interfaces like [Overleaf](https://www.overleaf.com/).

## Submissions

**Electronic submissions**: All homeworks are submitted via [Gradescope](https://www.gradescope.com/courses/342794) at either 3:00 PM Pacific Time (paper reviews) or 11:00 PM Pacific Time (assignments) on the due date. Do not submit your assignment via email. If anything goes wrong, please ask a question on Ed or contact a course assistant. If you need to sign up for a Gradescope account, please use your @stanford.edu email address. You can submit as many times as you'd like until the deadline: we will only grade the last submission. Partial work is better than not submitting any work. If you are working in a group for a homework, **please make sure all group members are selected as part of the submission on Gradescope.** 

**Regrades**: If you believe that the course staff made an objective error in grading, then you may submit a regrade request. **Regrade requests will only be accepted via Gradescope for one week after the initial grades were released.** Note that we may regrade your entire submission, so that depending on your submission you may actually lose more points than you gain.

**Late days**: A homework is ⌈d⌉ days late if it is turned in d days past the due date (note that this means if you are 1 second late, ⌈d⌉=1 and it is 1 day late). You have TBD late days in total that can be distributed among the homeworks without penalty. After that, the maximum possible grade is decreased by 25% each day (so the best you can do with d=1 is 75%; paper reviews will be graded as fails if they are late and you have no late days). As an example, if you are out of late days and submit one day late, a 90 will be capped at a 75, but a 72 will not be changed. Note that we will only allow a max of d=2 late days per homework.

## Teaching team

{% assign instructors = site.staffers | sort: 'index' | where: 'role', 'Instructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

{% assign teaching_assistants = site.staffers | sort: 'index' | where: 'role', 'Course Assistant' %}
{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}

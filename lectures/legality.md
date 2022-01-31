---
layout: page
parent: Lectures
title: Legality
nav_order: 3.3
usemathjax: true
---
- In this lecture, we will discuss what the **law** has to say about the
development and deployment of large language models.
- As with previous lectures, for example the one on social bias,
much of what we will discuss is not necessarily specific to large language models
(there is no Large Language Model Act).
- But whenever a **new powerful technology** emerges,
it raises many questions about whether existing laws still apply or make sense.
- For example, [**Internet law**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3191751) (or cyberlaw) has emerged with the rising importance of the Internet.
  * It draws from existing fields such as intellectual property law, privacy law, and contract law.
  * Judge Frank Easterbrook used the term [Law of the Horse](https://en.wikipedia.org/wiki/Law_of_the_Horse) in 1996 to question
    why Internet law should be its own section of legal studies and litigation.
- But the Internet clearly has its own unique challenges:
  * Laws usually had clear **jurisdiction** (e.g., state, federal), but the Internet is not geographically bound.
  * It is possible to remain **anonymous** on the Internet.
  * **Anyone can post** a piece of content that in principle can get be viewed by anyone.

**Non-legal considerations**.
There is a distinction between **law** and **ethics**.
- Law is enforceable by government, whereas
- ethics is not enforceable and can be created by any organization.
- Examples of [code of conducts](https://en.wikipedia.org/wiki/Code_of_conduct), which aren't legal, but nonetheless important:
  * [Hippocratic Oath](https://en.wikipedia.org/wiki/Hippocratic_Oath): from Ancient Greece, physicians swear to do no harm, respect privacy of patients, etc.
  * [ACM Code of Ethics and Professional Conduct](https://www.acm.org/code-of-ethics)
  * [NeurIPS code of conduct](https://neurips.cc/public/CodeOfConduct): no harassment, no plagiarism
  * [Stanford Honor Code](https://communitystandards.stanford.edu/policies-and-guidance/honor-code): no plagiarism, giving/receiving aid on an exam
- We will focus on law in this lecture, but let us not forget about ethics and norms, which is can be more agile.

**Jurisdiction**.
Depending on where you live (which country, which state, etc.), which laws apply vary.
- Different **countries** (United States, China, EU) have different laws.
  * For example, the EU's data privacy laws from [GDPR](https://gdpr-info.eu/) are much more comprehensive that what exists in the United States.
- Laws can exist at the **federal, state, or local** level.
  * For example, California has privacy laws via the [California Consumer Privacy Act](https://en.wikipedia.org/wiki/California_Consumer_Privacy_Act),
    which is analogous to GDPR, but has no federal counterpart.
  * In Baldwin Park, California, it is illegal to ride a bicycle in a swimming pool ([reference](http://www.laalmanac.com/crime/cr08a.php)).
- We will focus by default on United States, but will mention the EU at various times,
  since the EU are leading the charge with data privacy (GDPR) and AI regulation ([EU AI Act](https://www.eipa.eu/publications/briefing/the-artificial-intelligence-act-proposal-and-its-implications-for-member-states/)).

**Types of law**.
- **Common law** (judiciary): Also known as case law,
  common law is based on judges referencing previous similar cases and making a ruling (**precedent**).
  * Example of a case (lawsuit): Oracle v. Google
- **Statutory law** (legislature):
  Also known as written law, statutory law is produced by government agencies through the legislative process (e.g., congress passing a bill).
  * Example of a statute: Copyright Act of 1976
  * Often common law exists for a while before being codified into a statute (fair use was common law since the 1840s and finally became codified in 1976).
- **Regulatory law** (executive): Also known as administrative law, this is law that is created by the executive branch of government,
  often focusing on procedures.
  * Example: the legislative branch passes a law authorizing the creation of a new executive agency (e.g., Environmental Protection Agency),
    and then the EPA passes regulations to meet its mandate.

**Large language models**. Now let turn our attention to large language models.
Recall the **lifecycle** of a large language model:
1. Collect training data (e.g., Common Crawl).
1. Train a large language model (e.g., GPT-3).
1. Adapt it to downstream tasks (e.g., dialogue).
1. Deploy the language model to users (e.g., customer service chatbot).

There are two main areas where the law intersects the large language models lifecycle:
- **Data**.
  * All machine learning relies on **data**.
  * Language models rely on a lot of data, especially other people's data made for a different purpose, and often scraped without consent.
  * **Copyright law** protects creators (of data). Is training language models on this data a copyright violation?
  * **Privacy law** protects individuals right to privacy.  Can training language models on either public or private data violate privacy?
    For private data, when is collection and aggregation of this data even allowed?
  * While these laws are centered around data, also relevant is what you do with the data.
- **Applications**.
  * Language models can be used for a wide range of downstream tasks (e.g., question answering, chatbots).
  * Technologies can be used [intentionally for harm](https://crfm.stanford.edu/assets/report.pdf#misuse) (e.g., spam, phishing attacks, harassment, disinformation).
    Existing Internet fraud and abuse laws might cover some of this.
  * They could be deployed in various **high-stakes** settings (e.g., healthcare, lending, education).
    Existing regulation in the respective areas (e.g., healthcare) could cover some of this.
  * Of course, the expanded capabilities of large language models (e.g., realistic text generation, chatbots) will introduce new challenges.

Today, we will mostly focus on [copyright law](#copyright-law).

## Copyright law

Large language models, or any machine learning model,
is trained on data, which results from the fruits of a human being's labor (e.g., author, programmer, photographer, etc.).
What can someone other than the creators can do with these creations (e.g., books, code, photographs, etc.)
is in the realm of intellectual property law.

**Intellectual property law**.
- Motivation: encourage the creation of a wide variety of intellectual goods.
  If anyone could just take your hard work and profit from it, people would be less incentivized to create or share.
- Types of intellectual property: copyright, patents, trademarks, trade secrets.

[**Copyright law**](https://en.wikipedia.org/wiki/Copyright).
The key legislation that governs copyright in the United States is [Copyright Act of 1976](https://en.wikipedia.org/wiki/Copyright_Act_of_1976).
- Copyright protection applies to "original works of authorship **fixed** in any
  tangible medium of **expression**, now known or later developed, from which they
  can be perceived, reproduced, or otherwise communicated, either directly or
  with the aid of a machine or device".
- Expanded scope from "published" (1909) to "fixed", basing on the [Berne Convention](https://en.wikipedia.org/wiki/Berne_Convention) of 1886.
- **Registration is not required** for copyright protection (in contrast with patents).
- Registration is required before creator can sue someone for copyright infringement.
- Note: the threshold for copyright is extremely low (you have copyright protection on many things you probably didn't realize).
- Lasts for 75 years, and then the copyright expires and it becomes part of the **public domain** (works of Shakespeare, Beethoven, etc.).
  Most of [Project Gutenberg](https://www.gutenberg.org/) are books in the public domain.

There are two ways you can use a copyrighted work:
1. Get a license for it.
1. Appeal to the fair use clause.

[**Licenses**](https://en.wikipedia.org/wiki/License).
- A **license** (from contract law) is granted by a licensor to a licensee.
- Effectively, "a license is a promise not to sue".
- The [Creative Commons license](https://en.wikipedia.org/wiki/Creative_Commons_license), enable free distribution of copyrighted work.
- [Examples](https://en.wikipedia.org/wiki/List_of_major_Creative_Commons_licensed_works) include Wikipedia, Open Courseware, Khan Academy,
  Free Music Archive, 307 million images from Flickr, 39 million images from MusicBrainz, 10 million videos from YouTube, etc.

**Fair use (section 107)**.
- Previously common law since the 1840s.
- Four factors to determine whether fair use applies:
  * the **purpose** and character of the use (educational favored over commercial, **transformative** favored over reproductive);
  * the nature of the copyrighted work (fictional favored over factual, the degree of creativity);
  * the amount and substantiality of the portion of the **original work** used; and
  * the effect of the use upon the **market** (or potential market) for the original work.
- Example of fair use: watch a movie, write a summary of it
- Example of fair use: reimplement an algorithm (the idea) rather than copying the code (the expression).

**Terms of service**.
There is one additional hurdle: [terms of service](https://en.wikipedia.org/wiki/Terms_of_service),
which might impose additional restrictions.
- Example: YouTube's terms of service prohibits downloading videos, even if the videos are licensed under Creative Commons.

**Notes**:
- Facts and ideas are not copyrightable.
- Database of facts can be copyrightable if curation / arrangement is considered expression.
- Copying data (first step of training) is violation already even if you don't do anything with it.
- Statutory damages are up to $150,000 per work (Section 504 of Copyright Act).
- Plaintiffs are small (owners of books), defendants are big companies.

Next, we will go over a number of cases that have ruled for or against fair use.

[Authors Guild v. Google](https://en.wikipedia.org/wiki/Authors_Guild,_Inc._v._Google,_Inc.)
- Google Book Search scanned printed books and made them searchable online (showed snippets), launched in 2002.
- [Authors Guild](https://en.wikipedia.org/wiki/Authors_Guild) complained that Google had not sought their permission for books still protected by copyright.
- 2013: District Court granted [summary judgment](https://en.wikipedia.org/wiki/Summary_judgment) in favor of Google, deemed **fair use**.

[Google v. Oracle](https://en.wikipedia.org/wiki/Google_LLC_v._Oracle_America,_Inc.)
- Google replicated 37 Java APIs in Android operating system that was owned by Oracle (formerly Sun Microsystems).
- Oracle sued Google for copyright infringement.
- April 2021: Supreme Court ruled that Google's use of Java APIs covered by **fair use**.

[Fox News v. TVEyes](https://www.lexisnexis.com/community/casebrief/p/casebrief-fox-news-network-llc-v-tveyes-inc)
- TVEyes recorded television programming, created a service that allows people to search (via text) and watch 10-second clips.
- Fox News sued TVEyes.
- 2018: 2nd district ruled in favor of Fox News, **not fair use**.
- Justification: While transformative, deprives Fox News of revenue.

[Kelly v. Arriba](https://en.wikipedia.org/wiki/Kelly_v._Arriba_Soft_Corp.)
- Arriba created a search engine that shows thumbnails.
- Kelly (an individual) sued Arriba.
- 2003: 9th circuit ruled in favor of favor Arriba, deemed it **fair use**.

[Sega v. Accolade](https://en.wikipedia.org/wiki/Sega_v._Accolade)
- Sega Genesis game console released in 1989.
- Accolade wanted to release games on Genesis, but Sega charged extra, wants to be exclusive publisher.
- Accolade reverse engineered Sega's code to make new version, bypassing security lockouts.
- Sega sued Accolade in 1991.
- 1992: 9th circuit ruled in favor of Accolade, deeming it **fair use** (mostly original content, competition benefits public, no evidenced it diminished Sega's market).
- "Non-expressive": Accessing ideas, facts, not expression

[Fair learning](https://texaslawreview.org/fair-learning/) argues that machine learning is fair use:
- ML system's use of data is **transformative**, doesn't change work, but changes purpose.
- ML system is interested in **idea** (e.g., stop sign) not in the concrete **expression** (e.g., exact artistic choices of a particular image of a stop sign).
- Arguments for ML as fair use:
  * Broad access to training data makes better systems for society.
  * If don't allow, then most works cannot be used to produce new value.
  * Using copyrighted data can be more fair [Levendowski, 2018](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3024938).
- Arguments against ML as fair use:
  * Argue that ML systems don't produce a creative "end product" but just make money.
  * Generative models (e.g., language models) can compete with creative professionals.
  * Problems with ML systems (spread disinformation, enable surveillance, etc.), so don't give ML systems the benefit of the doubt.
- Challenge: hard to separate protectable (e.g., expression) from unprotectable (e.g., ideas).
- There are many reasons why building an ML system might be bad, but is copyright the right tool to stop it?

Whether training large language models is fair use is rapidly evolving.
Looking back at the history of information technology, we see three phases:
- First phase: text data mining (search engines), based on simple pattern matching.
- Second phase: classification (e.g., classify stop signs or sentiment analysis), recommendation systems.
- Third phase: **generative models** that learn to mimic expression.
  * Last time, we saw that it was possible to [extract training data from GPT-2](https://arxiv.org/pdf/2012.07805.pdf),
    which was potentially problematic from a point of view of privacy.
  * If a language model spits out Harry Potter verbatim, this is problematic for fair use.
  * However, even if the language model doesn't generate previous works verbatim,
    copyright is still relevant since the previous copyrighted works were used to train the language model.
  * In fact, a language model can **compete** with writers.  For example, a writer writes 3 books, a language model trains on these 3 books, and auto-generates the 4th.
- Conclusion: the future of copyright and machine learning in light of large language models is very much open.

## Privacy law

Next we will briefly discuss some examples of privacy laws.

[Clearview AI](https://en.wikipedia.org/wiki/Clearview_AI)
- The company was founded in 2017.
- [New York Times article](https://www.nytimes.com/2020/01/18/technology/clearview-privacy-facial-recognition.html) exposes it in 2019.
- As of October 2021, they have scraped 10 billion images of faces from Facebook, Twitter, Google, YouTube, Venmo, etc.
- It sells data to law enforcement agencies (e.g., FBI) and commercial organizations.
- Company argues a First Amendment right to public information.
- Lawsuit for violation of privacy.
- Illinois's [Biometric Information Privacy Act](https://en.wikipedia.org/wiki/Biometric_Information_Privacy_Act) (2008) regulates biometric identifiers by private entities (doesn't include government entities).
  Clearview removed Illinois data.
- Deemed illegal by the EU by the Hamburg data protection authority (DPA).

[California Consumer Privacy Act (2018)](https://en.wikipedia.org/wiki/California_Consumer_Privacy_Act)
- Provide California residents with the right to:
  * Know what personal data is being collected about them.
  * Know whether their personal data is sold or disclosed and to whom.
  * Say no to the sale of personal data.
  * Access their personal data.
  * Request a business to delete any personal information about a consumer collected from that consumer.
  * Not be discriminated against for exercising their privacy rights.
- Personal data: real name, alias, postal address, unique personal identifier,
  online identifier, Internet Protocol address, email address, account name,
  social security number, driver's license number, license plate number,
  passport number, etc.
- Applies to business that operate in California and has at least $25 million in revenue.
- There is no equivalent at the federal level yet.
- Unlike GDPR, doesn't allow users to correct the data.

[California Privacy Rights Act of 2020](https://en.wikipedia.org/wiki/California_Privacy_Rights_Act)
- Creates California Privacy Protection Agency.
- Take effect Jan 1, 2023, applies to data collected after Jan 1, 2022.
- Intentions:
  * Know who is collecting their and their children's personal information, how it is being used, and to whom it is disclosed.
  * Control the use of their personal information, including limiting the use of their sensitive personal information.
  * Have access to their personal information and the ability to correct, delete, and transfer their personal information.
  * Exercise their privacy rights through easily accessible self-serve tools.
  * Exercise their privacy rights without being penalized.
  * Hold businesses accountable for failing to take reasonable information security precautions.
  * Benefit from businesses' use of their personal information.
  * Have their privacy interests protected even as employees and independent contractors.

[GDPR](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation)
- Regulation in EU law concerning data privacy.
- Adopted in 2016, enforceable in 2018.
- Broader than CCPA.
- Doesn't apply to processing of personal data for national security activities or law enforcement.
- Data subjects can provide consent to processing of personal data, and can withdraw at any time.
- People should have the right to access their own personal data.
- [Google was fined $57 million](https://www.bbc.co.uk/news/technology-46944696) because they did not obtain consent for ads personalization during Android phone setup.

## Other laws

[California's bot disclosure bill](https://www.natlawreview.com/article/california-s-bot-disclosure-law-sb-1001-now-effect):
- Illegal to use a bot to communicate with a person without disclosing that it's a bot
- Restriction: applies only to incentivize a sale or influence a vote in an election.
- Restriction: applies only to public-facing websites with 10 million monthly US visitors.

## Summary

- As we're training large language models, we have to confront copyright and fair use.
- The **uncurated** nature of web crawls means you have to appeal to fair use (it would be very difficult to get licenses from everyone).
- The **generative** aspect of models might present challenges for arguing fair use (can compete with humans).
- What level does it make sense to regulate (language models or downstream applications)?
- This space is quickly evolving and will require deep legal and AI expertise to make sensible decisions!

## Further reading

- [Foundation models report (legality section)](https://crfm.stanford.edu/assets/report.pdf#legality)
- [AI Regulation is coming](https://hbr.org/2021/09/ai-regulation-is-coming)
- [Fair Learning](https://texaslawreview.org/fair-learning/). *Mark Lemley, Bryan Casey*. Texas Law Review, 2021.
- [You might be a robot](https://www.cornelllawreview.org/wp-content/uploads/2020/07/Casey-Lemley-final-2.pdf)

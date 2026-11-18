# Design note: Velmora help-center assistant

## How I understood the need
The team doesn't trust an assistant it can't measure, so for me the evaluation was the real deliverable and the assistant came second. I assumed that a wrong answer is worse than "I don't know, please contact support", since there is a support team behind it. Every answer should point to an article. The help center is messy on purpose, and today is 1 July 2026.

## Architecture
Plain Python with no dependencies. For each question:
1. BM25 ranks the 30 articles.
2. A refusal rule decides whether to answer.
3. The answer is one or two sentences copied from the top article, with its id.
4. A last check looks for another article that gives a different number.

Articles older than two years get half their score. In practice that only affects KB-07, from 2022. `python evaluate.py` prints recall@1/3/5, the share of correct answers, the refusal counts, a threshold table and a BM25 vs TF-IDF comparison. It also writes one line per question to a JSONL file.

Two choices I hesitated over:
- **How to decide to refuse.** A raw BM25 score threshold was the obvious option, but the score depends on how long the question is. I went with "coverage" instead: the idf-weighted share of the question's words that appear in the top article. Words the help center never uses weigh the most, so questions about Lyon, resellers or carbon footprint score low. The downside is that a reworded question looks the same as a missing topic.
- **What to do when articles disagree.** When an old article disagrees with a recent one, the recent one wins, and the answer mentions the old one. When two recent articles disagree, I show both. The case I had in mind was the summer sale (14 days vs 30 days for returns). That isn't really an error, it's an exception, and hiding it would give sale customers a wrong answer.

## What I tested and what I couldn't
There are 18 unit tests. They cover recall@k and refusal counting on small hand-made examples, the keyword matching, and the assistant on a five-article toy help center. I wanted to keep eval.jsonl as a test set, so I wrote 28 questions of my own and chose the threshold on them. Then I ran eval once and didn't touch the assistant afterwards.

Result: recall@1/3/5 is 0.84/0.90/0.90. 21 of 35 answers are correct according to the keyword judge; I count about 24 when I read them myself. All 5 unanswerable questions are refused, but 6 answerable ones are refused by mistake. I couldn't test on real customer questions or other languages. Five unanswerable questions are too few to know the real refusal rate.

## What I cut
I left out an LLM for the answer: copied sentences can't invent anything, and they are easier to evaluate. I also left out embeddings, which would probably fix the reworded questions but need a model download. A synonym list would have been tuned on the eval questions, so I skipped that too.

## How I used AI
I used AI tools only to help me understand the project, clarify technical concepts, and guide me through the initial project setup. I made the implementation decisions and wrote the code myself. AI was used as a learning and support tool, not to generate the project's code or complete the development work.

## With one more day
I'd try local embeddings mixed with BM25, both for ranking and for the refusal score, and compare them on the dev set. I'd pick answer sentences based on the question type (a duration, a price, a list). I'd also ask a second person to review the answers, so I can see how often the keyword judge agrees with humans.

## Limits
- **Invented answers.** The assistant can't make up a fact, because it only copies sentences. It can still quote a true sentence that doesn't answer the question (gift card validity) or quote the wrong article (payment methods). That happened in 8 of 35 answers. Next I would check that the sentence contains the kind of thing asked for, and refuse if it doesn't.
- **Old or conflicting articles.** KB-07 contradicts KB-06 on Express delivery. The date rule catches it, but the real fix is to update or delete KB-07. My conflict check only compares numbers with the same unit, so it gives false alarms on unrelated facts and misses conflicts with no number. The two-year cut-off is arbitrary. Next I'd report conflicts to whoever owns the help center, and match on the topic as well as the unit.
- **Questions with no answer.** It refuses with a fixed message and no citation. This works for all 5 in eval, but it also refuses about one good question in six, mostly reworded ones. Next I'd use better matching to bring those back, and log every refusal so the team can see what is missing from the help center.

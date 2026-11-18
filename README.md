# Velmora help-center assistant and its evaluation

A question-answering assistant over the 30 Velmora help-center articles (`ai/kb/`), and an evaluation harness
that measures it on `ai/eval.jsonl`. **Python 3.10+ with the standard library only**: nothing to install, no key, no network.
`ai/` is unchanged.

## How to run

```bash
python evaluate.py                     # THE command: summary + per-question file + threshold sweep + BM25/TF-IDF table
python -m unittest -v                  # 18 tests (metrics on hand-made examples, assistant on a tiny hand-made KB)
python ask.py "How long is a gift card valid?"
```

Options of `evaluate.py`: `--retriever tfidf`, `--threshold 0.3`, `--no-dates`, `--max-sentences 99`,
`--questions dev/dev_questions.jsonl`, `--quick` (skip the sweep), `--out path.jsonl`.
Outputs: `results/eval_results.jsonl` (one line per question: answer, citations, top-5 ranking, confidence, verdict)
and `results/eval_results_summary.json`. There is no randomness anywhere, so there is no seed to fix: two runs give
byte-identical files (checked).

## Results (`python evaluate.py`, on the 40 questions of eval.jsonl)

| Metric | Value |
|---|---|
| recall@1 / @3 / @5 (35 answerable, before refusal) | **0.843 / 0.900 / 0.900** (MRR 0.881) |
| hit@1 / @3 / @5 (at least one source article found) | 0.857 / 0.914 / 0.914 |
| Correct answers (keyword judge) | **21/35 = 60%** (all 21 cite a source article) |
| Wrong answers / refused by mistake | 8/35 / **6/35** |
| Unanswerable questions correctly refused | **5/5** (0 answered) |
| Citation precision (cited ids that are source articles) | 85.3% |
| My manual review of the same answers | ~24/35 acceptable (see FAILURES.md) |

Recall@k for Q04 and Q08 (two source articles each) counts halves: finding 1 of the 2 gives 0.5.
That is why recall@1 is below hit@1.

**Second retrieval method and date rule** (same answer and refusal rules):

| Retriever | Date rule | recall@1 | recall@3 | recall@5 | correct | refused (unanswerable) |
|---|---|---|---|---|---|---|
| **BM25** | on | **0.843** | 0.900 | 0.900 | **21/35** | 5/5 |
| BM25 | off | 0.786 | 0.900 | 0.900 | 21/35 | 5/5 |
| TF-IDF | on | 0.771 | 0.857 | 0.900 | 20/35 | 5/5 |
| TF-IDF | off | 0.743 | 0.857 | 0.871 | 20/35 | 5/5 |

**"I don't know" threshold sweep** (eval set, for information only: the threshold was chosen on the dev set):

| threshold | correct | refused by mistake | wrong answers | unanswerable refused |
|---|---|---|---|---|
| 0.00 | 23/35 | 0/35 | 12/35 | 0/5 |
| 0.20 | 23/35 | 2/35 | 10/35 | 1/5 |
| 0.30 | 21/35 | 5/35 | 9/35 | 5/5 |
| **0.40 (chosen)** | **21/35** | **6/35** | **8/35** | **5/5** |
| 0.50 | 20/35 | 8/35 | 7/35 | 5/5 |
| 0.60 | 15/35 | 15/35 | 5/35 | 5/5 |
| 0.80 | 6/35 | 25/35 | 4/35 | 5/5 |

Reading the curve: never refusing gives 23 correct, but answers all 5 unanswerable questions. Every refusal threshold that
catches the unanswerable questions also costs 5 to 6 answerable ones, mostly paraphrases (FAILURES.md, group A).
On eval, 0.30 trades one wrong refusal for one wrong answer. I would not move the threshold based on eval anyway,
because that would be tuning on the test set.

## How I used the 40 questions (please read)

1. **Before the first eval run**, I wrote my own **dev set** (`dev/dev_questions.jsonl`: 20 answerable and 8 near-miss
   unanswerable questions, written from the articles and covering other facts than eval). I chose the threshold (0.40)
   on it, and I found one stemmer bug on it ("shipped" → "shipp"). The other constants are textbook defaults or were set before any run:
   BM25 k1=1.5/b=0.75, stale after 2 years, stale penalty 0.5, second sentence if ≥ 50% of the best.
2. Then I ran eval.jsonl **once**, with everything frozen. The numbers above are that run. **I changed nothing in the assistant
   after seeing the eval results**. The only change after it was the `--max-sentences` option, for an experiment;
   its default (2) is the original behaviour.
3. Caveats: I had *read* eval.jsonl before writing the dev set (to understand the format), so the dev set is not
   truly blind. The method is coarse and has few parameters, so the risk of over-fitting is low, but it is not zero. Also, 40 questions is a
   small sample: one question is 2.9 points of answer accuracy, and 5 unanswerable questions say little about the real refusal rate.
   A 95% interval on 21/35 is roughly 43–75%.

## How the assistant works (`velmora/`)

- `kb.py`: parses the front matter (`id`, `title`, `category`, `updated`) and splits the text into sentences.
- `retrieval.py`: tokenizer (lower case, stop words, a small suffix stemmer, keeps `7.90`, `14:00`, `10%`), BM25 and TF-IDF cosine.
- `assistant.py`, with five rules:
  1. **Rank** the articles with BM25 (title + body).
  2. **Dates**: an article last updated more than 2 years before 1 July 2026 is *stale* and its score is halved. Only KB-07 (2022) is stale;
     all the others are from 2025 or 2026.
  3. **Refuse** when the *coverage* of the top article is below 0.40. Coverage = (sum of the idf of the question terms found in the article) /
     (sum of the idf of all the question terms). Terms that appear in no article get the highest idf, so a question
     about something the help center never mentions ("reseller", "carbon footprint") gets a low coverage.
  4. **Answer** = the best sentence of the top article, plus the second best if it scores at least half as much. The sentences are copied
     word for word, so there is no generated text. The top article is cited.
  5. **Disagreement**: if one of the next two articles has a relevant sentence with a *different number for the same unit*
     (days, EUR, hours…), the answer shows it. If that article is stale, the answer says so and follows the recent article;
     the stale one is not cited. Otherwise both are cited and the customer is told to check which one applies.
     Examples: Q03 (KB-06 7.90 EUR vs stale KB-07 9.90 EUR), Q04 (KB-12 30 days vs KB-13 14 days for sale items).
- `metrics.py`: pure functions (recall@k, hit@k, MRR, keyword judge with word boundaries, refusal counting).
  `evaluate.py` only does the I/O.

## The judge

The judge passes an answer when every `answer_keywords` phrase appears in it (case-insensitive, with word boundaries, so "10 days" does not
match "110 days"). An answer to an answerable question is *correct* only if it was not refused and passes the judge. The judge's known
limits: it rewards long answers (returning the whole article raises 21 → 27/35), it cannot see wrong or misleading extra text
(Q01, Q14), and it fails short, correct answers (Q25, Q29, Q31). That is why I also reviewed the answers by hand (FAILURES.md).

## Remarks on the data (I did not edit `ai/`)

- KB-07 (2022) contradicts KB-06 (2026) on Express delivery: 9.90 vs 7.90 EUR, 48 h vs next day, cut-off 12:00 vs 14:00,
  and free standard delivery from 80 vs 60 EUR. KB-07 should be removed or updated.
- KB-13 (sale terms) is not really a contradiction of KB-12 and KB-15. It is an exception for a narrower scope (sale items, 20 June to 20 July).
  Q04 and Q08 call it "inconsistent". I would rather tell the customer "30 days, or 14 days for sale items".
- Q25, Q29 and Q31: the keywords ask for details the question does not ask for (see FAILURES.md, group E).
- Q17's keyword is a whole sentence ("do not ship to countries outside the european union"), so only a verbatim copy can pass. That is fine for an
  extractive assistant, but too strict for a model that rephrases.

## Files

`velmora/` the assistant · `evaluate.py` the harness · `ask.py` one question · `tests/` unit tests ·
`dev/dev_questions.jsonl` my dev set · `results/` outputs · `FAILURES.md` failure analysis · `DESIGN_NOTE.md` the one-page note.

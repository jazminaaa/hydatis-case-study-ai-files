# Failure analysis

Run: `python evaluate.py` (BM25, threshold 0.40, date rule on, 2-sentence answers). Every line comes from
`results/eval_results.jsonl`. The keyword judge marks **14 of 40** questions as failures: 6 refused by mistake and
8 wrong answers. I read all 40 answers myself. Below they are grouped by **cause** rather than by question.

| # | Cause | Questions | What it costs |
|---|---|---|---|
| A | Paraphrase: the question uses words the article does not use, so coverage falls below the threshold | Q05, Q10, Q15, Q22, Q27 | 5 wrong refusals |
| B | An unknown entity in a question the help center does answer | Q17 | 1 wrong refusal |
| C | Right article, wrong sentence picked | Q07, Q16, Q20 | 3 wrong answers |
| D | Wrong article ranked first | Q02, Q08 | 2 wrong answers |
| E | The judge is too strict: the answer is acceptable | Q25, Q29, Q31 | 3 false "wrong" |
| F | The disagreement detector raises a false alarm (the judge does not see it) | Q01, Q14 (and Q08) | misleading notes in "correct" answers |

## A. Paraphrase → refused by mistake (the biggest problem)
The refusal rule computes the share of the question's idf weight found in the top article. Words that appear in no
article get the *highest* idf. That is exactly what makes unanswerable questions fail the rule ("Lyon", "reseller",
"carbon footprint"), but it also punishes plain paraphrases:

- **Q27** "I can't remember how to get into my profile anymore." → refused (coverage 0.27). KB-20 says *forgot password*
  and *login*; the question says *remember*, *get into* and *profile*. "profile" appears only in KB-21 (delete account), which ranks first.
- **Q15** "Can I send back something I bought by mistake?" → refused (0.15). KB-12 says *return*, never *send back*.
  Top article: KB-29.
- **Q05** "My package has been stuck on the same status for a week." → refused (0.12). KB-09 says *parcel*,
  *tracking has not moved*, *7 days*. Lexical search cannot know that a *week* is *7 days*. Top article: KB-18 (because of "status").
- **Q22** "I received a broken item. How fast must I report it?" → the right article (KB-16) ranks first, but the coverage is 0.29:
  *broken* ≠ *damaged*.
- **Q10** "How fast is Express delivery and what is the order cut-off time?" → KB-06 ranks first, coverage 0.24
  (*fast*, *cut-off*, *time* are not in KB-06).

Why: lexical matching has no synonyms. Lowering the threshold does not fix this. At 0.20 (see the sweep), Q10 and Q22
come back, but 4 of the 5 unanswerable questions are then answered. Next step: local embeddings
(e.g. a small sentence-transformer) for ranking *and* for the refusal score, or a small synonym list written
from the help-center vocabulary and validated on a dev set, not on eval.

## B. Unknown entity in an answerable question
- **Q17** "Can you deliver to Switzerland?" → refused (coverage 0.35, the right article KB-11 ranks first). The answer
  is implied by a general rule ("we do not ship outside the EU"), but *Switzerland* is unknown to the KB, so it has the highest idf.
  This is the same mechanism that correctly refuses Q06 ("physical shop in **Lyon**"). The rule cannot tell
  "unknown thing the KB says nothing about" from "unknown thing covered by a general rule". I prefer this error
  (a safe refusal) to the opposite one. A model that knows Switzerland is outside the EU would be needed.

## C. Right article, wrong sentence
Sentence scoring adds up the idf of the question words found in each sentence, so a sentence that repeats the
*topic* words beats the sentence that holds the *requested detail*:
- **Q07** "How long is a gift card valid?" → returned "Gift cards come in three amounts…" and "…cannot be used to buy another
  gift card". The sentence "valid for 24 months" only matches *valid*.
- **Q16** "Which countries do you deliver to outside France?" → returned "We do not ship to countries outside the European
  Union." (it matches *countries* and *outside*) instead of the list of countries.
- **Q20** "How do I start a return?" → first step only ("Open My orders… click Return…"), without "Print the label". Partly correct.

Next step: score sentences with the question type (how long → durations, how much → prices, which → lists), or simply
return the whole article. Returning the whole article raises the judge score from 21/35 to 27/35 (`--max-sentences 99`),
but that measures the judge's leniency more than a better answer.

## D. Wrong article ranked first
- **Q02** "Which payment methods can I use?" → KB-15 (Refunds: "…refund to your original **payment method**…") beats KB-02,
  where "payment methods" appears only in the title (the body says *pay with*).
- **Q08** "How long do refunds take?" → KB-09 (Parcel lost: "investigation… **takes** up to 10 working days. Then we… **refund**
  you") beats KB-15 (Refunds), which ranks 3rd. The disagreement note then quotes KB-15 ("5 business days"), but the main answer is wrong,
  and KB-13 ("10 business days for sale items"), which the expected answer wants, is not found.

Next step: a title/field boost and fewer question-form words ("take", "long") in the index. Both should be tuned on a
dev set. I did not change them after seeing these failures (see "How I used the 40 questions" in the README).

## E. The judge is too strict (my manual review)
- **Q25** "Until when can I cancel an order?" → "…while its status is Received or Preparing." This is correct, but the keyword is
  *packed* (the next sentence).
- **Q29** "How do I delete my account?" → "Go to Settings, then Privacy, then Delete my account." This is correct. The keyword *30 days*
  answers a question the user did not ask.
- **Q31** "How do I unsubscribe from the newsletter?" → the link or Settings → Communication. This is correct. The keyword *48 hours* is extra.

## F. Disagreement detector: false alarms
The detector says "two articles disagree" when the best sentences of two retrieved articles give **different numbers with
the same unit**. It does not check that the numbers are about the *same thing*:
- **Q01**: it shows KB-11 ("Delivery takes 3 to 7 working days") as a different figure. KB-11 is about shipping *outside France*, not a contradiction.
- **Q14**: it shows KB-06 ("Pickup point delivery takes 2 to 4 working days") next to "held there for 10 days". These are different facts.

These answers pass the judge, but a customer could be confused. The true positives are Q04 (KB-12 30 days vs KB-13 14 days), Q03 and
Q18 (stale KB-07 vs KB-06).

## Manual review vs keyword judge (all 29 answered, answerable questions)
| | judge: correct (21) | judge: wrong (8) |
|---|---|---|
| me: acceptable | 19 (+ Q01, Q14 with a misleading note) | 3 (Q25, Q29, Q31) + Q20 partly |
| me: wrong | 0 | 4 (Q02, Q07, Q08, Q16) |

My estimate is therefore **24/35 acceptable answers** where the judge says 21/35. Q32 (57 cm head) passes the judge
because "size M" appears in the size table, but the assistant never says "choose M". A human has to read the table, so this one is borderline.
One reviewer, 29 answers: this is a sanity check, not a measurement.

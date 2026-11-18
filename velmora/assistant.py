"""The assistant: retrieve, decide whether to answer, extract an answer, cite, flag disagreements.

Rules (all explained in DESIGN_NOTE.md):
1. Rank articles with BM25 (or TF-IDF).
2. Stale rule: an article not updated for more than STALE_AFTER_DAYS before TODAY is demoted
   (its score is multiplied by STALE_PENALTY). If it disagrees with the article we answer from,
   we say so, but we follow the more recent article.
3. Refusal rule: "coverage" = share of the question's idf weight whose terms appear in the top
   article. Words that appear nowhere in the help center have the highest idf, so a question about
   something the help center never mentions gets a low coverage. Below the threshold -> "I don't know".
4. Answer = the best sentence of the top article (+ the second best if it is at least half as good),
   copied verbatim, in article order. Never generated text.
5. Disagreement rule: if another article in the top 3 has a sentence relevant to the question that
   gives a different number for the same unit (days, EUR, hours...), we show it next to the answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .kb import Article
from .retrieval import BM25, RETRIEVERS, tokenize

TODAY = date(2026, 7, 1)          # the brief asks us to pretend today is 1 July 2026
STALE_AFTER_DAYS = 2 * 365        # chosen a priori (policy/price pages older than 2 years are suspect)
STALE_PENALTY = 0.5
DEFAULT_THRESHOLD = 0.40         # chosen on dev/dev_questions.jsonl, NOT on eval.jsonl (see README)
SECOND_SENTENCE_RATIO = 0.5
IDK = "I don't know: the help center does not answer this question. Please contact customer support."

QUANTITY_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(eur|%|working days?|business days?|days?|hours?|months?|years?|minutes?|times|miles?|cm)\b",
    re.I,
)


def quantities(text: str) -> set[tuple[str, str]]:
    """(value, unit) pairs, e.g. 'within 5 business days' -> {('5', 'day')}."""
    out = set()
    for value, unit in QUANTITY_RE.findall(text):
        unit = unit.lower().split()[-1].rstrip("s") or unit.lower()
        out.add((value.replace(",", "."), unit))
    return out


def disagree(a: str, b: str) -> bool:
    """True if both texts give a quantity with the same unit but no common value for that unit."""
    qa, qb = quantities(a), quantities(b)
    for unit in {u for _, u in qa} & {u for _, u in qb}:
        if {v for v, u in qa if u == unit}.isdisjoint({v for v, u in qb if u == unit}):
            return True
    return False


@dataclass
class Answer:
    question: str
    answered: bool
    text: str
    citations: list[str]
    ranking: list[str]                 # article ids, best first (after the stale rule)
    confidence: float                  # coverage of the top article
    notes: list[str] = field(default_factory=list)


class Assistant:
    def __init__(self, articles: list[Article], retriever: str = "bm25",
                 threshold: float = DEFAULT_THRESHOLD, use_dates: bool = True, max_sentences: int = 2):
        self.articles = articles
        self.retriever = RETRIEVERS[retriever](articles)
        # idf weights for coverage and sentence scoring always come from BM25 statistics, so the
        # refusal rule is the same whichever retriever ranks the articles.
        self.stats = self.retriever if isinstance(self.retriever, BM25) else BM25(articles)
        self.threshold = threshold
        self.use_dates = use_dates
        self.max_sentences = max_sentences  # 2 by default; a large value returns the whole article

    # ---------- helpers ----------
    def is_stale(self, article: Article) -> bool:
        return (TODAY - article.updated).days > STALE_AFTER_DAYS

    def rank(self, question: str) -> list[tuple[Article, float]]:
        ranked = self.retriever.rank(question)
        if self.use_dates:
            ranked = [(a, s * STALE_PENALTY if self.is_stale(a) else s) for a, s in ranked]
            ranked.sort(key=lambda pair: -pair[1])
        return ranked

    def coverage(self, query_terms: set[str], article: Article) -> float:
        if not query_terms:
            return 0.0
        doc_terms = set(tokenize(f"{article.title}. {article.body}"))
        total = sum(self.stats.idf(t) for t in query_terms)
        return sum(self.stats.idf(t) for t in query_terms & doc_terms) / total

    def sentence_score(self, query_terms: set[str], sentence: str) -> float:
        return sum(self.stats.idf(t) for t in query_terms & set(tokenize(sentence)))

    def best_sentences(self, query_terms: set[str], article: Article) -> list[tuple[int, str, float]]:
        scored = [(i, s, self.sentence_score(query_terms, s)) for i, s in enumerate(article.sentences)]
        return sorted(scored, key=lambda x: (-x[2], x[0]))

    # ---------- main entry point ----------
    def answer(self, question: str) -> Answer:
        query_terms = set(tokenize(question))
        ranked = self.rank(question)
        ranking = [a.id for a, _ in ranked]
        top, top_score = ranked[0]
        conf = self.coverage(query_terms, top) if top_score > 0 else 0.0

        if conf < self.threshold:
            return Answer(question, False, IDK, [], ranking, conf)

        best = self.best_sentences(query_terms, top)
        chosen = [best[0]] + [b for b in best[1:self.max_sentences]
                              if b[2] > 0 and b[2] >= SECOND_SENTENCE_RATIO * best[0][2]]
        if self.max_sentences > 2:  # experiment mode: whole article, to show how lenient the judge is
            chosen = best[:self.max_sentences]
        chosen.sort(key=lambda x: x[0])  # keep the article's own order
        text = " ".join(s for _, s, _ in chosen)
        citations, notes = [top.id], []

        # Disagreement check against the next two articles.
        for other, score in ranked[1:3]:
            if score <= 0:
                continue
            o_idx, o_sent, o_score = self.best_sentences(query_terms, other)[0]
            if o_score < 0.5 * best[0][2] or not disagree(text, o_sent):
                continue
            if self.use_dates and self.is_stale(other) and not self.is_stale(top):
                notes.append(f"Older article {other.id} (last updated {other.updated}) says: \"{o_sent}\" "
                             f"We follow the more recent {top.id} (updated {top.updated}).")
            else:
                notes.append(f"Note: {other.id} ({other.title}, updated {other.updated}) gives a different "
                             f"figure: \"{o_sent}\" Check which one applies to your case.")
                citations.append(other.id)

        full_text = text + ("\n" + "\n".join(notes) if notes else "")
        return Answer(question, True, full_text, citations, ranking, conf, notes)

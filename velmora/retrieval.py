"""Lexical retrieval: a tokenizer, BM25 and TF-IDF (cosine). No external dependency."""
from __future__ import annotations

import math
import re
from collections import Counter

from .kb import Article

STOPWORDS = set("""
a an the and or but if of to in on at by for with from into about as is are was were be been being
i me my we our you your it its this that these those there here he she they them his her their
do does did done can could will would should shall may might must have has had having
how what when where which who whom why whether not no yes so than then too very just also only
get got am any some all more most up out over after before again t s don can't im i'm
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.:][0-9]+)?%?")


def stem(word: str) -> str:
    """Very small suffix stripper: enough to match returns/returned/return, days/day, items/item."""
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            base = word[: -len(suffix)]
            # Porter-style undoubling: shipped -> shipp -> ship, stopped -> stop.
            if base[-1] == base[-2] and base[-1] not in "aeioulsz":
                base = base[:-1]
            return base
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower().replace("’", "'"))
    return [stem(t) for t in tokens if t not in STOPWORDS]


def doc_text(article: Article) -> str:
    return f"{article.title}. {article.body}"


class BM25:
    """Okapi BM25 with the textbook defaults k1=1.5, b=0.75 (not tuned)."""

    name = "bm25"

    def __init__(self, articles: list[Article], k1: float = 1.5, b: float = 0.75):
        self.articles = articles
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(doc_text(a)) for a in articles]
        self.tf = [Counter(toks) for toks in self.doc_tokens]
        self.avg_len = sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens)
        self.df = Counter(term for toks in self.doc_tokens for term in set(toks))
        self.n = len(articles)

    def idf(self, term: str) -> float:
        # Always > 0. A term that appears in no article gets the highest idf, which matters for the
        # coverage score: an unknown important word should count against answering.
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], i: int) -> float:
        tf, length = self.tf[i], len(self.doc_tokens[i])
        total = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if f:
                total += self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
        return total

    def rank(self, query: str) -> list[tuple[Article, float]]:
        q = tokenize(query)
        scored = [(a, self.score(q, i)) for i, a in enumerate(self.articles)]
        # Stable sort: ties keep KB order, so results are deterministic.
        return sorted(scored, key=lambda pair: -pair[1])


class TfIdf(BM25):
    """TF-IDF with log-scaled term frequency and cosine similarity. Shares tokenizer and idf with BM25."""

    name = "tfidf"

    def __init__(self, articles: list[Article]):
        super().__init__(articles)
        self.vectors = [self._vector(tf) for tf in self.tf]

    def _vector(self, tf: Counter) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self.idf(t) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        q = self._vector(Counter(query_tokens))
        doc = self.vectors[i]
        return sum(w * doc.get(t, 0.0) for t, w in q.items())


RETRIEVERS = {"bm25": BM25, "tfidf": TfIdf}

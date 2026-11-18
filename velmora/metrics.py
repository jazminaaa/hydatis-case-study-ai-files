"""Pure evaluation functions (no I/O), so they can be unit-tested on tiny hand-made examples."""
from __future__ import annotations

import re


def normalize(text: str) -> str:
    """Lower case, unify apostrophes / non-breaking spaces / euro sign, collapse whitespace."""
    text = text.lower().replace(" ", " ").replace("’", "'").replace("€", " eur")
    return " ".join(text.split())


def keywords_match(answer: str, keywords: list[str]) -> bool:
    """The provided cheap judge: every keyword phrase must appear in the answer.
    Matching is on word boundaries, so '10 days' does not match inside '110 days'."""
    norm = normalize(answer)
    for kw in keywords:
        pattern = r"(?<![\w.])" + re.escape(normalize(kw)) + r"(?![\w])"
        if not re.search(pattern, norm):
            return False
    return True


def recall_at_k(ranking: list[str], relevant: list[str], k: int) -> float:
    """Share of the relevant articles found in the top k. Undefined (error) when nothing is relevant."""
    if not relevant:
        raise ValueError("recall@k is undefined for a question without relevant articles")
    return len(set(ranking[:k]) & set(relevant)) / len(set(relevant))


def hit_at_k(ranking: list[str], relevant: list[str], k: int) -> float:
    """1.0 if at least one relevant article is in the top k."""
    return 1.0 if set(ranking[:k]) & set(relevant) else 0.0


def reciprocal_rank(ranking: list[str], relevant: list[str]) -> float:
    for i, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def judge(row: dict, answered: bool, answer_text: str, citations: list[str]) -> dict:
    """Per-question verdict. `row` is one line of eval.jsonl."""
    if not row["answerable"]:
        return {"outcome": "correct_refusal" if not answered else "answered_unanswerable",
                "answer_correct": None, "citation_ok": None}
    if not answered:
        return {"outcome": "wrong_refusal", "answer_correct": False, "citation_ok": False}
    correct = keywords_match(answer_text, row["answer_keywords"])
    cited_ok = bool(set(citations) & set(row["source_articles"]))
    return {"outcome": "correct" if correct else "wrong_answer", "answer_correct": correct, "citation_ok": cited_ok}


def summarize(results: list[dict], ks=(1, 3, 5)) -> dict:
    """Aggregate per-question results (each has: answerable, ranking, source_articles, outcome,
    answer_correct, citation_ok, citations)."""
    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]
    s: dict = {"n_questions": len(results), "n_answerable": len(ans), "n_unanswerable": len(unans)}
    for k in ks:
        s[f"recall@{k}"] = _mean([recall_at_k(r["ranking"], r["source_articles"], k) for r in ans])
        s[f"hit@{k}"] = _mean([hit_at_k(r["ranking"], r["source_articles"], k) for r in ans])
    s["mrr"] = _mean([reciprocal_rank(r["ranking"], r["source_articles"]) for r in ans])
    s["answer_correct"] = sum(1 for r in ans if r["answer_correct"])
    s["answer_correct_rate"] = _ratio(s["answer_correct"], len(ans))
    s["correct_and_cited"] = sum(1 for r in ans if r["answer_correct"] and r["citation_ok"])
    s["wrong_refusals"] = sum(1 for r in ans if r["outcome"] == "wrong_refusal")
    s["wrong_answers"] = sum(1 for r in ans if r["outcome"] == "wrong_answer")
    s["correct_refusals"] = sum(1 for r in unans if r["outcome"] == "correct_refusal")
    s["answered_unanswerable"] = sum(1 for r in unans if r["outcome"] == "answered_unanswerable")
    answered = [r for r in ans if r["outcome"] != "wrong_refusal"]
    cited = [c for r in answered for c in r["citations"]]
    s["citation_precision"] = _ratio(sum(1 for r in answered for c in r["citations"] if c in r["source_articles"]), len(cited))
    return s


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def _ratio(a: int, b: int) -> float:
    return round(a / b, 3) if b else 0.0

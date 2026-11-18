"""Evaluation harness. One command:  python evaluate.py

Runs the assistant on every question of ai/eval.jsonl, prints a summary, writes per-question
results to results/eval_results.jsonl, and prints a threshold sweep plus a BM25 / TF-IDF comparison.
Deterministic: no randomness anywhere (no seed to fix), standard library only, no network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from velmora.assistant import DEFAULT_THRESHOLD, Assistant
from velmora.kb import load_kb
from velmora.metrics import judge, summarize

ROOT = Path(__file__).parent
SWEEP = [0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]


def load_questions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        missing = {"id", "question", "answerable", "answer_keywords", "source_articles"} - r.keys()
        if missing:
            raise ValueError(f"{path}: question {r.get('id')} is missing {sorted(missing)}")
    return rows


def run(assistant: Assistant, questions: list[dict]) -> list[dict]:
    results = []
    for row in questions:
        a = assistant.answer(row["question"])
        verdict = judge(row, a.answered, a.text, a.citations)
        results.append({
            "id": row["id"], "question": row["question"], "answerable": row["answerable"],
            "answered": a.answered, "answer": a.text, "citations": a.citations,
            "confidence": round(a.confidence, 3), "ranking": a.ranking[:5],
            "source_articles": row["source_articles"], "answer_keywords": row["answer_keywords"],
            "expected_answer": row.get("expected_answer"), "disagreement_flagged": bool(a.notes),
            **verdict,
        })
    return results


def print_summary(s: dict, title: str) -> None:
    na, nu = s["n_answerable"], s["n_unanswerable"]
    print(f"\n=== {title} ===")
    print(f"questions: {s['n_questions']} ({na} answerable, {nu} unanswerable)")
    print("retrieval on answerable questions (ranking before any refusal):")
    print(f"  recall@1 {s['recall@1']:.3f}   recall@3 {s['recall@3']:.3f}   recall@5 {s['recall@5']:.3f}   MRR {s['mrr']:.3f}")
    print(f"  hit@1    {s['hit@1']:.3f}   hit@3    {s['hit@3']:.3f}   hit@5    {s['hit@5']:.3f}")
    print("answers on answerable questions (keyword judge):")
    print(f"  correct            {s['answer_correct']}/{na} = {s['answer_correct_rate']:.1%}")
    print(f"  correct and cited  {s['correct_and_cited']}/{na}")
    print(f"  wrong answer       {s['wrong_answers']}/{na}")
    print(f"  refused by mistake {s['wrong_refusals']}/{na}")
    print(f"  citation precision {s['citation_precision']:.1%}")
    print("unanswerable questions:")
    print(f"  correctly refused  {s['correct_refusals']}/{nu}")
    print(f"  answered anyway    {s['answered_unanswerable']}/{nu}")


def sweep(articles, questions, retriever: str, use_dates: bool) -> list[dict]:
    rows = []
    for t in SWEEP:
        s = summarize(run(Assistant(articles, retriever, t, use_dates), questions))
        rows.append({"threshold": t, **s})
    return rows


def print_sweep(rows: list[dict], title: str) -> None:
    print(f"\n--- 'I don't know' threshold sweep: {title} ---")
    print("threshold | correct answers | refused by mistake | wrong answers | unanswerable refused")
    for r in rows:
        mark = "  <- default" if abs(r["threshold"] - DEFAULT_THRESHOLD) < 1e-9 else ""
        print(f"   {r['threshold']:.2f}   |      {r['answer_correct']:2d}/{r['n_answerable']}      |"
              f"       {r['wrong_refusals']:2d}/{r['n_answerable']}        |    {r['wrong_answers']:2d}/{r['n_answerable']}      |"
              f"        {r['correct_refusals']}/{r['n_unanswerable']}{mark}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", default=str(ROOT / "ai" / "eval.jsonl"))
    p.add_argument("--kb", default=str(ROOT / "ai" / "kb"))
    p.add_argument("--retriever", choices=["bm25", "tfidf"], default="bm25")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--max-sentences", type=int, default=2,
                   help="answer length; e.g. 99 returns the whole article (shows the keyword judge is gameable)")
    p.add_argument("--no-dates", action="store_true", help="disable the stale-article rule")
    p.add_argument("--out", default=str(ROOT / "results" / "eval_results.jsonl"))
    p.add_argument("--quick", action="store_true", help="skip the sweep and the retriever comparison")
    args = p.parse_args(argv)

    articles = load_kb(args.kb)
    questions = load_questions(Path(args.questions))
    use_dates = not args.no_dates

    results = run(Assistant(articles, args.retriever, args.threshold, use_dates, args.max_sentences), questions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(results)
    print_summary(summary, f"{Path(args.questions).name} | retriever={args.retriever} | "
                           f"threshold={args.threshold} | dates={'on' if use_dates else 'off'} | max_sentences={args.max_sentences}")
    print(f"\nper-question results: {out}")
    failures = [r for r in results if r["outcome"] not in ("correct", "correct_refusal")]
    if failures:
        print("failures:", ", ".join(f"{r['id']}({r['outcome']})" for r in failures))
    (out.with_name(out.stem + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.quick:
        print_sweep(sweep(articles, questions, args.retriever, use_dates), f"{args.retriever}, for information only")
        print("\n--- retriever comparison (same answer and refusal rules) ---")
        print("retriever | dates | recall@1 | recall@3 | recall@5 | MRR   | correct | unanswerable refused")
        for name in ("bm25", "tfidf"):
            for dates in (True, False):
                s = summarize(run(Assistant(articles, name, args.threshold, dates), questions))
                print(f"{name:9} | {'on ' if dates else 'off'}   |  {s['recall@1']:.3f}   |  {s['recall@3']:.3f}   |"
                      f"  {s['recall@5']:.3f}   | {s['mrr']:.3f} |  {s['answer_correct']:2d}/{s['n_answerable']}  |"
                      f"        {s['correct_refusals']}/{s['n_unanswerable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

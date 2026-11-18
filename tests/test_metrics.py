"""Tests of the evaluation code on tiny hand-made examples. Run: python -m unittest -v"""
import unittest

from velmora.metrics import hit_at_k, judge, keywords_match, recall_at_k, reciprocal_rank, summarize


class RecallAtK(unittest.TestCase):
    def test_single_relevant(self):
        ranking = ["KB-03", "KB-06", "KB-01"]
        self.assertEqual(recall_at_k(ranking, ["KB-06"], 1), 0.0)
        self.assertEqual(recall_at_k(ranking, ["KB-06"], 2), 1.0)

    def test_two_relevant_counts_fractions(self):
        ranking = ["KB-12", "KB-01", "KB-02", "KB-13"]
        self.assertEqual(recall_at_k(ranking, ["KB-12", "KB-13"], 1), 0.5)
        self.assertEqual(recall_at_k(ranking, ["KB-12", "KB-13"], 3), 0.5)
        self.assertEqual(recall_at_k(ranking, ["KB-12", "KB-13"], 5), 1.0)
        self.assertEqual(hit_at_k(ranking, ["KB-12", "KB-13"], 1), 1.0)

    def test_unanswerable_question_is_rejected(self):
        with self.assertRaises(ValueError):
            recall_at_k(["KB-01"], [], 1)

    def test_mrr(self):
        self.assertEqual(reciprocal_rank(["a", "b", "c"], ["c"]), 1 / 3)
        self.assertEqual(reciprocal_rank(["a"], ["z"]), 0.0)


class KeywordJudge(unittest.TestCase):
    def test_all_keywords_required_case_insensitive(self):
        self.assertTrue(keywords_match("Pay with PayPal or SEPA transfer.", ["paypal", "sepa"]))
        self.assertFalse(keywords_match("Pay with PayPal.", ["paypal", "sepa"]))

    def test_word_boundaries(self):
        self.assertFalse(keywords_match("held for 110 days", ["10 days"]))
        self.assertFalse(keywords_match("costs 17.90 EUR", ["7.90"]))
        self.assertTrue(keywords_match("costs 7.90 EUR.", ["7.90"]))
        self.assertTrue(keywords_match("Size M fits 55 to 58 cm", ["size m"]))

    def test_normalisation(self):
        self.assertTrue(keywords_match("free from 60 €", ["60 eur"]))
        self.assertTrue(keywords_match("a   reset\nlink", ["reset link"]))


class RefusalCounting(unittest.TestCase):
    def _row(self, answerable, sources=("KB-01",), keywords=("ok",)):
        return {"answerable": answerable, "source_articles": list(sources) if answerable else [],
                "answer_keywords": list(keywords) if answerable else None}

    def test_judge_outcomes(self):
        self.assertEqual(judge(self._row(False), False, "I don't know", [])["outcome"], "correct_refusal")
        self.assertEqual(judge(self._row(False), True, "ok", ["KB-01"])["outcome"], "answered_unanswerable")
        self.assertEqual(judge(self._row(True), False, "I don't know", [])["outcome"], "wrong_refusal")
        self.assertEqual(judge(self._row(True), True, "ok", ["KB-01"])["outcome"], "correct")
        self.assertEqual(judge(self._row(True), True, "nope", ["KB-01"])["outcome"], "wrong_answer")

    def test_summary_counts(self):
        def result(answerable, answered, text, ranking):
            row = self._row(answerable)
            cites = ranking[:1] if answered else []
            return {**row, "ranking": ranking, "citations": cites, **judge(row, answered, text, cites)}

        results = [
            result(True, True, "ok", ["KB-01", "KB-02"]),       # correct, rank 1
            result(True, False, "", ["KB-02", "KB-01"]),        # refused by mistake, rank 2
            result(True, True, "wrong", ["KB-03", "KB-04"]),    # wrong answer, not retrieved in top 2
            result(False, False, "", ["KB-05"]),                # correct refusal
            result(False, True, "x", ["KB-05"]),                # answered an unanswerable question
        ]
        s = summarize(results)
        self.assertEqual((s["n_answerable"], s["n_unanswerable"]), (3, 2))
        self.assertEqual(s["answer_correct"], 1)
        self.assertEqual(s["wrong_refusals"], 1)
        self.assertEqual(s["wrong_answers"], 1)
        self.assertEqual(s["correct_refusals"], 1)
        self.assertEqual(s["answered_unanswerable"], 1)
        self.assertAlmostEqual(s["recall@1"], round(1 / 3, 3))
        self.assertAlmostEqual(s["recall@3"], round(2 / 3, 3))
        # unanswerable questions must not enter recall
        self.assertEqual(s["n_answerable"], 3)


if __name__ == "__main__":
    unittest.main()

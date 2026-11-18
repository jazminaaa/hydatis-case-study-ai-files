"""Behaviour tests of the assistant on a tiny hand-made help center (not the real KB, not eval.jsonl)."""
import unittest
from datetime import date
from pathlib import Path

from velmora.assistant import IDK, Assistant, disagree, quantities
from velmora.kb import Article, load_kb, parse_article
from velmora.retrieval import stem, tokenize


def art(id_, title, body, updated=date(2026, 3, 1)):
    return Article(id_, title, "Test", updated, body)


TINY_KB = [
    art("T-1", "Gift cards", "Gift cards are valid for 24 months. They cannot be exchanged for cash."),
    art("T-2", "Delivery prices", "Express delivery costs 7.90 EUR. Standard delivery costs 4.90 EUR."),
    art("T-3", "Express guide", "Express delivery costs 9.90 EUR.", updated=date(2021, 1, 1)),
    art("T-4", "Returns", "You can return an item within 30 days of delivery."),
    art("T-5", "Sale terms", "Sale items can be returned within 14 days of delivery."),
]


class AssistantBehaviour(unittest.TestCase):
    def setUp(self):
        self.a = Assistant(TINY_KB)

    def test_answers_and_cites(self):
        ans = self.a.answer("How long is a gift card valid?")
        self.assertTrue(ans.answered)
        self.assertIn("24 months", ans.text)
        self.assertEqual(ans.citations, ["T-1"])

    def test_refuses_with_no_citation(self):
        ans = self.a.answer("Do you have a physical shop in Lyon?")
        self.assertFalse(ans.answered)
        self.assertEqual((ans.text, ans.citations), (IDK, []))

    def test_stale_article_is_demoted_and_mentioned(self):
        ans = self.a.answer("How much does Express delivery cost?")
        self.assertEqual(ans.citations, ["T-2"])          # the recent article wins
        self.assertIn("7.90", ans.text)
        self.assertIn("Older article T-3", ans.text)      # the old one is disclosed, not cited

    def test_disagreement_between_recent_articles_shows_both(self):
        ans = self.a.answer("How many days do I have to return an item?")
        self.assertEqual(set(ans.citations), {"T-4", "T-5"})
        self.assertIn("30 days", ans.text)
        self.assertIn("14 days", ans.text)

    def test_empty_question_is_refused(self):
        self.assertFalse(self.a.answer("   ").answered)
        self.assertFalse(self.a.answer("how do I?").answered)


class TextHelpers(unittest.TestCase):
    def test_quantities_and_disagree(self):
        self.assertEqual(quantities("within 5 business days"), {("5", "day")})
        self.assertTrue(disagree("within 30 days", "within 14 days"))
        self.assertFalse(disagree("within 30 days", "costs 30 EUR"))
        self.assertFalse(disagree("held 10 days", "within 10 days"))

    def test_stemming(self):
        self.assertEqual(stem("shipped"), "ship")
        self.assertEqual(stem("returns"), "return")
        self.assertEqual(tokenize("Can't I return 2 items for 7.90?"), ["return", "2", "item", "7.90"])

    def test_parse_article_header(self):
        a = parse_article("---\nid: KB-99\ntitle: T\ncategory: C\nupdated: 2025-01-02\n---\n\n# T\n\nHello. World.\n")
        self.assertEqual((a.id, a.updated, a.sentences), ("KB-99", date(2025, 1, 2), ["Hello.", "World."]))

    def test_real_kb_loads(self):
        kb = load_kb(Path(__file__).resolve().parents[1] / "ai" / "kb")
        self.assertEqual(len(kb), 30)
        self.assertEqual(len({a.id for a in kb}), 30)


if __name__ == "__main__":
    unittest.main()

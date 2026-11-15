from __future__ import annotations

import unittest

from drawingqa.intents import COUNT_TASKS, resolve_intent
from drawingqa.prompts import build_prompt
from drawingqa.schemas import Question


class IntentTests(unittest.TestCase):
    def test_all_twelve_bilingual_pairs_share_canonical_inventory_prompt(self):
        self.assertEqual(len(COUNT_TASKS), 12)
        for category, (english, finnish, _rule, _nouns) in COUNT_TASKS.items():
            with self.subTest(category=category):
                left = resolve_intent(Question(path="test.png", question=english, answer=999))
                right = resolve_intent(Question(path="test.png", question=finnish, answer=999))
                self.assertEqual(left, right)
                self.assertEqual(left.category, category)
                self.assertEqual(build_prompt(left, "inventory", english), build_prompt(right, "inventory", finnish))

    def test_qualified_counts_preserve_question(self):
        for text in ["How many windows are on the second floor?", "How many doors are wider than 900 mm?", "How many bedrooms have balconies?"]:
            intent = resolve_intent(Question(path="test.png", question=text))
            self.assertEqual(intent.operation, "general")
            self.assertEqual(intent.canonical_question, text)

    def test_fixture_room_categories_are_distinct(self):
        expected = {"How many toilets?": "toilets", "How many separate WC rooms?": "wc_rooms", "How many bathrooms?": "bathrooms"}
        for text, category in expected.items():
            self.assertEqual(resolve_intent(Question(path="test.png", question=text)).category, category)

    def test_ground_truth_is_not_in_prompt(self):
        q = Question(path="test.png", question="How many windows?", answer="SECRET GROUND TRUTH")
        prompt = "\n".join(build_prompt(resolve_intent(q), "inventory", q.question))
        self.assertNotIn("SECRET GROUND TRUTH", prompt)
        self.assertIn("normalized coordinates", prompt)

    def test_unknown_general_question_is_not_rewritten(self):
        q = Question(path="test.png", question="What is the printed ceiling height?", answer=3.5)
        intent = resolve_intent(q)
        self.assertEqual(intent.operation, "general")
        self.assertIn(q.question, build_prompt(intent, "inventory", q.question)[1])


if __name__ == "__main__":
    unittest.main()

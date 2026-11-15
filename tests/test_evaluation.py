import json
import tempfile
import unittest
from pathlib import Path

from drawingqa.evaluation import _csv_cell, compare_runs, exact_match, summarize
from drawingqa.schemas import Prediction, Usage


def prediction(**overrides):
    data = dict(question_id="q1", path="data/cubicasa5k/plan.png", question="How many windows?", model="qwen", model_id="model", status="answered", answer=2, unit="count", category="windows", expected_answer=2, has_ground_truth=True, source_sha256="abc", config_sha256="config")
    data.update(overrides)
    return Prediction(**data)


class EvaluationTests(unittest.TestCase):
    def test_csv_formula_injection_is_escaped(self):
        self.assertEqual(_csv_cell("=HYPERLINK(\"bad\")"), "\'=HYPERLINK(\"bad\")")
        self.assertEqual(_csv_cell(-2), -2)

    def test_boolean_does_not_match_numeric_count(self):
        self.assertFalse(exact_match(True, 1))
        self.assertFalse(exact_match(0, False))
        self.assertTrue(exact_match(True, True))
        self.assertTrue(exact_match(2, 2.0))
        self.assertFalse(exact_match("2", 2))

    def test_abstentions_are_in_denominator_not_mae(self):
        rows = [prediction(), prediction(question_id="q2", status="needs_review", answer=None, proposed_answer=2), prediction(question_id="q3", status="error", answer=None)]
        result = summarize(rows)
        self.assertAlmostEqual(result["exact_accuracy_all_labeled"], 1 / 3)
        self.assertEqual(result["exact_accuracy_answered_only"], 1)
        self.assertEqual(result["numeric_mae_samples"], 1)
        self.assertAlmostEqual(result["answer_coverage"], 1 / 3)

    def test_unlabeled_no_invented_accuracy(self):
        result = summarize([prediction(has_ground_truth=False, expected_answer=None)])
        self.assertIsNone(result["exact_accuracy_all_labeled"])
        self.assertIsNone(result["numeric_mae_answered_only"])
        self.assertEqual(result["answer_coverage"], 1)

    def test_model_usage_separates_costs_tokens_and_api_latency(self):
        rows = [prediction(model="a", usage=Usage(requests=1, input_tokens=100, output_tokens=10, latency_seconds=4, estimated_cost_usd=0.1)), prediction(model="b", question_id="b", usage=Usage(cached_requests=1, latency_seconds=0, estimated_cost_usd=0))]
        result = summarize(rows)
        self.assertEqual(result["by_model"]["a"]["usage"]["estimated_cost_usd"], 0.1)
        self.assertEqual(result["by_model"]["b"]["usage"]["cached_requests"], 1)
        self.assertEqual(result["by_model"]["a"]["usage"]["reported_input_tokens"], 100)
        self.assertEqual(result["by_model"]["a"]["usage"]["latency_p50_seconds"], 4)
        self.assertIn("Excludes image preprocessing", result["usage"]["latency_note"])

    def test_proposals_are_diagnostic_not_final_accuracy(self):
        rows = [prediction(status="needs_review", answer=None, proposed_answer=2), prediction(question_id="q2", status="needs_review", answer=None, proposed_answer=4), prediction(question_id="q3", status="answered", answer=2, proposed_answer=2), prediction(question_id="q4", status="error", answer=None, proposed_answer=2)]
        result = summarize(rows)
        self.assertEqual(result["exact_accuracy_all_labeled"], 0.25)
        diagnostic = result["proposal_diagnostics"]
        self.assertEqual(diagnostic["eligible_questions"], 3)
        self.assertEqual(diagnostic["excluded_error_or_unsupported"], 1)
        self.assertEqual(diagnostic["proposal_coverage"], 1)
        self.assertAlmostEqual(diagnostic["exact_accuracy_proposed_only"], 2/3)
        self.assertAlmostEqual(diagnostic["numeric_mae_proposed_only"], 2/3)
        self.assertEqual(result["by_model"]["qwen"]["proposal_diagnostics"], diagnostic)

    def test_missing_proposals_count_against_proposal_coverage(self):
        result = summarize([prediction(proposed_answer=2), prediction(question_id="q2", status="needs_review", answer=None, proposed_answer=None)])["proposal_diagnostics"]
        self.assertEqual(result["proposal_coverage"], 0.5)
        self.assertEqual(result["exact_accuracy_all_eligible_labeled"], 0.5)
        self.assertEqual(result["exact_accuracy_proposed_only"], 1)

    def test_unlabeled_or_boolean_proposals_do_not_invent_numeric_scores(self):
        result = summarize([prediction(status="needs_review", answer=None, proposed_answer=True, expected_answer=1)])["proposal_diagnostics"]
        self.assertEqual(result["exact_accuracy_proposed_only"], 0)
        self.assertIsNone(result["numeric_mae_proposed_only"])
        result = summarize([prediction(has_ground_truth=False, proposed_answer=2)])["proposal_diagnostics"]
        self.assertIsNone(result["exact_accuracy_proposed_only"])

    def test_unknown_cost_does_not_become_zero(self):
        rows = [prediction(usage=Usage(estimated_cost_usd=0.01)), prediction(question_id="q2", usage=Usage(estimated_cost_usd=None, usage_reported=False))]
        result = summarize(rows)["usage"]
        self.assertIsNone(result["estimated_cost_usd"])
        self.assertFalse(result["cost_complete"])
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["known_cost_subtotal_usd"], 0.01)

    def test_bilingual_counts_collapsed_but_consistency_not_accuracy(self):
        rows = [prediction(answer=3), prediction(question_id="qfi", question="Kuinka monta ikkunaa?", answer=3)]
        result = summarize(rows)
        self.assertEqual(result["unique_facts"]["facts"], 1)
        self.assertEqual(result["unique_facts"]["wording_consistency"], 1)
        self.assertEqual(result["unique_facts"]["exact_accuracy_all_wordings"], 0)

    def test_page_scopes_are_distinct(self):
        rows = [prediction(page=1), prediction(question_id="q2", page=2)]
        self.assertEqual(summarize(rows)["unique_facts"]["facts"], 2)

    def test_generic_questions_are_not_collapsed(self):
        rows = [prediction(category="general", unit=None), prediction(category="general", unit=None, question_id="q2")]
        self.assertEqual(summarize(rows)["unique_facts"]["facts"], 2)

    def test_conflicting_labels_are_reported_not_scored_as_one_fact(self):
        rows = [prediction(), prediction(question_id="q2", expected_answer=3)]
        result = summarize(rows)["unique_facts"]
        self.assertEqual(result["conflicting_ground_truth_groups"], 1)
        self.assertIsNone(result["exact_accuracy_all_wordings"])

    def test_zero_count_without_evidence_is_allowed(self):
        result = summarize([prediction(answer=0, expected_answer=0)])
        self.assertEqual(result["count_zero"]["exact_accuracy_all_labeled"], 1)
        self.assertEqual(result["evidence"]["answered_with_evidence"], 0)
        self.assertIsNone(result["evidence"]["localization_accuracy"])

    def test_empty_summary(self):
        result = summarize([])
        self.assertEqual(result["questions"], 0)
        self.assertIsNone(result["exact_accuracy_all_labeled"])
        json.dumps(result, allow_nan=False)

    def test_compare_uses_paired_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runs = []
            for name, rows in [("a", [prediction(), prediction(question_id="q2")]), ("b", [prediction(model="nemo", answer=4)])]:
                folder = base / name
                folder.mkdir()
                (folder / "predictions.jsonl").write_text("\n".join(p.model_dump_json() for p in rows))
                runs.append(folder)
            result = compare_runs(runs, base / "compare")
            self.assertEqual(result["shared_questions"], 1)
            self.assertEqual(result["union_questions"], 2)
            self.assertTrue(result["warnings"])
            self.assertEqual(result["runs"]["2:b/nemo"]["paired"]["numeric_mae_answered_only"], 2)
            self.assertTrue((base / "compare" / "paired_answers.csv").is_file())

    def test_compare_excludes_id_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runs = []
            for name, p in [("a", prediction()), ("b", prediction(model="nemo", source_sha256="different"))]:
                folder = base / name
                folder.mkdir()
                (folder / "predictions.jsonl").write_text(p.model_dump_json())
                runs.append(folder)
            result = compare_runs(runs, base / "out")
            self.assertEqual(result["shared_questions"], 0)
            self.assertEqual(result["excluded_identity_conflicts"], ["q1"])
            self.assertIsNone(result["runs"]["1:a/qwen"]["paired"]["exact_accuracy_all_labeled"])

    def test_compare_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runs = []
            for name in ("a", "b"):
                folder = base / name
                folder.mkdir()
                (folder / "predictions.jsonl").write_text(prediction().model_dump_json()+"\n"+prediction().model_dump_json())
                runs.append(folder)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                compare_runs(runs, base / "out")


if __name__ == "__main__":
    unittest.main()

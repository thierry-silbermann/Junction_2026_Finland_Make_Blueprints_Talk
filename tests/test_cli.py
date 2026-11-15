"""Batch/resume integration checks using an explicit synthetic model provider."""
import contextlib
import io
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image

from drawingqa.cli import read_predictions, run_evaluation
from drawingqa.data import load_questions
from drawingqa.evaluation import compare_runs
from drawingqa.schemas import AppConfig, ModelConfig, ModelResponse, PipelineConfig, Question, Usage


class SyntheticFactory:
    """In-memory fixture only: these tests never contact a model endpoint."""

    def __init__(self, counts=None, fail_remaining=0):
        self.calls = []
        self.counts = counts or {"test-qwen": 1, "test-nemotron": 2}
        self.fail_remaining = fail_remaining
        self.lock = threading.Lock()

    def __call__(self, config):
        parent = self

        class Provider:
            def generate(self, system, user, image_paths):
                with parent.lock:
                    parent.calls.append((config.model, system, user, list(image_paths)))
                    if parent.fail_remaining:
                        parent.fail_remaining -= 1
                        raise RuntimeError("Synthetic transient error")
                count = parent.counts[config.model]
                payload = {
                    "answer": count,
                    "candidates": [{"label": "synthetic window", "bbox": [0.1 + 0.3 * i, 0.1, 0.2 + 0.3 * i, 0.2]} for i in range(count)],
                    "complete": True,
                    "rationale": "Synthetic fixture; no real inference.",
                }
                return ModelResponse(content=json.dumps(payload), usage=Usage(requests=1, input_tokens=20, output_tokens=30, estimated_cost_usd=0.001), finish_reason="stop")

            def close(self):
                pass

        return Provider()


class CliBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.source = self.data / "plan.png"
        Image.new("RGB", (300, 200), "white").save(self.source)
        self.config = AppConfig(
            models={
                "qwen": ModelConfig(model="test-qwen"),
                "nemotron": ModelConfig(model="test-nemotron", base_url="http://localhost:8002/v1"),
            },
            pipeline=PipelineConfig(cache_dir=str(self.root / "cache"), max_concurrency=2),
        )
        self.question = Question(id="windows-en", path="plan.png", question="How many windows are shown on this floor plan?", answer=1, unit="count", type="count")
        self.factory = SyntheticFactory()

    def tearDown(self):
        self.temp.cleanup()

    def run_batch(self, questions=None, *, models=None, output=None, resume=False, config=None, factory=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_evaluation(config or self.config, questions or [self.question], self.data,
                                  output or self.root / "run", models or ["qwen"], resume=resume,
                                  provider_factory=factory or self.factory)

    def test_batch_continues_after_missing_image(self):
        missing = self.question.model_copy(update={"id": "missing", "path": "absent.png"})
        rows = self.run_batch([missing, self.question])
        self.assertEqual([r.status for r in rows], ["error", "answered"])
        self.assertEqual(len(self.factory.calls), 1)
        self.assertTrue((self.root / "run" / "report.html").is_file())
        self.assertTrue((self.root / "run" / "summary.json").is_file())
        saved = read_predictions(self.root / "run" / "predictions.jsonl")
        self.assertEqual(len(saved), 2)
        self.assertTrue(all(r.has_ground_truth for r in rows))

    def test_gpu_time_cost_reported_for_hourly_models(self):
        self.run_batch()
        summary = json.loads((self.root / "run" / "summary.json").read_text())
        self.assertIsNone(summary["estimated_gpu_cost_usd"])
        config = self.config.model_copy(deep=True)
        config.models["qwen"].gpu_cost_per_hour = 3600.0
        self.run_batch(config=config, output=self.root / "gpu-run")
        summary = json.loads((self.root / "gpu-run" / "summary.json").read_text())
        self.assertAlmostEqual(summary["estimated_gpu_cost_usd"], summary["run_wall_seconds"])

    def test_resume_does_not_make_more_inference_calls(self):
        first = self.run_batch()
        self.assertEqual(len(self.factory.calls), 1)
        again = self.run_batch(resume=True)
        self.assertEqual(len(self.factory.calls), 1)
        self.assertEqual(first[0].model_dump(), again[0].model_dump())
        self.assertEqual(len(read_predictions(self.root / "run" / "predictions.jsonl")), 1)

    def test_existing_run_requires_explicit_resume(self):
        self.run_batch()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_batch()
        self.assertEqual(len(self.factory.calls), 1)

    def test_changed_image_blocks_resume_without_overwriting(self):
        self.run_batch()
        predictions = self.root / "run" / "predictions.jsonl"
        before = predictions.read_bytes()
        Image.new("RGB", (300, 200), "black").save(self.source)
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.run_batch(resume=True)
        self.assertEqual(predictions.read_bytes(), before)
        self.assertEqual(len(self.factory.calls), 1)

    def test_changed_config_blocks_resume(self):
        self.run_batch()
        changed = self.config.model_copy(deep=True)
        changed.models["qwen"].temperature = 0.4
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.run_batch(config=changed, resume=True)
        self.assertEqual(len(self.factory.calls), 1)

    def test_changed_question_blocks_resume(self):
        self.run_batch()
        changed = self.question.model_copy(update={"question": "How many doors are shown on this floor plan?"})
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.run_batch([changed], resume=True)

    def test_truncated_final_append_is_recovered_and_retried(self):
        doors = self.question.model_copy(update={"id": "doors-en", "question": "How many doors are shown on this floor plan? Count doorways fitted with a door; open doorways without a door leaf do not count."})
        questions = [self.question, doors]
        self.run_batch(questions)
        target = self.root / "run" / "predictions.jsonl"
        lines = target.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        target.write_text(lines[0] + "\n" + lines[1][:50])
        shutil.rmtree(self.root / "cache")
        before = len(self.factory.calls)
        recovered = self.run_batch(questions, resume=True)
        self.assertEqual(len(recovered), 2)
        self.assertEqual(len(self.factory.calls), before + 1)
        self.assertEqual(len(read_predictions(target)), 2)
        self.assertEqual({r.question_id for r in recovered}, {"windows-en", "doors-en"})

    def test_schema_invalid_complete_final_row_is_not_silently_discarded(self):
        target = self.root / "broken.jsonl"
        target.write_text('{"question_id":"bad","status":"answered"}\n')
        with self.assertRaises(ValueError):
            read_predictions(target)

    def test_corrupt_newline_terminated_final_row_is_not_assumed_truncated(self):
        target = self.root / "broken.jsonl"
        target.write_text('{"broken":\n')
        with self.assertRaises(ValueError):
            read_predictions(target)

    def test_interior_corruption_raises(self):
        self.run_batch()
        valid = (self.root / "run" / "predictions.jsonl").read_text()
        target = self.root / "broken.jsonl"
        target.write_text('{broken}\n' + valid)
        with self.assertRaises(ValueError):
            read_predictions(target)

    def test_error_rows_are_retried_on_resume(self):
        failing = SyntheticFactory(fail_remaining=1)
        first = self.run_batch(factory=failing)
        self.assertEqual(first[0].status, "error")
        second = self.run_batch(resume=True, factory=failing)
        self.assertEqual(second[0].status, "answered")
        self.assertEqual(len(failing.calls), 2)
        saved = read_predictions(self.root / "run" / "predictions.jsonl")
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].status, "answered")

    def test_resume_preserves_usage_from_failed_attempts(self):
        from drawingqa.providers import ProviderError
        class FailedProvider:
            def generate(self, *args):
                raise ProviderError("Fixture error after billed request", Usage(requests=1, input_tokens=12,
                                    output_tokens=8, estimated_cost_usd=0.002, latency_seconds=2))
            def close(self):
                pass
        first = self.run_batch(factory=lambda config: FailedProvider())
        self.assertEqual(first[0].status, "error")
        resumed = self.run_batch(resume=True)
        self.assertEqual(resumed[0].status, "answered")
        self.assertEqual(resumed[0].usage.requests, 2)
        self.assertEqual(resumed[0].usage.input_tokens, 32)
        self.assertAlmostEqual(resumed[0].usage.estimated_cost_usd, 0.003)
        self.assertEqual(resumed[0].usage.latency_seconds, 2)

    def test_multi_model_runs_remain_separate_and_compare(self):
        rows = self.run_batch(models=["qwen", "nemotron"])
        self.assertEqual([(p.model, p.answer) for p in rows], [("qwen", 1), ("nemotron", 2)])
        self.assertEqual([c[0] for c in self.factory.calls], ["test-qwen", "test-nemotron"])
        second = self.root / "second"
        self.run_batch(output=second)
        self.assertEqual(len(self.factory.calls), 2, "Compatible qwen extraction should be cached")
        comparison = compare_runs([self.root / "run", second], self.root / "comparison")
        self.assertEqual(len(comparison["runs"]), 3)
        self.assertEqual(comparison["shared_questions"], 1)
        self.assertTrue((self.root / "comparison" / "comparison.csv").is_file())

    def test_duplicate_question_ids_are_rejected_before_inference(self):
        other = self.question.model_copy(update={"question": "How many floors (storeys) does this floor plan show?"})
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            self.run_batch([self.question, other])
        self.assertEqual(self.factory.calls, [])

    def test_jsonl_loader_rejects_duplicate_explicit_and_derived_ids(self):
        target = self.root / "questions.jsonl"
        for item in (self.question, Question(path="plan.png", question="Count windows")):
            with self.subTest(id=item.id):
                target.write_text(item.model_dump_json() + "\n" + item.model_dump_json() + "\n")
                with self.assertRaisesRegex(ValueError, "Duplicate question id"):
                    load_questions(target)

    def test_duplicate_or_unknown_model_alias_is_rejected(self):
        for aliases in (["qwen", "qwen"], ["not-configured"]):
            with self.subTest(aliases=aliases), self.assertRaises(ValueError):
                self.run_batch(models=aliases)
        self.assertEqual(self.factory.calls, [])

    def test_orphan_predictions_without_manifest_are_preserved(self):
        output = self.root / "orphan"
        output.mkdir()
        target = output / "predictions.jsonl"
        target.write_text("do not overwrite")
        with self.assertRaisesRegex(ValueError, "without a manifest"):
            self.run_batch(output=output)
        self.assertEqual(target.read_text(), "do not overwrite")

    def test_resume_rejects_unknown_saved_task_key(self):
        self.run_batch()
        target = self.root / "run" / "predictions.jsonl"
        row = json.loads(target.read_text())
        row["question_id"] = "not-in-manifest"
        target.write_text(json.dumps(row) + "\n")
        with self.assertRaises(ValueError):
            self.run_batch(resume=True)
        self.assertEqual(len(self.factory.calls), 1)

    def test_resume_rejects_duplicate_saved_task_keys(self):
        self.run_batch()
        target = self.root / "run" / "predictions.jsonl"
        line = target.read_text()
        target.write_text(line + line)
        with self.assertRaises(ValueError):
            self.run_batch(resume=True)
        self.assertEqual(len(self.factory.calls), 1)

    def test_resume_rejects_wrong_source_fingerprint(self):
        self.run_batch()
        target = self.root / "run" / "predictions.jsonl"
        row = json.loads(target.read_text())
        row["source_sha256"] = "incorrect-source-hash"
        target.write_text(json.dumps(row) + "\n")
        with self.assertRaises(ValueError):
            self.run_batch(resume=True)
        self.assertEqual(len(self.factory.calls), 1)

    def test_resume_rejects_wrong_config_fingerprint(self):
        self.run_batch()
        target = self.root / "run" / "predictions.jsonl"
        row = json.loads(target.read_text())
        row["config_sha256"] = "incorrect-config-hash"
        target.write_text(json.dumps(row) + "\n")
        with self.assertRaises(ValueError):
            self.run_batch(resume=True)
        self.assertEqual(len(self.factory.calls), 1)


if __name__ == "__main__":
    unittest.main()

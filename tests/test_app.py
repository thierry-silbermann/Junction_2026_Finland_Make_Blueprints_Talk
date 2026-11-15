"""Demo app endpoints, driven by a fake provider; benchmark labels never reach it."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from PIL import Image

try:
    from fastapi.testclient import TestClient
    from drawingqa.app.server import create_app
except ImportError:  # the app extra is optional
    TestClient = None

from drawingqa.evaluation import summarize
from drawingqa.schemas import AppConfig, ModelConfig, ModelResponse, PipelineConfig, Prediction, Usage

INVENTORIES = {
    "openings": {"items": [{"kind": "door", "bbox": [0.1, 0.1, 0.2, 0.3]},
                           {"kind": "door", "bbox": [0.6, 0.1, 0.7, 0.3]},
                           {"kind": "window", "bbox": [0.4, 0.0, 0.5, 0.05]}], "complete": True},
    "fixtures": {"items": [{"kind": "toilet", "bbox": [0.3, 0.6, 0.35, 0.7]}], "complete": True},
    "spaces": {"items": [{"kind": "room", "bbox": [0.0, 0.0, 0.5, 0.5], "text": "MH", "room_type": "bedroom"}],
               "complete": True},
    "structure": {"items": [], "complete": True},
}


class FakeFactory:
    def __init__(self):
        self.calls = []
        self.guard = threading.Lock()

    def __call__(self, config):
        factory = self

        class Fake:
            def generate(self, system, user, image_paths):
                with factory.guard:
                    factory.calls.append(system + user)
                group = next((g for g in INVENTORIES if f"Inventory group: {g}\n" in user), None)
                payload = INVENTORIES[group] if group else {
                    "answer": 2, "complete": True, "rationale": "Two openings.",
                    "candidates": [{"label": "window", "bbox": [0.1, 0.1, 0.2, 0.3]},
                                   {"label": "window", "bbox": [0.6, 0.1, 0.7, 0.3]}]}
                return ModelResponse(content=json.dumps(payload),
                                     usage=Usage(requests=1, input_tokens=20, output_tokens=10, estimated_cost_usd=0))

            def close(self):
                pass
        return Fake()


@unittest.skipIf(TestClient is None, "app extra not installed")
class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data = root / "data"
        (self.data / "plans").mkdir(parents=True)
        Image.new("RGB", (640, 480), "white").save(self.data / "plans" / "plan.png")
        (self.data / "questions.jsonl").write_text("\n".join(json.dumps(row) for row in [
            {"path": "data/plans/plan.png", "question": "How many doors are shown on this floor plan?",
             "answer": 2, "unit": "count", "type": "count"},
            {"path": "data/plans/plan.png", "question": "What colour is the roof?", "answer": "PRIVATE_EXPECTED_999"},
        ]) + "\n")
        run = root / "runs" / "saved"
        (run / "assets").mkdir(parents=True)
        Image.new("RGB", (64, 48), "white").save(run / "assets" / "page.png")
        saved = Prediction(question_id="q1", path="data/plans/plan.png",
                           question="How many doors are shown on this floor plan?", model="test",
                           model_id="nvidia/nemotron-test", status="answered", answer=2, proposed_answer=2,
                           unit="count", category="doors", page_images={"1": "assets/page.png"},
                           expected_answer=2, has_ground_truth=True)
        (run / "predictions.jsonl").write_text(saved.model_dump_json() + "\n")
        (run / "summary.json").write_text(json.dumps(summarize([saved])))
        (run / "manifest.json").write_text("{}")
        self.factory = FakeFactory()
        showcase = root / "showcase.json"
        showcase.write_text(json.dumps([
            {"title": "Plan · doors", "mode": "live", "expect": "exact", "path": "plans/plan.png",
             "question": "How many doors are shown on this floor plan?", "answer": 2, "unit": "count"},
            {"title": "Missing", "mode": "instant", "expect": "exact", "path": "plans/gone.png", "question": "x"}]))
        config = AppConfig(models={"test": ModelConfig(model="fake-vision")},
                           pipeline=PipelineConfig(cache_dir=str(root / "cache"), strategy="per_drawing"))
        app = create_app(config, self.data, runs_dir=root / "runs", output=root / "runs" / "live",
                         questions=self.data / "questions.jsonl", default_run="saved",
                         provider_factory=self.factory, probe_models=False, showcase=showcase)
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_detect_counts_every_group_then_reuses_the_cache(self):
        first = self.client.post("/api/detect", json={"path": "plans/plan.png", "model": "test"}).json()
        self.assertEqual(first["counts"]["door"], {"total": 2, "uncertain": 0})
        self.assertEqual(first["counts"]["window"]["total"], 1)
        self.assertEqual(first["room_types"], {"bedroom": 1})
        self.assertEqual(set(first["groups"]), {"openings", "fixtures", "spaces", "structure"})
        self.assertTrue(first["live"])
        self.assertTrue(self.client.get(first["page_urls"]["1"]).status_code == 200)
        second = self.client.post("/api/detect", json={"path": "plans/plan.png", "model": "test"}).json()
        self.assertFalse(second["live"])
        self.assertEqual(len(self.factory.calls), 4)

    def test_ask_reuses_inventory_and_attaches_ground_truth_after_inference(self):
        self.client.post("/api/detect", json={"path": "plans/plan.png", "model": "test"})
        answer = self.client.post("/api/ask", json={"path": "plans/plan.png", "model": "test",
                                                    "question": "How many doors are shown on this floor plan?"}).json()
        self.assertEqual((answer["status"], answer["answer"], answer["correct"]), ("answered", 2, True))
        self.assertFalse(answer["live"])
        self.assertEqual(answer["route_label"], "Vision model")
        self.client.post("/api/ask", json={"path": "plans/plan.png", "model": "test", "question": "What colour is the roof?"})
        self.assertFalse(any("PRIVATE_EXPECTED_999" in call for call in self.factory.calls))

    def test_fresh_ask_calls_the_model_again_and_keeps_the_cache(self):
        question = {"path": "plans/plan.png", "model": "test", "question": "How many doors are shown on this floor plan?"}
        self.assertTrue(self.client.post("/api/ask", json=question).json()["live"])
        self.assertFalse(self.client.post("/api/ask", json=question).json()["live"])
        fresh = self.client.post("/api/ask", json={**question, "fresh": True}).json()
        self.assertTrue(fresh["live"])
        self.assertEqual((fresh["answer"], fresh["correct"]), (2, True))
        self.assertFalse(self.client.post("/api/ask", json=question).json()["live"])

    def test_showcase_lists_only_drawings_that_exist(self):
        examples = self.client.get("/api/showcase").json()["examples"]
        self.assertEqual([e["title"] for e in examples], ["Plan · doors"])
        self.assertEqual(examples[0]["dataset"], "plans")
        self.assertEqual(self.client.get(examples[0]["thumb"]).status_code, 200)

    def test_upload_render_and_detect(self):
        buffer = io.BytesIO()
        Image.new("RGB", (300, 200), "white").save(buffer, "PNG")
        uploaded = self.client.post("/api/upload", files={"file": ("my plan.png", buffer.getvalue(), "image/png")}).json()
        rendered = self.client.get("/api/render", params={"upload": uploaded["upload"]}).json()
        self.assertEqual(list(rendered["pages"]), ["1"])
        detected = self.client.post("/api/detect", json={"upload": uploaded["upload"], "model": "test"})
        self.assertEqual(detected.status_code, 200)
        self.assertEqual(self.client.post("/api/upload", files={"file": ("notes.txt", b"hi", "text/plain")}).status_code, 415)
        self.assertEqual(self.client.post("/api/upload", files={"file": ("fake.png", b"not an image", "image/png")}).status_code, 422)
        self.assertEqual(self.client.get("/api/render", params={"upload": "../../secret.png"}).status_code, 404)

    def test_benchmark_endpoints_join_questions_and_saved_run(self):
        run = self.client.get("/api/runs").json()
        self.assertEqual(run, {"default": "saved", "runs": ["saved"]})
        detail = self.client.get("/api/runs/saved").json()
        self.assertEqual(detail["summary"]["exact_accuracy_all_labeled"], 1.0)
        datasets = self.client.get("/api/datasets", params={"run": "saved"}).json()["datasets"]
        self.assertEqual(datasets[0]["name"], "plans")
        self.assertEqual(datasets[0]["routes"], {"Nemotron": 1})
        images = self.client.get("/api/datasets/plans/images", params={"run": "saved"}).json()
        self.assertEqual((images["total"], images["images"][0]["correct"]), (1, 1))
        rows = self.client.get("/api/image", params={"path": "plans/plan.png", "run": "saved"}).json()["questions"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["saved"]["correct"])
        self.assertEqual(self.client.get(rows[0]["saved"]["page_urls"]["1"]).status_code, 200)
        self.assertEqual(self.client.get("/api/thumb", params={"path": "plans/plan.png"}).status_code, 200)

    def test_files_stay_inside_their_roots(self):
        for url in ("/api/runs/saved/files/manifest.json", "/api/runs/saved/files/assets/../summary.json",
                    "/api/live/uploads/x.png", "/api/runs/unknown", "/api/datasets/../images"):
            self.assertEqual(self.client.get(url).status_code, 404, url)
        self.assertEqual(self.client.get("/api/thumb", params={"path": "../questions.jsonl"}).status_code, 404)
        self.assertEqual(self.client.post("/api/ask", json={"path": "plans/plan.png", "model": "nope", "question": "x"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()

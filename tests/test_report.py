import json
import re
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from drawingqa.report import _safe_image_path, create_report
from drawingqa.schemas import Evidence, Prediction


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "assets").mkdir()
        Image.new("RGB", (10, 10), "white").save(self.root / "assets" / "plan.png")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, **extra):
        data = dict(question_id="q1", path="plan.png", question="Count windows", model="qwen", model_id="x", status="answered", answer=1, category="windows", unit="count", page_images={"1": "assets/plan.png"})
        data.update(extra)
        p = Prediction(**data)
        (self.root / "predictions.jsonl").write_text(p.model_dump_json())
        return create_report(self.root).read_text()

    def payload(self, document):
        return json.loads(re.search(r'<script type="application/json" id="reportData">(.*?)</script>', document, re.S).group(1))

    def test_injection_is_inert(self):
        attack = '</script><script>alert("bad")</script>&'
        document = self.write(question=attack, rationale=attack)
        self.assertNotIn(attack, document)
        self.assertEqual(self.payload(document)["predictions"][0]["question"], attack)
        self.assertNotIn("innerHTML", document)

    def test_placeholder_like_user_text_remains_unchanged(self):
        document = self.write(question="__RUN_KEY__ __PAYLOAD__ __TITLE__")
        self.assertEqual(self.payload(document)["predictions"][0]["question"], "__RUN_KEY__ __PAYLOAD__ __TITLE__")

    def test_outside_images_are_not_linked(self):
        document = self.write(page_images={"1": "../private.png", "2": "https://remote.example/x.png", "3": "/etc/passwd", "4": "assets/plan.png"})
        payload = self.payload(document)["predictions"][0]
        self.assertEqual(payload["page_images"], {"4": "assets/plan.png"})
        self.assertEqual(len(payload["warnings"]), 3)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            source = Path(outside) / "secret.png"
            Image.new("RGB", (2, 2)).save(source)
            (self.root / "assets" / "escape.png").symlink_to(source)
            self.assertIsNone(_safe_image_path(self.root, "assets/escape.png"))

    def test_spaces_encoded_for_image_url(self):
        Image.new("RGB", (2, 2)).save(self.root / "assets" / "plan name.png")
        self.assertEqual(_safe_image_path(self.root, "assets/plan name.png"), "assets/plan%20name.png")

    def test_ifc_has_guid_without_fabricated_box(self):
        e = Evidence(id="entity-1", label="IfcWindow", bbox=None, page=1, category="windows", provenance="ifc_entity", entity_guid="guid")
        document = self.write(path="plan.ifc", page_images={}, evidence=[e])
        payload = self.payload(document)["predictions"][0]
        self.assertIsNone(payload["evidence"][0]["bbox"])
        self.assertIn("no 3D geometry viewer", document)

    def test_reviews_separate_from_predictions(self):
        document = self.write()
        self.assertIn("drawingqa-human-reviews.json", document)
        self.assertIn("evidence_locations_checked", document)
        self.assertIn("original_answer", document)
        self.assertNotIn("fetch(", document)

    def test_model_comparison_and_proposal_labels_present(self):
        document = self.write(status="needs_review", answer=None, proposed_answer=2)
        self.assertIn('id="modelMetrics"', document)
        self.assertIn('unaccepted values', document)
        self.assertIn('Median API time', document)
        metric = self.payload(document)["summary"]["by_model"]["qwen"]
        self.assertEqual(metric["proposal_diagnostics"]["proposal_coverage"], 1)
        self.assertIn("usage", metric)

    def test_missing_images_warn(self):
        document = self.write(page_images={"1": "assets/absent.png"})
        self.assertEqual(self.payload(document)["predictions"][0]["page_images"], {})


if __name__ == "__main__":
    unittest.main()

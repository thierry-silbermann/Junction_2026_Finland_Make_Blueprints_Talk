"""Integration tests use a fake provider, never benchmark labels as responses."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest

from PIL import Image

from drawingqa.imaging import View
from drawingqa.pipeline import Pipeline, attach_ground_truth, flag_repeated_boxes, fuse_evidence
from drawingqa.schemas import AppConfig, Evidence, ModelConfig, ModelResponse, PipelineConfig, Question, Usage


def visual_response(**overrides):
    result = {
        "answer": 2,
        "complete": True,
        "candidates": [
            {"label": "window", "bbox": [0.1, 0.1, 0.2, 0.3]},
            {"label": "window", "bbox": [0.6, 0.1, 0.7, 0.3]},
        ],
        "rationale": "Two distinct framed openings are visible.",
    }
    result.update(overrides)
    return result


class FakeFactory:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else visual_response()
        self.calls = []
        self.guard = threading.Lock()

    def __call__(self, config):
        factory = self
        class Fake:
            def generate(self, system, user, image_paths):
                with factory.guard:
                    factory.calls.append((system, user, list(image_paths), config.model))
                return ModelResponse(
                    content=factory.payload if isinstance(factory.payload, str) else json.dumps(factory.payload),
                    usage=Usage(requests=1, input_tokens=20, output_tokens=10, estimated_cost_usd=0.00003),
                )
            def close(self):
                pass
        return Fake()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        (self.data / "source").mkdir(parents=True)
        self.image = self.data / "source" / "plan.png"
        Image.new("RGB", (640, 480), "white").save(self.image)
        self.config = AppConfig(
            models={"test": ModelConfig(model="fake-vision")},
            pipeline=PipelineConfig(cache_dir=str(self.root / "cache")),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def pipeline(self, factory=None, config=None):
        return Pipeline(config or self.config, self.data, self.root / "run", provider_factory=factory or FakeFactory())

    def q(self, text="How many windows are shown on this floor plan?", **kwargs):
        return Question(path="plan.png", question=text, **kwargs)

    def test_count_uses_evidence_and_ground_truth_never_reaches_provider(self):
        factory = FakeFactory()
        q = self.q(answer="PRIVATE_EXPECTED_999")
        result = self.pipeline(factory).predict(q, "test")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer, 2)
        self.assertFalse(result.has_ground_truth)
        self.assertIsNone(result.expected_answer)
        self.assertEqual(len(result.evidence), 2)
        for system, prompt, _images, _model in factory.calls:
            self.assertNotIn("PRIVATE_EXPECTED_999", system + prompt)
        attached = attach_ground_truth(result, q)
        self.assertTrue(attached.has_ground_truth)
        self.assertEqual(attached.expected_answer, "PRIVATE_EXPECTED_999")

    def test_bilingual_and_label_changes_reuse_same_visual_cache(self):
        factory = FakeFactory()
        pipeline = self.pipeline(factory)
        a = pipeline.predict(self.q(answer=999), "test")
        b = pipeline.predict(self.q("Kuinka monta ikkunaa pohjapiirustuksessa on?", answer=777), "test")
        c = pipeline.predict(self.q(answer=555), "test")
        self.assertEqual(len(factory.calls), 1)
        self.assertEqual(a.answer, b.answer)
        self.assertEqual(b.usage.cached_requests, 1)
        self.assertEqual(c.usage.requests, 0)
        self.assertEqual(c.usage.estimated_cost_usd, 0)

    def test_source_pixels_and_model_settings_invalidate_extraction_cache(self):
        factory = FakeFactory()
        pipeline = self.pipeline(factory)
        first = pipeline.predict(self.q(), "test")
        Image.new("RGB", (640, 480), "black").save(self.image)
        second = pipeline.predict(self.q(), "test")
        changed = self.config.model_copy(deep=True)
        changed.models["test"].temperature = 0.5
        third = self.pipeline(factory, changed).predict(self.q(), "test")
        self.assertEqual(len(factory.calls), 3)
        self.assertNotEqual(first.source_sha256, second.source_sha256)
        self.assertNotEqual(second.config_sha256, third.config_sha256)

    def test_concurrent_identical_requests_share_single_flight(self):
        factory = FakeFactory()
        pipeline = self.pipeline(factory)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: pipeline.predict(self.q(), "test"), range(2)))
        self.assertTrue(all(r.status == "answered" for r in results))
        self.assertEqual(len(factory.calls), 1)

    def test_incomplete_or_uncertain_inventory_abstains(self):
        for payload in [visual_response(complete=False), visual_response(candidates=[
            {"label": "window", "bbox": [0.1, 0.1, 0.2, 0.3], "uncertain": True},
        ], answer=0)]:
            # Different isolated caches prevent one fixture affecting another.
            config = self.config.model_copy(deep=True)
            config.pipeline.cache_dir = str(self.root / f"cache-{len(json.dumps(payload))}")
            result = self.pipeline(FakeFactory(payload), config).predict(self.q(), "test")
            self.assertEqual(result.status, "needs_review")
            self.assertIsNone(result.answer)

    def test_stated_inventory_count_mismatch_requires_review(self):
        result = self.pipeline(FakeFactory(visual_response(answer=5, candidates=[]))).predict(self.q(), "test")
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.answer)

    def test_general_answer_without_localized_evidence_abstains(self):
        result = self.pipeline(FakeFactory({"answer": "3 m", "complete": True, "candidates": []})).predict(
            self.q("What is the printed ceiling height?"), "test")
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.answer)

    def test_invalid_model_json_errors_without_creating_good_cache_entry(self):
        factory = FakeFactory('{"answer": 2, incomplete')
        pipeline = self.pipeline(factory)
        first = pipeline.predict(self.q(), "test")
        second = pipeline.predict(self.q(), "test")
        self.assertEqual(first.status, "error")
        self.assertEqual(second.status, "error")
        self.assertEqual(len(factory.calls), 2)

    def test_direct_count_disagreement_does_not_become_verified_answer(self):
        config = self.config.model_copy(deep=True)
        config.pipeline.mode = "direct"
        result = self.pipeline(FakeFactory(visual_response(answer=3)), config).predict(self.q(), "test")
        self.assertEqual(result.proposed_answer, 3)
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.answer)

    def test_multipage_tiff_occurrences_require_review_but_selected_page_is_scoped(self):
        frames = [Image.new("RGB", (80, 60), color) for color in ["white", "gray"]]
        file = self.data / "two.tiff"
        frames[0].save(file, save_all=True, append_images=frames[1:])
        factory = FakeFactory()
        pipeline = self.pipeline(factory)
        all_pages = pipeline.predict(Question(path="two.tiff", question="How many windows?"), "test")
        page_two = pipeline.predict(Question(path="two.tiff", question="How many windows?", page=2), "test")
        self.assertEqual(all_pages.status, "needs_review")
        self.assertEqual(all_pages.proposed_answer, 4)
        self.assertEqual(page_two.status, "answered")
        self.assertEqual(page_two.answer, 2)
        self.assertEqual(set(page_two.page_images), {"2"})
        self.assertTrue(all(e.page == 2 for e in page_two.evidence))


class FusionTests(unittest.TestCase):
    def item(self, bbox, *, page=1, view="overview", label="window", category="windows"):
        return Evidence(id="temp", label=label, bbox=bbox, page=page, category=category, view_ids=[view])

    def test_tile_box_coordinates_are_mapped_to_page(self):
        view = View("tile", 1, Path("unused.png"), (0.25, 0.5, 0.75, 1.0))
        for actual, expected in zip(view.to_page((0.2, 0.2, 0.8, 0.8)), (0.35, 0.6, 0.65, 0.9)):
            self.assertAlmostEqual(actual, expected)

    def test_same_entity_across_views_merges_but_separate_pages_do_not(self):
        box = (0.1, 0.1, 0.2, 0.3)
        items = [self.item(box), self.item(box, view="tile"), self.item(box, page=2)]
        fused, _ = fuse_evidence(items, 0.55)
        self.assertEqual(len(fused), 2)
        self.assertEqual(set(fused[0].view_ids), {"overview", "tile"})

    def test_conflicting_count_flags_make_merged_entity_uncertain(self):
        one = self.item((0.1, 0.1, 0.2, 0.3))
        two = self.item((0.1, 0.1, 0.2, 0.3), view="tile")
        two.counted = False
        fused, warnings = fuse_evidence([one, two], 0.55)
        self.assertEqual(len(fused), 1)
        self.assertTrue(fused[0].uncertain)
        self.assertTrue(warnings)

    def test_overlapping_different_entity_classes_are_not_silently_merged(self):
        box = (0.1, 0.1, 0.2, 0.3)
        fused, warnings = fuse_evidence([
            self.item(box, label="shower", category="showers_bathtubs"),
            self.item(box, label="bathtub", category="showers_bathtubs", view="tile"),
        ], 0.55)
        self.assertEqual(len(fused), 2)
        self.assertTrue(warnings or any(e.uncertain for e in fused))

    def test_high_overlap_same_view_is_flagged_not_silently_collapsed(self):
        box = (0.1, 0.1, 0.2, 0.3)
        fused, warnings = fuse_evidence([self.item(box), self.item(box)], 0.55)
        self.assertEqual(len(fused), 2)
        self.assertTrue(warnings or any(e.uncertain for e in fused))

    def test_identical_boxes_stamped_along_a_line_are_flagged(self):
        stamped = [self.item((0.21, y, 0.23, y + 0.05)) for y in (0.365, 0.5, 0.65, 0.75)]
        self.assertTrue(flag_repeated_boxes(stamped))
        self.assertTrue(all(e.uncertain for e in stamped))
        scattered = [self.item(box) for box in ((0.1, 0.1, 0.2, 0.3), (0.5, 0.1, 0.6, 0.3),
                                                (0.1, 0.6, 0.2, 0.8), (0.5, 0.6, 0.6, 0.8))]
        self.assertEqual(flag_repeated_boxes(scattered), [])
        self.assertFalse(any(e.uncertain for e in scattered))


if __name__ == "__main__":
    unittest.main()

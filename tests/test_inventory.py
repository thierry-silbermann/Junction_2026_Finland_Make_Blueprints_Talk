"""Per-drawing inventory strategy: selection, deterministic answers and request sharing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from drawingqa import inventory
from drawingqa.config import load_config
from drawingqa.intents import resolve_intent
from drawingqa.pipeline import Pipeline
from drawingqa.schemas import AppConfig, InventoryItem, ModelConfig, ModelResponse, PipelineConfig, Profile, Question, Usage

PLAN = Path("plan.png")


def selector(text, path=PLAN, unit=None):
    q = Question(path=str(path), question=text, unit=unit)
    return inventory.select(q, resolve_intent(q), path)


def item(kind, box=(0.1, 0.1, 0.2, 0.2), **fields):
    return InventoryItem(kind=kind, bbox=box, **fields)


INVENTORY = {
    "complete": True,
    "items": [
        {"kind": "door", "bbox": [10, 10, 60, 60]},
        {"kind": "door", "bbox": [300, 10, 350, 60]},
        {"kind": "window", "bbox": [500, 0, 600, 30]},
        {"kind": "room", "bbox": [0, 100, 400, 500], "text": "MH", "room_type": "bedroom"},
        {"kind": "room", "bbox": [400, 100, 900, 500], "text": "OH", "room_type": "living"},
    ],
}


class Factory:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def __call__(self, config):
        factory = self

        class Fake:
            def generate(self, system, user, image_paths):
                factory.calls += 1
                payload = factory.payload
                return ModelResponse(content=payload if isinstance(payload, str) else json.dumps(payload),
                                     usage=Usage(requests=1, input_tokens=100, output_tokens=50))

            def close(self):
                pass
        return Fake()


class SelectTests(unittest.TestCase):
    def test_canonical_counts_and_room_types(self):
        self.assertEqual(selector("How many doors are shown on this floor plan?").kinds, ("door",))
        self.assertEqual(selector("Kuinka monta ovea pohjapiirustuksessa on?").kinds, ("door",))
        bedrooms = selector("How many bedrooms (makuuhuone, usually marked MH) are on this floor plan?")
        self.assertEqual((bedrooms.kinds, bedrooms.room_type), (("room",), "bedroom"))
        self.assertEqual(selector("How many sinks and wash basins (kitchen and bathroom combined; a double-bowl sink counts as one) are drawn on this floor plan?").kinds,
                         ("washbasin", "kitchen_sink"))

    def test_swiss_templates(self):
        named = selector("Wie viele Räume mit der Bezeichnung 'Zimmer' gibt es auf diesem Geschoss?")
        self.assertEqual((named.operation, named.name), ("count", "Zimmer"))
        room = selector("What is the net floor area of room 7.01 ('Balkon') in m²?")
        self.assertEqual((room.operation, room.number), ("area", "7.01"))
        apartment = selector("What is the net floor area of apartment B in m², summing all its rooms except balconies")
        self.assertEqual((apartment.apartment, apartment.indoor_only), ("B", True))
        self.assertEqual(selector("How many apartments (residential units) are shown on this floor plan?").operation, "count_labels")

    def test_ifc_plan_storey_questions_only(self):
        pdf = Path("m_plan-01.pdf")
        area = selector("What is the net floor area of the room 'Bedroom (414B-5-003-001)' on '3. Level', in m²?", pdf, "m2")
        self.assertEqual((area.operation, area.name), ("area", "Bedroom (414B-5-003-001)"))
        self.assertIsNone(selector("How many doors are there in the model 'Duplex'?", pdf))
        self.assertIsNone(selector("What is the total length of all walls on 'Grundplan', in metres?", pdf, "m"))
        self.assertIsNone(selector("Which room has the largest window?"))


class AnswerTests(unittest.TestCase):
    def entries(self, *items):
        return [(1, "p1-overview", i) for i in items]

    def test_count_reviews_uncertain_items(self):
        sel = selector("How many doors are shown on this floor plan?")
        clean = inventory.answer(sel, self.entries(item("door"), item("door", (0.5, 0.5, 0.6, 0.6))), True)
        self.assertEqual((clean["proposed"], clean["review"]), (2, False))
        doubtful = inventory.answer(sel, self.entries(item("door"), item("door", uncertain=True)), True)
        self.assertTrue(doubtful["review"])
        self.assertTrue(inventory.answer(sel, self.entries(item("door")), False)["review"])

    def test_room_area_by_number_and_ifc_label(self):
        rooms = self.entries(item("room", text="Balkon", number="7.01", area=6.5),
                             item("room", text="Bedroom", number="414B-5-003-001", area=10.71))
        self.assertEqual(inventory.answer(selector("What is the net floor area of room 7.01 ('Balkon') in m²?"), rooms, True)["proposed"], 6.5)
        ifc = selector("What is the net floor area of the room 'Bedroom (414B-5-003-001)' on '3. Level', in m²?", Path("m_plan-01.pdf"), "m2")
        self.assertEqual(inventory.answer(ifc, rooms, True)["proposed"], 10.71)

    def test_sums_exclude_outdoor_and_require_printed_areas(self):
        total = selector("What is the total net floor area in m² of all numbered indoor rooms on this floor (excluding balconies)")
        rooms = self.entries(item("room", text="Zimmer", number="0.1", area=12.0),
                             item("room", text="Bad", number="0.2", area=4.25),
                             item("room", text="Balkon", number="0.3", area=6.0))
        self.assertEqual(inventory.answer(total, rooms, True)["proposed"], 16.25)
        missing = inventory.answer(total, rooms + self.entries(item("room", text="Küche", number="0.4")), True)
        self.assertIsNone(missing["proposed"])
        self.assertTrue(missing["review"])


class PipelineInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "data").mkdir()
        Image.new("RGB", (640, 480), "white").save(root / "data" / "plan.png")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def run_questions(self, strategy, *texts):
        config = AppConfig(models={"test": ModelConfig(model="fake", bbox_scale=1000)},
                           pipeline=PipelineConfig(cache_dir=str(self.root / "cache"), strategy=strategy))
        factory = Factory(INVENTORY)
        pipeline = Pipeline(config, self.root / "data", self.root / f"run-{strategy}", provider_factory=factory)
        return [pipeline.predict(Question(path="plan.png", question=t), "test") for t in texts], factory

    def test_one_request_answers_every_question_on_the_page(self):
        results, factory = self.run_questions(
            "per_drawing",
            "How many doors are shown on this floor plan?",
            "Kuinka monta ovea pohjapiirustuksessa on?",
            "How many windows are shown on this floor plan?",
            "How many bedrooms (makuuhuone, usually marked MH) are on this floor plan?",
        )
        # Doors and windows share the openings request; bedrooms need the spaces request.
        self.assertEqual(factory.calls, 2)
        self.assertEqual([r.answer for r in results], [2, 2, 1, 1])
        self.assertTrue(all(r.route == "vlm-inventory" and r.status == "answered" for r in results))
        self.assertEqual(results[0].evidence[0].bbox, (0.01, 0.01, 0.06, 0.06))
        self.assertEqual(sum(r.usage.requests for r in results), 2)
        self.assertEqual(sum(r.usage.cached_requests for r in results), 2)

    def test_failed_inventory_is_requested_once_then_falls_back(self):
        config = AppConfig(models={"test": ModelConfig(model="fake")},
                           pipeline=PipelineConfig(cache_dir=str(self.root / "cache"), strategy="per_drawing"))
        prompts = []
        factory = Factory("not json")
        original = factory.__call__

        def recording(model_config):
            fake = original(model_config)
            generate = fake.generate
            fake.generate = lambda system, user, images: (prompts.append(user), generate(system, user, images))[1]
            return fake
        pipeline = Pipeline(config, self.root / "data", self.root / "run-fail", provider_factory=recording)
        first = pipeline.predict(Question(path="plan.png", question="How many doors are shown on this floor plan?"), "test")
        second = pipeline.predict(Question(path="plan.png", question="How many windows are shown on this floor plan?"), "test")
        self.assertEqual(sum("Inventory group: openings" in p for p in prompts), 1)
        self.assertIsNone(first.route)
        self.assertTrue(any("inventory failed" in w for w in second.warnings))

    def test_unrecognized_questions_keep_the_per_question_route(self):
        results, factory = self.run_questions("per_drawing", "Which room has the largest window?")
        self.assertIsNone(results[0].route)
        self.assertEqual(factory.calls, 1)

    def test_default_strategy_is_unchanged(self):
        results, _ = self.run_questions("per_question", "How many doors are shown on this floor plan?")
        self.assertIsNone(results[0].route)


class ProfileTests(unittest.TestCase):
    def test_profile_overrides_pipeline_and_models(self):
        config = AppConfig(models={"m": ModelConfig(model="x")},
                           profiles={"reduced": Profile(models=["m"], pipeline={"strategy": "per_drawing"})})
        self.assertEqual(config.apply_profile("reduced"), ["m"])
        self.assertEqual(config.pipeline.strategy, "per_drawing")
        with self.assertRaises(ValueError):
            config.apply_profile("gpu")

    def test_shipped_profiles_cover_the_three_modes(self):
        for name, strategy, concurrency in [("normal", "per_question", 2), ("reduced", "per_drawing", 2),
                                            ("gpu", "per_drawing", 48)]:
            config = load_config(Path(__file__).resolve().parents[1] / "configs" / "models.yaml")
            models = config.apply_profile(name)
            self.assertEqual(config.pipeline.strategy, strategy)
            self.assertEqual(config.pipeline.max_concurrency, concurrency)
            self.assertTrue(all(alias in config.models for alias in models))
        gpu = config.models["nemotron_gpu"]
        self.assertIsNone(gpu.api_key_env)
        self.assertTrue(gpu.base_url.startswith("http://localhost:"))
        self.assertGreater(gpu.gpu_cost_per_hour, 0)


if __name__ == "__main__":
    unittest.main()

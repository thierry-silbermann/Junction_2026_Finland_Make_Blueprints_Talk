import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from drawingqa.ifc import _classify_space, answer_ifc
from drawingqa.schemas import Intent, Question


class Entity:
    def __init__(self, number, kind, guid=None, **attributes):
        self.number, self.kind, self.GlobalId = number, kind, guid
        self.__dict__.update(attributes)

    def id(self):
        return self.number

    def is_a(self, kind=None):
        return self.kind if kind is None else kind == self.kind


class Model:
    def __init__(self, entities, schema="IFC4"):
        self.entities, self.schema = entities, schema

    def by_type(self, kind):
        if self.schema == "IFC2X3" and kind == "IfcSanitaryTerminal":
            raise RuntimeError("Entity not found in schema")
        return [e for e in self.entities if e.kind == kind]


def intent(category):
    return Intent(operation="count", category=category, canonical_question=f"Count {category}", rule="one modeled instance")


class IfcTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "model.ifc"
        self.path.write_text("ISO-10303-21; END-ISO-10303-21;")
        self.question = Question(path="model.ifc", question="How many doors?", type="count")

    def tearDown(self):
        self.temp.cleanup()

    def query(self, model, category):
        fake = types.SimpleNamespace(open=lambda path: model)
        with patch.dict(sys.modules, {"ifcopenshell": fake}):
            return answer_ifc(self.path, self.question, intent(category))

    def test_doors_exclude_openings_and_deduplicate_references(self):
        door = Entity(1, "IfcDoor", "door-guid", Name="Door A")
        result = self.query(Model([door, door, Entity(2, "IfcOpeningElement", "opening-guid")]), "doors")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer, 1)
        self.assertIsNone(result.evidence[0].bbox)
        self.assertEqual(result.evidence[0].entity_guid, "door-guid")
        self.assertFalse(result.evidence[0].verified)
        self.assertEqual(result.usage.requests, 0)
        self.assertEqual(result.usage.estimated_cost_usd, 0)

    def test_floor_entities_not_stair_entities(self):
        result = self.query(Model([
            Entity(1, "IfcBuildingStorey", "floor1", CompositionType="ELEMENT"),
            Entity(2, "IfcBuildingStorey", "floor2", CompositionType="ELEMENT"),
            Entity(3, "IfcStair", "stairs"),
        ]), "floors")
        self.assertEqual(result.answer, 2)

    def test_duplicate_guid_on_distinct_entities_needs_review(self):
        result = self.query(Model([
            Entity(1, "IfcDoor", "duplicate"),
            Entity(2, "IfcDoor", "duplicate"),
        ]), "doors")
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.proposed_answer, 1)

    def test_complex_storey_is_not_whole_storey(self):
        result = self.query(Model([
            Entity(1, "IfcBuildingStorey", "floor1", CompositionType="ELEMENT"),
            Entity(2, "IfcBuildingStorey", "storey-complex", CompositionType="COMPLEX"),
        ]), "floors")
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.proposed_answer, 1)
        self.assertIsNone(result.answer)

    def test_toiletpan_is_not_toilet_seat(self):
        result = self.query(Model([
            Entity(1, "IfcSanitaryTerminal", "pan", PredefinedType="TOILETPAN"),
            Entity(2, "IfcSanitaryTerminal", "seat", PredefinedType="WCSEAT"),
            Entity(3, "IfcSanitaryTerminal", "cistern", PredefinedType="CISTERN"),
        ]), "toilets")
        self.assertEqual(result.answer, 1)
        self.assertEqual(result.evidence[0].entity_guid, "pan")

    def test_typed_fixtures_inherit_ifc2x3_type(self):
        sink_type = Entity(2, "IfcSanitaryTerminalType", "sink-type", PredefinedType="SINK")
        terminal = Entity(1, "IfcFlowTerminal", "sink", IsDefinedBy=[types.SimpleNamespace(RelatingType=sink_type)])
        result = self.query(Model([terminal, sink_type], "IFC2X3"), "sinks")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer, 1)

    def test_missing_sanitary_type_abstains(self):
        result = self.query(Model([Entity(1, "IfcSanitaryTerminal", "unknown", PredefinedType="NOTDEFINED")]), "sinks")
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.answer)
        self.assertEqual(result.proposed_answer, 0)

    def test_no_sanitary_inventory_is_not_zero(self):
        result = self.query(Model([]), "toilets")
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.answer)

    def test_missing_guid_abstains(self):
        result = self.query(Model([Entity(1, "IfcDoor")]), "doors")
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.proposed_answer, 1)

    def test_explicit_space_labels_and_conflicting_labels(self):
        space = Entity(1, "IfcSpace", "bedroom", Name="MH1")
        other = Entity(2, "IfcSpace", "storage", LongName="Storage")
        result = self.query(Model([space, other]), "bedrooms")
        self.assertEqual(result.answer, 1)
        ambiguous = Entity(3, "IfcSpace", "unknown", Name="MH", LongName="Storage")
        self.assertIsNone(_classify_space(ambiguous)[0])
        result = self.query(Model([space, ambiguous]), "bedrooms")
        self.assertEqual(result.status, "needs_review")

    def test_unrecognized_space_names_do_not_invent_zero(self):
        result = self.query(Model([Entity(1, "IfcSpace", "space", Name="R101")]), "bedrooms")
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.answer)

    def test_no_ground_truth_added_when_not_supplied(self):
        result = self.query(Model([]), "doors")
        self.assertFalse(result.has_ground_truth)

    def test_missing_ifcopenshell_is_actionable(self):
        with patch("drawingqa.ifc.importlib.import_module", side_effect=ImportError):
            result = answer_ifc(self.path, self.question, intent("doors"))
        self.assertEqual(result.status, "unsupported")
        self.assertIn(".[ifc]", result.error)

    def test_parse_failure_is_clean(self):
        fake = types.SimpleNamespace(open=lambda path: (_ for _ in ()).throw(RuntimeError("private source bytes")))
        with patch.dict(sys.modules, {"ifcopenshell": fake}):
            result = answer_ifc(self.path, self.question, intent("doors"))
        self.assertEqual(result.status, "error")
        self.assertNotIn("private source bytes", result.error)

    def test_non_count_and_page_queries_are_unsupported(self):
        result = answer_ifc(self.path, self.question, Intent(operation="general", category="general", canonical_question="height", rule=""))
        self.assertEqual(result.status, "unsupported")
        page_question = self.question.model_copy(update={"page": 2})
        result = answer_ifc(self.path, page_question, intent("doors"))
        self.assertEqual(result.status, "unsupported")


if __name__ == "__main__":
    unittest.main()

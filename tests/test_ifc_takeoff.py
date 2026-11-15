from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from drawingqa.ifc_takeoff import TakeoffQuery, answer_takeoff, parse_takeoff, sibling_ifc
from drawingqa.schemas import Question

HAS_IFC = importlib.util.find_spec("ifcopenshell") is not None


class TakeoffParsingTests(unittest.TestCase):
    def test_storey_counts_in_several_languages(self):
        expected = TakeoffQuery("count", "doors", storey="Level 1")
        for text in [
            "How many doors are shown on the floor plan of 'Level 1'?",
            "Wie viele Türen sind im Grundriss „Level 1“ dargestellt?",
            "Hvor mange døre er vist på plantegningen for 'Level 1'?",
            "¿Cuál es el número de puertas en el plano de la planta «Level 1»?",
            "Mitu ust on näidatud korruse „Level 1“ plaanil?",
            "Ile drzwi pokazano na rzucie kondygnacji „Level 1”?",
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_takeoff(text, "count"), expected)

    def test_model_scope_and_quantities(self):
        self.assertEqual(parse_takeoff("Wie viele Träger enthält das Modell „Haus“?", "count"),
                         TakeoffQuery("count", "beams", scope="model"))
        self.assertEqual(parse_takeoff("What is the total length of all walls on 'EG', in metres?", "m"),
                         TakeoffQuery("sum", "walls", storey="EG", quantity="Length"))
        self.assertEqual(parse_takeoff("Hvad er det samlede nettovolumen af alle dæk og plader på 'EG' i m³?", "m3"),
                         TakeoffQuery("sum", "slabs", storey="EG", quantity="NetVolume"))
        self.assertEqual(parse_takeoff("What is the total net floor area of all rooms (spaces) on 'EG', in m²?", "m2"),
                         TakeoffQuery("storey_area", "rooms", storey="EG"))

    def test_room_names_may_contain_apostrophes(self):
        self.assertEqual(
            parse_takeoff("What is the net floor area of the room 'IMMUNIZ'N ROOM (1D40)' on 'First Floor', in m²?", "m2"),
            TakeoffQuery("room_area", "rooms", room="IMMUNIZ'N ROOM (1D40)", storey="First Floor"))

    def test_unquoted_questions_are_left_to_other_routes(self):
        self.assertIsNone(parse_takeoff("How many doors are shown on this floor plan?", "count"))

    def test_plans_pair_with_their_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "house.ifc").write_text("")
            self.assertEqual(sibling_ifc(root / "house_plan-02-level-1.pdf"), root / "house.ifc")
            self.assertIsNone(sibling_ifc(root / "other_plan-01-ground.pdf"))
            self.assertIsNone(sibling_ifc(root / "house.pdf"))


@unittest.skipUnless(HAS_IFC, "ifcopenshell is optional")
class TakeoffAnswerTests(unittest.TestCase):
    def setUp(self):
        import ifcopenshell
        import ifcopenshell.api
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "house.ifc"
        run = ifcopenshell.api.run
        model = ifcopenshell.file(schema="IFC4")
        project = run("root.create_entity", model, ifc_class="IfcProject", name="House")
        run("unit.assign_unit", model)
        site = run("root.create_entity", model, ifc_class="IfcSite")
        building = run("root.create_entity", model, ifc_class="IfcBuilding")
        storeys = [run("root.create_entity", model, ifc_class="IfcBuildingStorey", name=n) for n in ("EG", "OG")]
        run("aggregate.assign_object", model, products=[site], relating_object=project)
        run("aggregate.assign_object", model, products=[building], relating_object=site)
        run("aggregate.assign_object", model, products=storeys, relating_object=building)
        # The default project length unit is millimetres; answers are reported in metres.
        for storey, lengths in zip(storeys, ([4000.0, 3000.0], [5000.0])):
            for length in lengths:
                wall = run("root.create_entity", model, ifc_class="IfcWall")
                run("spatial.assign_container", model, products=[wall], relating_structure=storey)
                qto = run("pset.add_qto", model, product=wall, name="Qto_WallBaseQuantities")
                run("pset.edit_qto", model, qto=qto, properties={"Length": length})
        door = run("root.create_entity", model, ifc_class="IfcDoor")
        run("spatial.assign_container", model, products=[door], relating_structure=storeys[0])
        model.write(str(self.path))

    def tearDown(self):
        self.tmp.cleanup()

    def ask(self, text, unit):
        return answer_takeoff(self.path, self.path, Question(path="house.ifc", question=text, unit=unit),
                              parse_takeoff(text, unit))

    def test_counts_and_sums_follow_storey_containment(self):
        walls = self.ask("How many walls are shown on the floor plan of 'EG'?", "count")
        self.assertEqual((walls.status, walls.answer), ("answered", 2))
        self.assertTrue(all(e.provenance == "ifc_entity" and e.entity_guid for e in walls.evidence))
        self.assertEqual(self.ask("How many walls are there in the model 'House'?", "count").answer, 3)
        self.assertEqual(self.ask("How many doors are shown on the floor plan of 'OG'?", "count").answer, 0)
        self.assertEqual(self.ask("What is the total length of all walls on 'EG', in metres?", "m").answer, 7.0)

    def test_unknown_storey_is_an_abstention(self):
        missing = self.ask("How many walls are shown on the floor plan of 'Attic'?", "count")
        self.assertEqual(missing.status, "needs_review")
        self.assertIsNone(missing.answer)


if __name__ == "__main__":
    unittest.main()

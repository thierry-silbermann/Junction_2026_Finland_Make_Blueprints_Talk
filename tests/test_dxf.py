from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from drawingqa.dxf import DxfQuery, answer_dxf, parse_query, sibling_dxf
from drawingqa.evaluation import answer_matches
from drawingqa.schemas import Question

HAS_EZDXF = importlib.util.find_spec("ezdxf") is not None


class QueryParsingTests(unittest.TestCase):
    def test_four_languages_map_to_the_same_query(self):
        for text in [
            "How many doors (interior doors and entrance doors together) are drawn on this floor plan?",
            "Wie viele Türen (Innentüren und Eingangstüren zusammen) sind auf diesem Grundriss eingezeichnet?",
            "Combien de portes (portes intérieures et portes d'entrée confondues) sont dessinées sur ce plan ?",
            "Quante porte (porte interne e porte d'ingresso insieme) sono disegnate in questa pianta?",
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_query(text), DxfQuery("count_doors"))

    def test_named_rooms_and_areas_keep_their_scope(self):
        self.assertEqual(parse_query("How many rooms labelled 'Küche' are there on this floor?"),
                         DxfQuery("count_named", name="Küche"))
        self.assertEqual(parse_query("Combien de pièces portant la désignation «Cage d'escalier» y a-t-il à cet étage ?"),
                         DxfQuery("count_named", name="Cage d'escalier"))
        self.assertEqual(parse_query("What is the net floor area of room 2.10 ('Zimmer') in m²?"),
                         DxfQuery("area_room", name="Zimmer", number="2.10"))
        self.assertEqual(parse_query("What is the total net area in m² of all rooms labelled 'Zimmer' on this floor?"),
                         DxfQuery("area_named", name="Zimmer"))
        self.assertEqual(parse_query("Wie gross ist die Nettowohnfläche von Wohnung B in m² (Summe aller Räume)?"),
                         DxfQuery("area_apartment", letter="B"))

    def test_unrelated_questions_are_not_routed(self):
        self.assertIsNone(parse_query("What is the ceiling height?"))
        self.assertIsNone(parse_query("How many bedrooms (makuuhuone, usually marked MH) are on this floor plan?"))

    def test_measurements_are_scored_within_two_percent(self):
        self.assertTrue(answer_matches(101.9, 100, "m2"))
        self.assertFalse(answer_matches(102.1, 100, "m2"))
        self.assertFalse(answer_matches(4, 5, "count"))


@unittest.skipUnless(HAS_EZDXF, "ezdxf is optional")
class DxfAdapterTests(unittest.TestCase):
    def setUp(self):
        import ezdxf
        import fitz
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.pdf, dxf_path = root / "plan.pdf", root / "plan.dxf"
        doc = ezdxf.new()
        msp = doc.modelspace()
        for name in ("DOOR", "WINDOW", "FIXTURE", "ROOM_OUTLINE", "ROOM_TEXT", "UNIT_OUTLINE"):
            doc.layers.add(name)
        for x in (0, 3):
            msp.add_lwpolyline([(x, 0), (x + 0.1, 0), (x + 0.1, 0.9), (x, 0.9), (x, 0)], dxfattribs={"layer": "DOOR"})
            msp.add_lwpolyline([(x + 0.1, 0.9), (x + 1, 0.9)], dxfattribs={"layer": "DOOR"})
            msp.add_arc((x + 0.1, 0), 0.9, 0, 90, dxfattribs={"layer": "DOOR"})
        msp.add_lwpolyline([(0, 5), (1.5, 5), (1.5, 5.3), (0, 5.3)], close=True, dxfattribs={"layer": "WINDOW"})
        for y in (5.05, 5.15):
            msp.add_lwpolyline([(0.05, y), (1.45, y), (1.45, y + 0.05), (0.05, y + 0.05)], close=True, dxfattribs={"layer": "WINDOW"})

        def ellipse(cx, cy, rx, ry):
            return [(cx + rx * math.cos(2 * math.pi * i / 47), cy + ry * math.sin(2 * math.pi * i / 47)) for i in range(48)]
        msp.add_lwpolyline(ellipse(1, 2, 0.18, 0.2), dxfattribs={"layer": "FIXTURE"})
        msp.add_lwpolyline([(0.8, 2.2), (1.2, 2.2)], dxfattribs={"layer": "FIXTURE"})
        msp.add_lwpolyline(ellipse(3, 2, 0.26, 0.14), dxfattribs={"layer": "FIXTURE"})
        rooms = [("Bad", "1.01", [(0, 1), (4, 1), (4, 3), (0, 3)]), ("Zimmer", "1.02", [(0, 3), (4, 3), (4, 6), (0, 6)])]
        for name, number, outline in rooms:
            msp.add_lwpolyline(outline, close=True, dxfattribs={"layer": "ROOM_OUTLINE"})
            cx = sum(x for x, _ in outline) / 4
            cy = sum(y for _, y in outline) / 4
            for offset, text in ((0.5, name), (0, number), (-0.5, "1 m²")):
                entity = msp.add_text(text, dxfattribs={"layer": "ROOM_TEXT", "height": 0.25})
                entity.set_placement((cx, cy + offset), align=ezdxf.enums.TextEntityAlignment.BOTTOM_CENTER)
        entity = msp.add_text("Whg. A", dxfattribs={"layer": "ROOM_TEXT", "height": 0.25})
        entity.set_placement((3, 5.5), align=ezdxf.enums.TextEntityAlignment.BOTTOM_CENTER)
        msp.add_lwpolyline([(0, 1), (4, 1), (4, 6), (0, 6)], close=True, dxfattribs={"layer": "UNIT_OUTLINE"})
        doc.saveas(dxf_path)
        pdf = fitz.open()
        page = pdf.new_page(width=200, height=200)
        for text, x, y in (("Bad", 2, 2.5), ("1.01", 2, 2), ("Zimmer", 2, 5), ("1.02", 2, 4.5), ("Whg. A", 3, 5.5)):
            page.insert_text((20 + 20 * x, 180 - 20 * y), text, fontsize=5)
        pdf.save(self.pdf)
        self.dxf = sibling_dxf(self.pdf)

    def tearDown(self):
        self.tmp.cleanup()

    def ask(self, text, unit="count"):
        question = Question(path="plan.pdf", question=text, unit=unit)
        return answer_dxf(self.pdf, self.dxf, question, parse_query(text), "test")

    def test_counts_come_from_semantic_layers_with_boxes(self):
        doors = self.ask("How many doors (interior doors and entrance doors together) are drawn on this floor plan?")
        self.assertEqual((doors.status, doors.answer), ("answered", 2))
        self.assertTrue(all(e.provenance == "dxf_entity" and e.bbox is not None for e in doors.evidence))
        self.assertEqual(self.ask("How many windows are drawn on this floor plan?").answer, 1)
        self.assertEqual(self.ask("How many toilets (WC) are drawn on this floor plan?").answer, 1)
        self.assertEqual(self.ask("How many washbasins are drawn on this floor plan?").answer, 1)
        self.assertEqual(self.ask("How many apartments (residential units) are shown on this floor plan?").answer, 1)
        self.assertEqual(self.ask("How many rooms labelled 'Zimmer' are there on this floor?").answer, 1)

    def test_areas_use_room_outlines_and_flag_printed_disagreement(self):
        room = self.ask("What is the net floor area of room 1.02 ('Zimmer') in m²?", "m2")
        self.assertEqual(room.proposed_answer, 12.0)
        self.assertEqual(room.status, "needs_review")
        self.assertTrue(any("differs from the printed area" in w for w in room.warnings))

    def test_missing_room_is_an_abstention(self):
        missing = self.ask("What is the net floor area of room 9.99 ('Zimmer') in m²?", "m2")
        self.assertEqual(missing.status, "needs_review")
        self.assertIsNone(missing.answer)


if __name__ == "__main__":
    unittest.main()

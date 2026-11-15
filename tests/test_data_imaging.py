from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from drawingqa.data import DatasetIndex, load_questions, sha256_file, stable_id
from drawingqa.imaging import make_views, render_pages
from drawingqa.schemas import PipelineConfig, Question


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        (self.data / "one").mkdir(parents=True)
        (self.data / "two").mkdir(parents=True)
        (self.data / "one" / "a.png").write_bytes(b"a")

    def tearDown(self):
        self.tmp.cleanup()

    def test_relative_windows_prefixed_and_unique_basename_paths(self):
        index = DatasetIndex(self.data)
        target = (self.data / "one" / "a.png").resolve()
        for reference in ["one/a.png", "one\\a.png", "data/one/a.png", "a.png"]:
            self.assertEqual(index.resolve(reference), target)

    def test_ambiguous_basename_fails_instead_of_guessing(self):
        (self.data / "two" / "a.png").write_bytes(b"b")
        index = DatasetIndex(self.data)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            index.resolve("a.png")
        self.assertEqual(index.resolve("one/a.png"), (self.data / "one" / "a.png").resolve())

    def test_absolute_drive_and_parent_traversal_paths_rejected(self):
        index = DatasetIndex(self.data)
        for reference in ["../secret", "/etc/passwd", "C:\\secret", "one/../../secret", ""]:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                index.resolve(reference)

    def test_wrong_explicit_folder_does_not_fallback_to_basename(self):
        with self.assertRaises(FileNotFoundError):
            DatasetIndex(self.data).resolve("wrong/a.png")

    def test_symlink_escaping_root_rejected(self):
        outside = self.root / "private.png"
        outside.write_bytes(b"outside")
        (self.data / "escape.png").symlink_to(outside)
        with self.assertRaises(ValueError):
            DatasetIndex(self.data).resolve("escape.png")

    def test_indexed_basename_rechecks_containment_after_symlink_replacement(self):
        index = DatasetIndex(self.data)
        file = self.data / "one" / "a.png"
        file.unlink()
        outside = self.root / "private.png"
        outside.write_bytes(b"outside")
        file.symlink_to(outside)
        with self.assertRaises(ValueError):
            index.resolve("a.png")

    def test_question_ids_exclude_answers_and_include_page_scope(self):
        a = Question(path="one/a.png", question="How many windows?", answer=1)
        b = a.model_copy(update={"answer": 999})
        self.assertEqual(stable_id(a), stable_id(b))
        self.assertNotEqual(stable_id(a), stable_id(a.model_copy(update={"page": 2})))

    def test_jsonl_duplicate_ids_and_invalid_rows_fail_early(self):
        file = self.root / "questions.jsonl"
        q = {"path": "one/a.png", "question": "How many windows?"}
        file.write_text(json.dumps(q) + "\n" + json.dumps(q), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_questions(file)
        file.write_text('{"path":"a.png"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid question"):
            load_questions(file)


class ImagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.settings = PipelineConfig(tile_size=256, max_image_side=512, max_views=50)

    def tearDown(self):
        self.tmp.cleanup()

    def render(self, source, **kwargs):
        return render_pages(source, self.root / "pages", sha256_file(source), self.settings, **kwargs)

    def test_transparency_is_composited_on_white(self):
        source = self.root / "transparent.png"
        Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(source)
        pages = self.render(source)
        with Image.open(pages[0].path) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_tiff_frame_selection_and_page_limit(self):
        source = self.root / "frames.tiff"
        frames = [Image.new("RGB", (20, 30), color) for color in ["red", "blue"]]
        frames[0].save(source, save_all=True, append_images=frames[1:])
        pages = self.render(source, selected_page=2)
        self.assertEqual([page.number for page in pages], [2])
        with Image.open(pages[0].path) as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
        self.settings.max_pages = 1
        with self.assertRaisesRegex(ValueError, "select page"):
            self.render(source)
        with self.assertRaisesRegex(ValueError, "document has"):
            self.render(source, selected_page=3)

    def test_pdf_page_numbers_preserved_and_general_page_limit(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PDF dependency not installed")
        source = self.root / "drawing.pdf"
        with fitz.open() as document:
            for title in ["Page one", "Page two"]:
                page = document.new_page(width=144, height=72)
                page.insert_text((12, 36), title)
            document.save(source)
        pages = self.render(source, selected_page=2)
        self.assertEqual([page.number for page in pages], [2])
        self.assertEqual(pages[0].width, 300)
        self.settings.max_pages = 1
        with self.assertRaises(ValueError):
            self.render(source)

    def test_tiles_cover_page_and_rooms_keep_overview_context(self):
        source = self.root / "large.png"
        Image.new("RGB", (600, 400), "white").save(source)
        pages = self.render(source)
        self.settings.views = "tiles"
        windows = make_views(pages, self.root / "views", self.settings, "windows")
        bedrooms = make_views(pages, self.root / "views", self.settings, "bedrooms")
        self.assertGreater(len(windows), 1)
        self.assertEqual(len(bedrooms), 1)
        tiles = [view for view in windows if "overview" not in view.id]
        self.assertEqual(min(view.region[0] for view in tiles), 0)
        self.assertEqual(max(view.region[2] for view in tiles), 1)
        self.assertEqual(max(view.region[3] for view in tiles), 1)
        self.settings.max_views = 1
        with self.assertRaisesRegex(ValueError, "no partial count"):
            make_views(pages, self.root / "views", self.settings, "windows")


if __name__ == "__main__":
    unittest.main()

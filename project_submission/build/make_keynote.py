"""Build an editable copy of the pitch deck for Keynote: PlanSight_pitch_deck.pptx, then .key.

Every title, bullet, number and caption is a native text box, so it can be edited in Keynote
or PowerPoint. Install the fonts in build/fonts first (double-click each .ttf) or Keynote
substitutes them. The .key conversion drives Keynote with AppleScript:
    .venv/bin/pip install python-pptx
    .venv/bin/python project_submission/build/make_keynote.py [--no-key]
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

BUILD = Path(__file__).resolve().parent
OUT = BUILD.parent
ASSETS = OUT / "assets"
PPTX = OUT / "PlanSight_pitch_deck.pptx"
KEY = OUT / "PlanSight_pitch_deck.key"

SERIF, SANS, MONO = "Gelasio", "Hanken Grotesk", "Menlo"
INK, FOREST, FOREST2, MUTED = "011e01", "152e14", "2c412b", "617061"
PAPER, LINE, WHITE, MOSS, NV = "f2f3f5", "dcdcdc", "ffffff", "a3c4a0", "76b900"
GREEN_BG, AMBER_BG, AMBER = "dfe8dc", "fbecc8", "8a5a00"
PAD = 110


def px(n: float) -> Emu:
    """Deck coordinates are 1920x1080 'pixels', as in deck.html."""
    return Emu(int(n * 6350))


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


def runs_of(text: str):
    """Tiny markup: **bold**, *italic*, `code`."""
    for part in re.split(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)", text):
        if not part:
            continue
        if part.startswith("**"):
            yield part[2:-2], {"bold": True}
        elif part.startswith("`"):
            yield part[1:-1], {"font": MONO}
        elif part.startswith("*"):
            yield part[1:-1], {"italic": True}
        else:
            yield part, {}


def text(slide, x, y, w, h, paragraphs, size, *, font=SANS, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.2, after=0, bullet=False):
    box = slide.shapes.add_textbox(px(x), px(y), px(w), px(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate([paragraphs] if isinstance(paragraphs, str) else paragraphs):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment, p.line_spacing, p.space_after = align, spacing, Pt(after)
        for chunk, style in runs_of(("•  " if bullet else "") + line):
            run = p.add_run()
            run.text = chunk
            run.font.size = Pt(size * 0.5)
            run.font.name = style.get("font", font)
            run.font.bold = style.get("bold", bold)
            run.font.italic = style.get("italic", False)
            run.font.color.rgb = rgb(color)
    return box


def rect(slide, x, y, w, h, fill, *, line=None, radius=18, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    item = slide.shapes.add_shape(shape, px(x), px(y), px(w), px(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        item.adjustments[0] = min(0.5, radius / min(w, h))
    item.fill.solid()
    item.fill.fore_color.rgb = rgb(fill)
    if line:
        item.line.color.rgb = rgb(line)
        item.line.width = Pt(0.75)
    else:
        item.line.fill.background()
    item.shadow.inherit = False
    return item


def pill(slide, x, y, label, fill, color, size=24):
    width = len(label) * size * 0.56 + 44
    shape = rect(slide, x, y, width, size + 22, fill, radius=99)
    frame = shape.text_frame
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = label
    run.font.size, run.font.name, run.font.bold = Pt(size * 0.5), SANS, True
    run.font.color.rgb = rgb(color)
    return x + width + 14


def image(slide, name, x, y, w, h, *, frame=True):
    if frame:
        rect(slide, x, y, w, h, WHITE, line=LINE)
        x, y, w, h = x + 14, y + 14, w - 28, h - 28
    with Image.open(ASSETS / name) as img:
        ratio = img.width / img.height
    fw, fh = (w, w / ratio) if w / h < ratio else (h * ratio, h)
    slide.shapes.add_picture(str(ASSETS / name), px(x + (w - fw) / 2), px(y + (h - fh) / 2), px(fw), px(fh))


def background(slide, dark):
    fill = slide.background.fill
    if dark:
        fill.gradient()
        fill.gradient_angle = 120
        stops = fill.gradient_stops
        stops[0].color.rgb, stops[0].position = rgb(FOREST2), 0
        stops[1].color.rgb, stops[1].position = rgb(INK), 1.0
    else:
        fill.solid()
        fill.fore_color.rgb = rgb(PAPER)


def page(prs, number, kicker, title=None, footer="PlanSight", *, dark=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background(slide, dark)
    if kicker:
        text(slide, PAD, 90, 1700, 34, kicker.upper(), 22, color=MOSS if dark else FOREST2, bold=True)
    if title:
        text(slide, PAD, 130, 1700, 150, title, 62, font=SERIF, color=INK, spacing=1.05)
    text(slide, PAD, 1030, 1500, 30, footer, 20, color="808e80")
    text(slide, 1760, 1030, 50, 30, str(number), 20, color="808e80", align=PP_ALIGN.RIGHT)
    return slide


def card(slide, x, y, w, h, paragraphs, size=25, **kw):
    rect(slide, x, y, w, h, WHITE, line=LINE)
    text(slide, x + 32, y + 26, w - 64, h - 52, paragraphs, size, spacing=1.35, **kw)


def build() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = px(1920), px(1080)
    # 1. Title
    s = page(prs, 1, None, footer="github.com/thierry-silbermann/Junction_2026_Finland_Make_Blueprints_Talk",
             dark=True)
    text(s, PAD, 330, 1700, 34, "JUNCTION 2026 · STOA CHALLENGE: MAKE BLUEPRINTS TALK", 22, color=MOSS, bold=True)
    text(s, PAD, 375, 1700, 180, "PlanSight", 132, font=SERIF, color=PAPER, spacing=1.0)
    text(s, PAD, 575, 1700, 150, ["Ask any blueprint a question, in any language.",
                                  "Every answer in your drawings, in plain sight."], 54, color=PAPER, spacing=1.25)
    x = pill(s, PAD, 780, "Powered by NVIDIA Nemotron 3 Nano Omni", NV, INK)
    pill(s, x, 780, "Built for Stoa", PAPER, INK)

    # 2. Problem
    s = page(prs, 2, "The problem", "Engineers lose hours in drawings. Generic AI gives numbers nobody can check.",
             "Stoa figure: design-partner interviews, July 2026 (stoabuild.com)")
    text(s, PAD, 310, 700, 480, [
        "Stoa measured **a week back per project engineer**, mostly time spent finding where something was written down.",
        "Questions like *\"How many doors on this floor?\"* or *\"Net area of bedroom 003?\"* hide in symbols, not text.",
        "Vision models miscount dense plans, and a bare number is **unverifiable**."], 32, bullet=True, after=12)
    text(s, PAD, 830, 700, 150, "Right: Nemotron's answer \"6 doors\" happens to be correct, but its boxes form an "
         "invented column. PlanSight's guard flags it as **needs review**.", 26, color=MUTED)
    image(s, "app_problem.png", 860, 300, 950, 680, frame=False)

    # 3. Idea
    s = page(prs, 3, "The idea", "Let the blueprint talk in its own language first.")
    rect(s, PAD, 290, 1700, 150, WHITE, line=LINE, radius=16)
    rect(s, PAD, 290, 10, 150, FOREST2, shape=MSO_SHAPE.RECTANGLE)
    text(s, PAD + 44, 290, 1620, 150, ["Stoa answers with **the document and page behind every sentence**.",
                                       "PlanSight adds **the symbol behind every number**."], 34, anchor=MSO_ANCHOR.MIDDLE)
    for i, (head, body) in enumerate([
            ("Most PDFs are exports.", "CAD and BIM files sit next to them in M-Files, SharePoint and Dalux. "
             "If we have the source, we read it, exactly and for free."),
            ("Scans need vision.", "NVIDIA Nemotron Nano Omni returns *evidence*, not answers: a JSON inventory "
             "of boxes that Python validates and counts."),
            ("Doubt is a feature.", "When the evidence looks wrong, PlanSight says *needs review* and shows its "
             "proposal, instead of a confident wrong number.")]):
        x = PAD + i * 587
        text(s, x, 520, 527, 60, head, 44, font=SERIF)
        text(s, x, 600, 527, 380, body, 36, color=MUTED, spacing=1.35)

    # 4. Architecture
    s = page(prs, 4, "How it works", "A router picks the most reliable source for every question.",
             "Questions in English, Finnish, German, French, Italian, Estonian and more")
    boxes = [("1 · BIM", GREEN_BG, "IFC take-off", "Plan rendered from an IFC model, or the .ifc itself",
              "Storey/room/model scope, unit-scaled Qto sums, room areas, GlobalIds as evidence", "0 tokens · 100%"),
             ("2 · CAD", GREEN_BG, "DXF layers", "PDF with a same-named DXF",
              "Entities counted per layer and mapped back onto the PDF page as numbered boxes", "0 tokens · 99%"),
             ("3 · Vision", "ecf5d9", "Nemotron Nano Omni", "Scans and raster plans",
              "Strict JSON evidence (0..1000 boxes), Python counting, de-duplication, invented-pattern guard, "
              "abstention", "~1k tokens / question")]
    for i, (tag, tag_bg, head, what, how, cost) in enumerate(boxes):
        x = PAD + i * 600
        rect(s, x, 290, 500, 440, WHITE, line=LINE)
        pill(s, x + 30, 320, tag, tag_bg, FOREST)
        text(s, x + 30, 390, 440, 60, head, 42, font=SERIF)
        text(s, x + 30, 455, 440, 90, what, 28)
        text(s, x + 30, 545, 440, 120, how, 26, color=MUTED, spacing=1.3)
        text(s, x + 30, 670, 440, 40, f"**{cost}**", 28)
        if i < 2:
            text(s, x + 500, 460, 100, 80, "›", 60, color="adb7ad", align=PP_ALIGN.CENTER)
    card(s, PAD, 770, 1700, 190, "Every route returns the same `Prediction`: answer · status (*answered* / *needs "
         "review*) · evidence with provenance (`dxf_entity`, `ifc_entity`, `model_prediction`) · justification · "
         "tokens and latency. All of it is drawn live in the PlanSight app and exported to a review report.")

    # 5. CAD and BIM proof
    s = page(prs, 5, "Reasoning you can verify · CAD and BIM", "\"How many doors on this floor?\" → 36, and here are all 36.",
             "Swiss Dwellings CC BY 4.0 · bimspeed Warmond IFC model")
    image(s, "swiss_doors_dxf.png", PAD, 290, 1010, 590)
    text(s, PAD, 895, 1010, 90, "Held-out Swiss plan. Every box is a DXF door entity mapped onto the PDF; numbers "
         "match the review list. Expected 36, answered 36.", 24, color=MUTED)
    text(s, 1180, 290, 630, 50, "IFC: no guessing, full provenance", 34, font=SERIF)
    card(s, 1180, 355, 630, 440, [
        "**Q:** Net floor area of the room 'Bedroom (414B-5-003-001)' on '3. Level', in m²?",
        "**A: 10.71 m²** · answered",
        "Evidence: `IfcSpace 414B-5-003-001`", "STEP #5838 · GlobalId `2J_oFeqlHDNPq7R3anXaug`",
        "Stored net area 10.71 m²"], size=24, after=10)
    text(s, 1180, 820, 630, 120, "A reviewer can open the model at that GlobalId and check in seconds. "
         "Route: `ifc-takeoff-v1`.", 26, color=MUTED)

    # 6. Live app
    s = page(prs, 6, "The live app · NVIDIA Nemotron on any blueprint",
             "Drop in a drawing. Detect everything, or ask in your own language.",
             "CubiCasa5K, CC BY-NC-SA 4.0 · scan from finna.fi · model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning "
             "(thinking off, top_k 1)")
    for i, (name, tag, caption) in enumerate([
            ("app_live_detect.png", "detect all", "A Finnish archive scan from the Finna collection: windows, rooms, "
             "stairs and fixtures boxed in one click."),
            ("app_live_answer.png", "answered", "Asked in Finnish: \"how many sinks and wash basins?\" → 2, both "
             "boxed, matches the dataset.")]):
        x = PAD + i * 880
        image(s, name, x, 290, 820, 530, frame=False)
        pill(s, x, 845, tag, GREEN_BG, FOREST)
        text(s, x, 900, 820, 100, caption, 24, color=MUTED)

    # 7. Detector
    s = page(prs, 7, "Creativity · what next year's models won't make obsolete", "CAD files are free training labels.",
             "In training; not yet wired into the answer path")
    image(s, "detector_swiss_heldout.png", PAD, 290, 1000, 560)
    image(s, "detector_cubicasa.png", 1170, 290, 640, 560)
    text(s, PAD, 865, 1000, 40, "Held-out Swiss site: 37 doors found (DXF says 36), plus windows, WCs, basins, "
         "bathtubs.", 24, color=MUTED)
    text(s, 1170, 865, 640, 70, "Finnish CubiCasa scan, never seen in training: the boxes land on the real doors.",
         24, color=MUTED)
    text(s, PAD, 945, 1700, 70, "YOLO11n, 2.6M parameters · 61k boxes generated from 526 DXFs, zero hand labelling · "
         "**mAP@50 0.89** on held-out sites · every CAD file Stoa connects makes it better.", 25)

    # 8. Results
    s = page(prs, 8, "Accuracy", "Near-perfect where the source exists. Honest everywhere else.",
             "Next: detector on CubiCasa doors, windows and fixtures (the 0% categories today)")
    rows = [("Dataset", "Questions (full set)", "Accuracy", "Route"),
            ("Swiss Dwellings", "37,584", "99.1%", "DXF · 99.0% on 18,528 held-out questions"),
            ("IFC models", "3,062", "100%", "IFC take-off · 3,062 / 3,062, 9 languages"),
            ("CubiCasa5K scans", "85,514", "43% (55% of answered)", "Nemotron Nano Omni"),
            ("Opening Design sets", "64", "12.5%", "Nemotron Nano Omni")]
    table = s.shapes.add_table(len(rows), 4, px(PAD), px(290), px(1700), px(400)).table
    for i, w in enumerate((360, 300, 440, 600)):
        table.columns[i].width = px(w)
    for r, row in enumerate(rows):
        table.rows[r].height = px(56 if r == 0 else 86)
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(WHITE)
            cell.margin_left = cell.margin_right = px(24)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = value.upper() if r == 0 else value
            header, number = r == 0, c == 2 and r > 0
            run.font.size = Pt((20 if header else 34 if number else 27) * 0.5)
            run.font.name = SERIF if number else SANS
            run.font.bold = header
            run.font.color.rgb = rgb(MUTED if header else FOREST2 if number and r < 3 else INK)
    text(s, PAD, 730, 1700, 80, "Dataset scoring rules: exact counts, ±2% on areas, lengths and volumes. Sample of "
         "805 questions, 6 whole drawings per folder, Swiss from held-out sites only. IFC-Bench free-text answers are "
         "not gradable by exact match.", 24, color=MUTED)
    card(s, PAD, 840, 820, 140, "**CubiCasa, mostly right:** floors, sinks, saunas, bedrooms, balconies.")
    card(s, PAD + 880, 840, 820, 140, "**CubiCasa, still 0%:** doors, windows, showers/bathtubs, WC rooms. These "
         "are exactly the symbols the detector learns.")

    # 9. Cost
    s = page(prs, 9, "Low inference cost", "The cheapest token is the one you never send.",
             "Token figures from measured requests: ~2.5k input + 0.7k output per vision request")
    for i, (big, body) in enumerate([
            ("0", "tokens for ~1/3 of all questions (CAD/BIM routes), ~30 ms each on a laptop CPU"),
            ("3B", "active parameters: Nemotron 3 Nano Omni is a 30B mixture-of-experts"),
            ("~1k", "tokens per question on average over all 127k questions"),
            ("0.5", "requests per vision question: one cached inventory answers EN and FI")]):
        x = PAD + i * 440
        rect(s, x, 290, 380, 6, FOREST2, shape=MSO_SHAPE.RECTANGLE)
        text(s, x, 315, 380, 130, big, 110, font=SERIF, spacing=1.0)
        text(s, x, 455, 380, 200, body, 28, spacing=1.3)
    card(s, PAD, 700, 820, 260, "**Open weights → self-host in the EU.** The same config runs against NVIDIA's "
         "hosted API or a single-GPU vLLM server (`Nemotron-3-Nano-Omni-30B-A3B`), matching Stoa's \"held in the EU, "
         "never trained on\".")
    card(s, PAD + 880, 700, 820, 260, "**Every run is metered.** Tokens, retries, cache hits and API latency are "
         "logged per answer; runs resume after rate limits without paying twice.")

    # 10. Judging
    s = page(prs, 10, "Judging criteria", "Built for all four, at 25% each.")
    for i, (head, body) in enumerate([
            ("Accuracy", "99–100% on CAD/BIM-backed drawings; abstains instead of guessing on scans; measured with "
             "the dataset's own scoring rules."),
            ("Visualized reasoning", "Numbered boxes on the real page, IFC GlobalIds, route and one-line reason per "
             "answer; a live app to detect or ask on any drawing, and a benchmark tab to replay every question."),
            ("Creativity", "The blueprint's own source as ground truth; CAD files as free detector labels; a guard "
             "against invented evidence."),
            ("Low inference cost", "0 tokens for a third of questions; 3B-active Nemotron for the rest; "
             "language-shared cache; open weights.")]):
        x, y = PAD + (i % 2) * 868, 290 + (i // 2) * 360
        rect(s, x, y, 832, 330, WHITE, line=LINE)
        rect(s, x, y, 832, 8, FOREST2, shape=MSO_SHAPE.RECTANGLE)
        text(s, x + 32, y + 36, 768, 60, head, 48, font=SERIF)
        text(s, x + 32, y + 110, 768, 200, body, 30, spacing=1.35)

    # 11. Stoa fit and roadmap
    s = page(prs, 11, "Why Stoa · what's next", "A Stoa skill: \"count and prove\".")
    for i, (head, items) in enumerate([
            ("Fits how Stoa already works", ["Stoa syncs M-Files, SharePoint, Dalux: CAD/BIM sit beside the PDFs.",
                                            "Stoa opens the drawing next to the answer; PlanSight adds the boxes.",
                                            "Self-hosted Nemotron keeps drawings in the EU."]),
            ("Next two weeks", ["Wire the detector into the vision path for doors, windows and fixtures.",
                                "Read schedules from the PDF text layer (door and finish tables).",
                                "Draw IFC answers on the plan via the SVG sibling.",
                                "Fine-tune on reviewer decisions exported from the report."])]):
        x = PAD + i * 880
        text(s, x, 300, 820, 70, head, 44, font=SERIF)
        text(s, x, 390, 820, 600, items, 36, bullet=True, after=14, spacing=1.3)

    # 12. Close
    s = page(prs, 12, None, footer="Junction 2026", dark=True)
    text(s, PAD, 300, 1700, 180, "PlanSight", 132, font=SERIF, color=PAPER, spacing=1.0)
    text(s, PAD, 495, 1700, 150, ["Ask any blueprint a question, in any language.",
                                  "Every answer in your drawings, in plain sight."], 54, color=PAPER, spacing=1.25)
    text(s, PAD, 700, 1700, 100, ["Code: github.com/thierry-silbermann/Junction_2026_Finland_Make_Blueprints_Talk",
                                  "Live app: `drawingqa serve` → Live blueprint + Benchmark tabs · static report in "
                                  "project_submission/demo"], 30, color="adb7ad", spacing=1.4)
    x = pill(s, PAD, 850, "NVIDIA Nemotron 3 Nano Omni", NV, INK)
    pill(s, x, 850, "Stoa · Make Blueprints Talk", PAPER, INK)

    prs.save(PPTX)
    print(f"{PPTX.name}: {len(prs.slides)} slides")


def _narration():
    """Speaker notes for the slides the video narrates, taken from build.py."""
    try:
        from build import SEGMENTS
    except ImportError:
        return []
    return [(visual, text_) for visual, text_ in SEGMENTS if isinstance(visual, int)]


def to_keynote() -> None:
    """Open the .pptx in Keynote, add the narration as presenter notes, save as .key.

    Keynote refuses to import python-pptx notes slides, so the notes are set here instead."""
    notes = _narration()
    script = '''
    on run argv
        tell application "Keynote"
            set doc to open POSIX file (item 1 of argv)
            repeat with i from 3 to (count of argv) by 2
                set presenter notes of slide ((item i of argv) as integer) of doc to item (i + 1) of argv
            end repeat
            save doc in POSIX file (item 2 of argv)
            close doc saving no
        end tell
    end run'''
    args = [str(PPTX), str(KEY)] + [str(v) for pair in notes for v in pair]
    subprocess.run(["osascript", "-e", script, *args], check=True)
    print(f"{KEY.name}: saved by Keynote, presenter notes on slides {', '.join(str(n) for n, _ in notes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-key", action="store_true", help="Only write the .pptx")
    args = parser.parse_args()
    build()
    if not args.no_key:
        to_keynote()


if __name__ == "__main__":
    main()

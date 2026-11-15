"""Build the pitch deck PDF and a narrated draft video (<= 1:55) from deck.html.

Needs Google Chrome, ffmpeg and macOS `say`. Run from the repo root:
    .venv/bin/python project_submission/build/build.py [--voice Samantha] [--skip-video]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import fitz

BUILD = Path(__file__).resolve().parent
OUT = BUILD.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DECK_PDF = OUT / "PlanProof_pitch_deck.pdf"
VIDEO = OUT / "PlanProof_pitch_draft.mp4"
MAX_SECONDS = 115

# (slide number, narration) in playback order; mirrors VIDEO_SCRIPT.md.
SEGMENTS = [
    (1, "Site engineers lose hours every day hunting through drawings. Stoa answers project questions "
        "with the page behind every sentence. We built PlanProof to give the symbol behind every number."),
    (2, "Ask a vision model how many doors are on this plan, and it may say six, and even be right. "
        "But look at its boxes: a straight column of doors that don't exist. You can't trust a number you can't check."),
    (4, "PlanProof routes every question to the most reliable source. Many drawings ship with their CAD or BIM file. "
        "When we find the DXF or the IFC model, we answer from it directly: zero tokens, deterministic, "
        "every element mapped back onto the sheet. For scans, NVIDIA Nemotron Nano Omni returns an inventory of boxes, "
        "and Python does the counting."),
    (5, "Every answer opens in a review report: the page, numbered boxes, a one-line reason, and which route answered."),
    (6, "Suspicious patterns are flagged as needs review, not guessed. A human verifies in seconds."),
    (8, "On held-out data: ninety-nine percent on Swiss plans, and one hundred percent on more than three thousand "
        "IFC questions in nine languages."),
    (9, "A third of all questions cost zero tokens. The rest run on Nemotron's three billion active parameters, "
        "about a thousand tokens per question, and its open weights can be self-hosted in the EU."),
    (7, "And this is what won't be obsolete next year: we turned CAD files into free training labels. "
        "Our tiny detector already finds the real doors on Finnish scans it has never seen. "
        "Every CAD file Stoa connects makes it better."),
    (12, "PlanProof. Every number, with the proof drawn on it."),
]


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def build_pdf() -> None:
    run(CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
        f"--print-to-pdf={DECK_PDF}", (BUILD / "deck.html").as_uri())
    with fitz.open(DECK_PDF) as doc:
        print(f"{DECK_PDF.name}: {doc.page_count} pages, {doc[0].rect.width:.0f}x{doc[0].rect.height:.0f} pt")


def rasterize(tmp: Path) -> dict[int, Path]:
    slides = {}
    with fitz.open(DECK_PDF) as doc:
        for i, page in enumerate(doc, 1):
            zoom = 1920 / page.rect.width
            path = tmp / f"slide{i:02d}.png"
            page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(path)
            slides[i] = path
    return slides


def duration(path: Path) -> float:
    info = json.loads(run("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)))
    return float(info["format"]["duration"])


def build_video(voice: str, rate: int) -> None:
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        slides = rasterize(tmp)
        parts = []
        for n, (slide, text) in enumerate(SEGMENTS):
            audio = tmp / f"seg{n}.aiff"
            run("say", "-v", voice, "-r", str(rate), "-o", str(audio), text)
            clip = tmp / f"seg{n}.mp4"
            run("ffmpeg", "-y", "-loop", "1", "-i", str(slides[slide]), "-i", str(audio),
                "-af", "apad=pad_dur=0.5", "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", str(clip))
            parts.append(clip)
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
            "-movflags", "+faststart", str(VIDEO))
    seconds = duration(VIDEO)
    print(f"{VIDEO.name}: {seconds:.1f} s")
    if seconds > MAX_SECONDS:
        raise SystemExit(f"Video is {seconds:.1f} s; the limit is 120 s. Raise --rate or shorten SEGMENTS.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--rate", type=int, default=175, help="words per minute for `say`")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    build_pdf()
    if not args.skip_video:
        build_video(args.voice, args.rate)


if __name__ == "__main__":
    main()

"""Build the pitch deck PDF and a narrated draft video (<= 1:55) from deck.html and app footage.

App clips come from record_app.py (build/footage/*.mp4); missing clips fall back to a slide.
Needs Google Chrome, ffmpeg and macOS `say`. Run from the repo root:
    .venv/bin/python project_submission/build/build.py [--voice Samantha] [--skip-video] [--slides-only]
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
FOOTAGE = BUILD / "footage"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DECK_PDF = OUT / "PlanSight_pitch_deck.pdf"
VIDEO = OUT / "PlanSight_pitch_draft.mp4"
MAX_SECONDS = 115

# (visual, narration) in playback order; mirrors VIDEO_SCRIPT.md.
# An int is a deck slide; a string is an app clip recorded by record_app.py.
SEGMENTS = [
    (1, "Site engineers lose hours every day hunting through drawings. Stoa answers with the page behind every "
        "sentence. PlanSight puts the symbol behind every number in plain sight."),
    ("bench_problem", "Ask a vision model how many doors are on this plan. It says six, and is even right. "
                      "But look at its boxes: a straight column of doors that don't exist. "
                      "PlanSight holds that answer back for review, instead of guessing."),
    (4, "So PlanSight routes every question to the most reliable source. When a drawing ships with its CAD or BIM "
        "file, we answer from it directly: zero tokens, deterministic. For scans, NVIDIA Nemotron Nano Omni returns "
        "an inventory of boxes, and Python does the counting."),
    ("live_any", "Drop in any blueprint, even a century-old scan. One click, and every window, room, stair and "
                 "fixture is boxed on the page."),
    ("live_ask", "Ask in English or Finnish. Here, in Finnish: how many sinks and wash basins? Two. Both are boxed, "
                 "a one-line reason explains the count, and it matches the dataset's answer."),
    ("live_dxf", "When the CAD file exists, the answer is exact: thirty-six doors, every one numbered on the sheet, "
                 "zero tokens, in a third of a second."),
    ("bench_results", "Across the benchmark: one hundred percent on IFC models, and ninety-nine percent on Swiss "
                      "plans. Every CAD- or BIM-backed answer is free; the rest run on Nemotron's three billion "
                      "active parameters, with open weights we can self-host in the EU."),
    (7, "And what won't be obsolete next year: CAD files become free training labels. Our tiny detector already "
        "finds real doors on Finnish scans it has never seen."),
    (12, "PlanSight. Every number, with the proof drawn on it."),
]
CLIP_FALLBACK = {"bench_problem": 2, "live_any": 6, "live_ask": 6, "live_dxf": 5, "bench_results": 8}
ENCODE = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
          "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def build_pdf() -> None:
    run(CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
        "--virtual-time-budget=10000", f"--print-to-pdf={DECK_PDF}", (BUILD / "deck.html").as_uri())
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


def render_segment(visual, audio: Path, clip: Path, slides: dict[int, Path], slides_only: bool) -> str:
    speech = duration(audio) + 0.5
    footage = FOOTAGE / f"{visual}.mp4" if isinstance(visual, str) else None
    if footage is not None and not slides_only and footage.exists():
        total = max(speech, duration(footage))
        run("ffmpeg", "-y", "-i", str(footage), "-i", str(audio), "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={total:.2f},scale=1920:1080,setsar=1[v];[1:a]apad[a]",
            "-map", "[v]", "-map", "[a]", "-t", f"{total:.2f}", *ENCODE, str(clip))
        return f"clip {visual}"
    if footage is not None and not slides_only:
        print(f"  {footage.name} missing; using slide {CLIP_FALLBACK[visual]} (run record_app.py)")
    slide = CLIP_FALLBACK[visual] if isinstance(visual, str) else visual
    run("ffmpeg", "-y", "-loop", "1", "-t", f"{speech:.2f}", "-i", str(slides[slide]), "-i", str(audio),
        "-filter_complex", "[0:v]scale=1920:1080,setsar=1[v];[1:a]apad[a]", "-map", "[v]", "-map", "[a]",
        "-t", f"{speech:.2f}", "-tune", "stillimage", *ENCODE, str(clip))
    return f"slide {slide}"


def build_video(voice: str, rate: int, slides_only: bool) -> None:
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        slides = rasterize(tmp)
        parts = []
        for n, (visual, text) in enumerate(SEGMENTS):
            audio = tmp / f"seg{n}.aiff"
            run("say", "-v", voice, "-r", str(rate), "-o", str(audio), text)
            clip = tmp / f"seg{n}.mp4"
            used = render_segment(visual, audio, clip, slides, slides_only)
            print(f"  {n + 1}. {used}: {duration(clip):.1f} s")
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
    parser.add_argument("--slides-only", action="store_true", help="Ignore recorded app footage")
    args = parser.parse_args()
    build_pdf()
    if not args.skip_video:
        build_video(args.voice, args.rate, args.slides_only)


if __name__ == "__main__":
    main()

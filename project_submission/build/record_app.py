"""Record the PlanSight app for the pitch video: one 1920x1080 clip per scene, plus deck stills.

Starts `drawingqa serve`, drives Chrome with Playwright (a drawn cursor shows every click) and
captures frames with Chrome's screencast. Run `scripts/warm_demo.py` first so answers are cached.
Needs Google Chrome, ffmpeg and `pip install playwright`. Run from the repo root:
    .venv/bin/python project_submission/build/record_app.py [--only live_ask]
"""
from __future__ import annotations

import argparse
import base64
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BUILD = Path(__file__).resolve().parent
FOOTAGE = BUILD / "footage"
ASSETS = BUILD.parent / "assets"
REPO = BUILD.parents[1]

SCAN = REPO / "data/finna/hka.139877668752900_168422869617700.jpg"
CUBICASA = "cubicasa5k/high_quality-4017.png"
SWISS = "swiss_dwellings/site1021_bldg2101_floor9477.pdf"
SINKS_FI = ("Kuinka monta pesu- ja tiskiallasta (keittiö ja kylpytilat yhteensä; "
            "kaksoisallas lasketaan yhdeksi) pohjapiirustukseen on piirretty?")
DOORS_EN = "How many doors (interior doors and entrance doors together) are drawn on this floor plan?"

CURSOR = """
window.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.innerHTML = '<svg width="28" height="28" viewBox="0 0 24 24"><path d="M4 2l15 8.5-6.6 1.8L9.6 19z" fill="#fff" stroke="#011e01" stroke-width="1.5" stroke-linejoin="round"/></svg>';
  c.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;transform:translate(1100px,640px);transition:transform .6s cubic-bezier(.3,.7,.2,1);filter:drop-shadow(0 2px 3px #0006)';
  document.body.appendChild(c);
  const style = document.createElement('style');
  style.textContent = '@keyframes pp-pulse{from{transform:scale(.3);opacity:1}to{transform:scale(1.5);opacity:0}}';
  document.head.appendChild(style);
  window.__cursorTo = (x, y) => { c.style.transform = `translate(${x - 4}px,${y - 2}px)`; };
  window.__pulse = (x, y) => {
    const r = document.createElement('div');
    r.style.cssText = `position:fixed;left:${x - 20}px;top:${y - 20}px;width:40px;height:40px;border-radius:50%;border:3px solid #76b900;z-index:2147483646;pointer-events:none;animation:pp-pulse .55s ease-out forwards`;
    document.body.appendChild(r);
    setTimeout(() => r.remove(), 700);
  };
});
"""


class Recorder:
    """Chrome screencast frames, re-timed with their own timestamps into a constant-rate MP4."""

    def __init__(self, page: Page):
        self.page = page
        self.session = page.context.new_cdp_session(page)
        self.session.on("Page.screencastFrame", self._frame)
        self.frames: list[tuple[float, Path]] = []
        self.tmp: Path | None = None

    def _frame(self, params):
        path = self.tmp / f"{len(self.frames):05d}.jpg"
        path.write_bytes(base64.b64decode(params["data"]))
        self.frames.append((params["metadata"]["timestamp"], path))
        self.session.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

    def start(self, tmp: Path):
        self.tmp, self.frames = tmp, []
        self.session.send("Page.startScreencast", {"format": "jpeg", "quality": 92,
                                                    "maxWidth": 1920, "maxHeight": 1080})

    def stop(self, destination: Path):
        self.page.wait_for_timeout(200)
        end = self.page.evaluate("Date.now() / 1000")
        self.session.send("Page.stopScreencast")
        if not self.frames:
            raise RuntimeError("No frames were captured")
        listing = self.tmp / "frames.txt"
        lines = []
        for (stamp, path), following in zip(self.frames, [f[0] for f in self.frames[1:]] + [end]):
            lines.append(f"file '{path}'\nduration {max(following - stamp, 0.001):.4f}\n")
        lines.append(f"file '{self.frames[-1][1]}'\n")
        listing.write_text("".join(lines))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                               "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xf2f3f5,fps=30,format=yuv420p",
                        "-c:v", "libx264", "-crf", "17", "-preset", "slow", str(destination)], check=True)
        print(f"{destination.name}: {end - self.frames[0][0]:.1f} s, {len(self.frames)} frames")


def point(page: Page, selector: str, click: bool = True, settle: int = 450):
    target = page.locator(selector).first
    target.wait_for(state="visible", timeout=30000)
    box = target.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.evaluate("([x, y]) => window.__cursorTo(x, y)", [x, y])
    page.wait_for_timeout(700)
    if click:
        page.evaluate("([x, y]) => window.__pulse(x, y)", [x, y])
        target.click()
    page.wait_for_timeout(settle)


def type_question(page: Page, text: str):
    point(page, "#askInput", settle=150)
    page.locator("#askInput").fill("")
    page.locator("#askInput").press_sequentially(text, delay=18)
    page.wait_for_timeout(350)
    point(page, "#askBtn", settle=100)
    page.wait_for_selector("#answerCard:not([hidden]) .answer-value", timeout=300000)


def wait_app(page: Page, url: str):
    page.goto(url)
    page.wait_for_selector(".sample", timeout=60000)
    page.wait_for_timeout(1200)


def live_any(page: Page, base: str):
    """Upload a century-old Finnish scan, then Detect everything."""
    wait_app(page, base + "/#live")
    yield
    page.wait_for_timeout(900)
    point(page, "#dropzone", click=False)
    page.evaluate("document.getElementById('dropzone').classList.add('over')")
    page.wait_for_timeout(500)
    page.set_input_files("#fileInput", str(SCAN))
    page.evaluate("document.getElementById('dropzone').classList.remove('over')")
    page.wait_for_function("!document.getElementById('detectBtn').disabled", timeout=60000)
    page.wait_for_timeout(900)
    point(page, "#detectBtn", settle=200)
    page.wait_for_selector("#kindChips .chip", timeout=300000)
    page.wait_for_timeout(2600)
    yield "app_live_detect"


def live_ask(page: Page, base: str):
    """CubiCasa plan: inventory, then a Finnish question that matches the dataset answer."""
    yield
    point(page, f'.sample[title="{CUBICASA}"]', settle=300)
    page.wait_for_function("!document.getElementById('detectBtn').disabled", timeout=60000)
    page.wait_for_timeout(500)
    point(page, "#detectBtn", settle=200)
    page.wait_for_selector("#kindChips .chip", timeout=300000)
    page.wait_for_timeout(1400)
    type_question(page, SINKS_FI)
    page.wait_for_timeout(1200)
    point(page, "#answerCard .ev", settle=2200)
    yield "app_live_answer"


def live_dxf(page: Page, base: str):
    """Swiss plan with its DXF: 36 doors, every one boxed, zero tokens."""
    yield
    point(page, f'.sample[title="{SWISS}"]', settle=300)
    page.wait_for_function("!document.getElementById('askInput').disabled", timeout=60000)
    page.wait_for_timeout(400)
    type_question(page, DOORS_EN)
    page.wait_for_timeout(3200)
    yield "app_live_dxf"


def bench_problem(page: Page, base: str):
    """The flagged CubiCasa answer: six boxes in a straight column, held for review."""
    page.goto(base + "/#bench")
    page.wait_for_selector(".bar-row", timeout=60000)
    page.wait_for_timeout(1500)
    yield
    point(page, '.bar-row:has-text("cubicasa5k")', settle=900)
    point(page, f'.gallery .sample[title="{CUBICASA}"]', settle=300)
    page.wait_for_selector(".qtable tbody tr.active", timeout=60000)
    page.wait_for_timeout(1200)
    page.evaluate("window.scrollTo({top: document.getElementById('benchWork').getBoundingClientRect().top"
                  " + scrollY - 12, behavior: 'smooth'})")
    page.wait_for_timeout(900)
    for box in range(1, 4):
        point(page, f"#benchViewer .v-overlay .box:nth-child({box})", click=False, settle=150)
    point(page, ".bench-detail .badge.needs_review", click=False, settle=3000)
    yield "app_problem"


def bench_results(page: Page, base: str):
    """KPI cards and the per-dataset accuracy bars animating in."""
    page.goto("about:blank")
    yield
    page.goto(base + "/#bench")
    page.wait_for_selector(".bar-row", timeout=60000)
    page.wait_for_timeout(2600)
    yield "app_bench"
    point(page, ".kpi.hero", click=False, settle=900)
    point(page, '.bar-row:has-text("swiss_dwellings")', click=False, settle=900)
    point(page, '.bar-row:has-text("ifc_models")', click=False, settle=2500)
    yield


SCENES = {"live_any": live_any, "live_ask": live_ask, "live_dxf": live_dxf,
          "bench_problem": bench_problem, "bench_results": bench_results}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(SCENES))
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    FOOTAGE.mkdir(exist_ok=True)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = subprocess.Popen([sys.executable, "-m", "drawingqa", "serve", "--port", str(port)], cwd=REPO,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/api/runs", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome", headless=not args.headed)
            context = browser.new_context(viewport={"width": 1536, "height": 864}, device_scale_factor=1.25)
            context.add_init_script(CURSOR)
            page = context.new_page()
            recorder = Recorder(page)
            for name in args.only or SCENES:
                steps = SCENES[name](page, base)
                next(steps)
                with tempfile.TemporaryDirectory() as tmp:
                    recorder.start(Path(tmp))
                    for still in steps:
                        if still:
                            page.screenshot(path=ASSETS / f"{still}.png")
                    recorder.stop(FOOTAGE / f"{name}.mp4")
            browser.close()
    finally:
        server.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

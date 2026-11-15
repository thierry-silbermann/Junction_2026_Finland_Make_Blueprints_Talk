"""Pre-run the demo app's samples so a screen recording answers instantly from the cache.

For every Live-tab sample (plus any --path), this runs Detect everything and the suggested
questions through the same endpoints the browser uses. Results land in the extraction
cache, so the app later labels them "cached" rather than "live".

    .venv/bin/python scripts/warm_demo.py --model nemotron_nvidia
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from drawingqa.app.server import create_app
from drawingqa.config import load_config
from drawingqa.schemas import PipelineConfig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run", default="per-folder-6-v2")
    parser.add_argument("--output", type=Path, default=Path("runs/live"))
    parser.add_argument("--model", default="nemotron_nvidia")
    parser.add_argument("--path", action="append", default=[], help="Extra drawing, relative to the data root")
    parser.add_argument("--questions-per-image", type=int, default=4)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.pipeline = PipelineConfig.model_validate({**config.pipeline.model_dump(), "strategy": "per_drawing"})
    questions = args.data_root / "questions.jsonl"
    app = create_app(config, args.data_root, runs_dir=args.runs_dir, output=args.output,
                     questions=questions if questions.is_file() else None, default_run=args.run, probe_models=False)
    client = TestClient(app)
    samples = client.get("/api/samples", params={"run": args.run}).json()
    paths = list(dict.fromkeys([s["path"] for s in samples["samples"]] + args.path))
    failures = 0
    for path in paths:
        started = time.monotonic()
        detected = client.post("/api/detect", json={"path": path, "model": args.model})
        if detected.status_code != 200:
            failures += 1
            print(f"detect failed {path}: {detected.json().get('detail')}", file=sys.stderr)
            continue
        counts = ", ".join(f"{c['total']} {k}" for k, c in detected.json()["counts"].items()) or "nothing"
        own = [q["question"] for q in client.get("/api/image", params={"path": path}).json()["questions"]]
        asked = list(dict.fromkeys(own[:args.questions_per_image] + samples["questions"]))
        answers = []
        for question in asked:
            reply = client.post("/api/ask", json={"path": path, "question": question, "model": args.model})
            if reply.status_code != 200 or reply.json()["status"] == "error":
                failures += 1
                answers.append("error")
            else:
                answers.append(str(reply.json()["answer"] if reply.json()["answer"] is not None else "review"))
        print(f"{path}: {counts} | answers {' '.join(answers)} | {time.monotonic() - started:.1f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

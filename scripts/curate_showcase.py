"""Verify the Live tab's showcase examples and write the ones that pass to the app.

Each candidate in scripts/showcase_candidates.json is asked through the same endpoint the
browser uses. "instant" examples are asked once (and stay cached); "live" examples bypass the
cache and must pass on every repeat, since the app calls the model again on each click.
An "exact" example must match the dataset answer; a "review" example must be held for
review with a proposal that matches it.

    .venv/bin/python scripts/curate_showcase.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from drawingqa.app.server import SHOWCASE, create_app
from drawingqa.config import load_config
from drawingqa.evaluation import answer_matches
from drawingqa.schemas import PipelineConfig


def passes(reply: dict, expect: str) -> bool:
    if reply.get("expected_answer") is None:
        return False
    if expect == "review":
        return reply["status"] == "needs_review" and reply.get("proposed_answer") is not None and \
            answer_matches(reply["proposed_answer"], reply["expected_answer"], reply.get("unit"))
    return reply.get("correct") is True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, default=Path("scripts/showcase_candidates.json"))
    parser.add_argument("--output", type=Path, default=SHOWCASE)
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", default="nemotron_nvidia")
    parser.add_argument("--repeats", type=int, default=3, help="Fresh model calls per live example")
    parser.add_argument("--max", type=int, default=10, help="Keep at most this many examples")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.pipeline = PipelineConfig.model_validate({**config.pipeline.model_dump(), "strategy": "per_drawing"})
    questions = args.data_root / "questions.jsonl"
    app = create_app(config, args.data_root, runs_dir=Path("runs"), output=Path("runs/live"),
                     questions=questions if questions.is_file() else None, probe_models=False, showcase=None)
    client = TestClient(app)
    kept = []
    for candidate in json.loads(args.candidates.read_text(encoding="utf-8")):
        live = candidate["mode"] == "live"
        replies = []
        for _ in range(args.repeats if live else 1):
            response = client.post("/api/ask", json={"path": candidate["path"], "question": candidate["question"],
                                                     "model": args.model, "fresh": live})
            replies.append(response.json() if response.status_code == 200 else {"status": "error", "error": response.text})
        ok = all(passes(r, candidate["expect"]) for r in replies)
        last = replies[-1]
        shown = last.get("answer") if last.get("answer") is not None else last.get("proposed_answer")
        seconds = sum(r.get("elapsed_seconds", 0) for r in replies) / len(replies)
        print(f"{'PASS' if ok else 'FAIL'}  {candidate['mode']:7} {last.get('route_label', '?'):12} "
              f"got {[r.get('answer', r.get('proposed_answer')) for r in replies]} "
              f"expected {last.get('expected_answer')}  {seconds:.1f} s  {candidate['title']}")
        if ok:
            kept.append({**candidate, "answer": shown, "unit": last.get("unit"), "route": last.get("route_label"),
                         "seconds": round(seconds, 1)})
    exact = [k for k in kept if k["expect"] == "exact"]
    review = [k for k in kept if k["expect"] == "review"]
    chosen = (exact[:max(0, args.max - min(2, len(review)))] + review[:2])[:args.max]
    args.output.write_text(json.dumps(chosen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(chosen)} examples ({len(chosen) - len([c for c in chosen if c['expect'] == 'review'])} exact) "
          f"-> {args.output}", file=sys.stderr)
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())

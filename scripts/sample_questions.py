"""Draw a small per-folder evaluation sample from data/questions.jsonl.

Whole drawings are sampled, keeping every question of a chosen drawing, so
translations of one question stay together and share one extraction. Swiss
drawings are taken from odd site numbers only; even sites are reserved for
training the detector.

    python scripts/sample_questions.py --per-folder 6 --output runs/samples/mixed.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re


def held_out(path: str) -> bool:
    match = re.search(r"/site(\d+)_", path)
    return "swiss_dwellings/" not in path or (match is not None and int(match.group(1)) % 2 == 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default="data/questions.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-folder", type=int, default=6, help="Drawings sampled per dataset folder")
    parser.add_argument("--folders", nargs="*", help="Restrict to these folders")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    by_folder: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in open(args.questions, encoding="utf-8"):
        row = json.loads(line)
        folder = row["path"].split("/")[1]
        if (not args.folders or folder in args.folders) and held_out(row["path"]):
            by_folder[folder][row["path"]].append(row)

    rng = random.Random(args.seed)
    chosen: list[dict] = []
    for folder, drawings in sorted(by_folder.items()):
        paths = sorted(drawings)
        if folder == "ifc_models":
            # Keep both formats represented: native IFC and its exported PDFs.
            groups = [[p for p in paths if p.endswith(".ifc")], [p for p in paths if not p.endswith(".ifc")]]
            half = max(1, args.per_folder // 2)
            picked = [p for group in groups for p in rng.sample(group, min(half, len(group)))]
        else:
            picked = rng.sample(paths, min(args.per_folder, len(paths)))
        rows = [row for path in sorted(picked) for row in drawings[path]]
        chosen.extend(rows)
        print(f"{folder:16} {len(picked):3} drawings {len(rows):5} questions")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
    print(f"{'total':16} {len(chosen):15} questions -> {output}")


if __name__ == "__main__":
    main()

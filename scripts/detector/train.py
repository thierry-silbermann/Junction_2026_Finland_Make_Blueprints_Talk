"""Train the plan-symbol detector on tiles from make_labels.py.

Runs in the separate detector environment, because Ultralytics is AGPL-3.0
and is kept out of the drawingqa package dependencies:

    python3 -m venv .venv-detector && .venv-detector/bin/pip install ultralytics
    .venv-detector/bin/python scripts/detector/train.py --epochs 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="datasets/swiss_symbols/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="swiss_symbols")
    parser.add_argument("--device", help="cpu, cuda or mps (default: cuda if available, else cpu)")
    args = parser.parse_args()
    # Apple MPS produces NaN losses within the first epoch on these tiles, with or without AMP.
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    YOLO(args.model).train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        project=str(Path("runs/detector").resolve()), name=args.name, exist_ok=True, workers=4, patience=8,
        # Plans have no canonical orientation; symbols appear mirrored and upside down.
        fliplr=0.5, flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, plots=True)


if __name__ == "__main__":
    main()

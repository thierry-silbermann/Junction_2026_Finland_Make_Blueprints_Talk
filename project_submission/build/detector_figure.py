"""Run the in-training YOLO11n symbol detector on held-out drawings (needs .venv-detector)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "assets"
WEIGHTS = ROOT / "runs/detector/swiss_symbols/weights/best.pt"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
PALETTE = [(22, 163, 74), (37, 99, 235), (219, 39, 119), (234, 88, 12), (124, 58, 237), (8, 145, 178)]


def detect(model: YOLO, img: Image.Image, tile: int, overlap: int = 256, conf: float = 0.35):
    boxes, scores, classes = [], [], []
    step = tile - overlap
    for y in range(0, max(1, img.height - overlap), step):
        for x in range(0, max(1, img.width - overlap), step):
            crop = img.crop((x, y, min(x + tile, img.width), min(y + tile, img.height)))
            r = model.predict(crop, imgsz=640, conf=conf, verbose=False, device="cpu")[0]
            for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                boxes.append(b + torch.tensor([x, y, x, y]))
                scores.append(s)
                classes.append(c)
    if not boxes:
        return []
    b, s, c = torch.stack(boxes), torch.stack(scores), torch.stack(classes)
    keep = torchvision.ops.batched_nms(b, s, c, 0.5)
    return [(b[i].tolist(), int(c[i])) for i in keep]


def render(model: YOLO, src: Path, name: str, tile: int) -> None:
    img = Image.open(src).convert("RGB")
    dets = detect(model, img, tile)
    canvas = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype(FONT, max(14, img.width // 110))
    counts: dict[str, int] = {}
    for (x0, y0, x1, y1), c in dets:
        color = PALETTE[c % len(PALETTE)]
        label = model.names[c]
        counts[label] = counts.get(label, 0) + 1
        canvas.rectangle((x0, y0, x1, y1), outline=color + (255,), fill=color + (35,), width=max(2, img.width // 500))
    legend_y = 10
    for c, label in model.names.items():
        color = PALETTE[c % len(PALETTE)]
        canvas.rectangle((10, legend_y, 10 + font.size, legend_y + font.size), fill=color + (255,))
        canvas.text((16 + font.size, legend_y - 2), f"{label}: {counts.get(label, 0)}", fill=(20, 20, 20), font=font)
        legend_y += font.size + 8
    img.thumbnail((1800, 1800))
    img.save(OUT / f"{name}.png")
    print(name, src.name, counts)


def main() -> None:
    model = YOLO(str(WEIGHTS))
    swiss, cubicasa = sys.argv[1], sys.argv[2]
    render(model, Path(swiss), "detector_swiss_heldout", tile=1024)
    render(model, Path(cubicasa), "detector_cubicasa", tile=int(sys.argv[3]) if len(sys.argv) > 3 else 1024)


if __name__ == "__main__":
    main()

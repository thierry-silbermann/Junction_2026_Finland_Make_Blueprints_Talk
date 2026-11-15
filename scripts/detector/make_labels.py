"""Generate YOLO training tiles for plan symbols from the Swiss semantic DXFs.

Boxes come from the DXF layers (via drawingqa.dxf) and are mapped onto the
same-named PNG with the DXF-to-page fit. Only even site numbers are used;
odd sites stay held out for evaluation. A further slice of even sites is
kept as the YOLO validation split.

    PYTHONPATH=src .venv/bin/python scripts/detector/make_labels.py --output datasets/swiss_symbols
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import ezdxf
from ezdxf import bbox as ezbbox
from PIL import Image

from drawingqa.dxf import _cached_transform, read_drawing

CLASSES = ["door", "window", "toilet", "washbasin", "bathtub", "shower"]


def site(path: Path) -> int:
    return int(re.match(r"site(\d+)_", path.name).group(1))


def door_symbols(dxf: Path, frames: list[tuple]) -> list[tuple]:
    """Grow each thin door frame to include its leaf and swing arc."""
    parts = []
    for entity in ezdxf.readfile(dxf).modelspace().query('*[layer=="DOOR"]'):
        extents = ezbbox.extents([entity])
        if extents.has_data:
            parts.append((extents.extmin.x, extents.extmin.y, extents.extmax.x, extents.extmax.y))
    grown = []
    for f in frames:
        x1, y1, x2, y2 = f
        for p in parts:
            if p[0] <= f[2] + 0.02 and p[2] >= f[0] - 0.02 and p[1] <= f[3] + 0.02 and p[3] >= f[1] - 0.02:
                x1, y1, x2, y2 = min(x1, p[0]), min(y1, p[1]), max(x2, p[2]), max(y2, p[3])
        grown.append((x1, y1, x2, y2))
    return grown


def page_boxes(pdf: Path, dxf: Path) -> list[tuple[int, tuple]] | None:
    transform = _cached_transform(str(pdf), pdf.stat().st_mtime_ns, str(dxf), dxf.stat().st_mtime_ns)
    if transform is None:
        return None
    drawing = read_drawing(dxf)
    groups = {"door": door_symbols(dxf, drawing.doors), "window": drawing.windows,
              "toilet": drawing.fixtures.get("toilets", []), "washbasin": drawing.fixtures.get("washbasins", []),
              "bathtub": drawing.fixtures.get("bathtubs", []), "shower": drawing.fixtures.get("showers", [])}
    return [(CLASSES.index(name), transform(box)) for name, boxes in groups.items() for box in boxes]


def tiles(width: int, height: int, size: int, overlap: int):
    step = size - overlap
    xs = list(range(0, max(1, width - size) + 1, step)) or [0]
    ys = list(range(0, max(1, height - size) + 1, step)) or [0]
    if xs[-1] + size < width:
        xs.append(width - size)
    if ys[-1] + size < height:
        ys.append(height - size)
    return [(x, y) for y in ys for x in xs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/swiss_dwellings")
    parser.add_argument("--output", default="datasets/swiss_symbols")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--val-every", type=int, default=10, help="Every Nth even site goes to validation")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p for p in Path(args.data).glob("*.pdf") if site(p) % 2 == 0)
    if args.limit:
        pdfs = pdfs[:args.limit]
    sites = sorted({site(p) for p in pdfs})
    val_sites = set(sites[::args.val_every])
    stats = {"drawings": 0, "skipped": 0, "tiles": 0, "boxes": [0] * len(CLASSES)}
    for pdf in pdfs:
        dxf, png = pdf.with_suffix(".dxf"), pdf.with_suffix(".png")
        boxes = page_boxes(pdf, dxf) if dxf.exists() and png.exists() else None
        if not boxes:
            stats["skipped"] += 1
            continue
        split = "val" if site(pdf) in val_sites else "train"
        image = Image.open(png).convert("RGB")
        width, height = image.size
        pixel = [(c, (b[0] * width, b[1] * height, b[2] * width, b[3] * height)) for c, b in boxes]
        for x0, y0 in tiles(width, height, args.tile, args.overlap):
            lines = []
            for c, (x1, y1, x2, y2) in pixel:
                cx1, cy1 = max(x1, x0), max(y1, y0)
                cx2, cy2 = min(x2, x0 + args.tile), min(y2, y0 + args.tile)
                area = max(1e-9, (x2 - x1) * (y2 - y1))
                if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) / area < 0.6:
                    continue
                s = args.tile
                lines.append(f"{c} {((cx1 + cx2) / 2 - x0) / s:.6f} {((cy1 + cy2) / 2 - y0) / s:.6f} "
                             f"{(cx2 - cx1) / s:.6f} {(cy2 - cy1) / s:.6f}")
                stats["boxes"][c] += 1
            name = f"{pdf.stem}_{x0}_{y0}"
            image.crop((x0, y0, x0 + args.tile, y0 + args.tile)).save(out / "images" / split / f"{name}.jpg", quality=90)
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            stats["tiles"] += 1
        stats["drawings"] += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print({**stats, "boxes": dict(zip(CLASSES, stats["boxes"])), "val_sites": len(val_sites)})


if __name__ == "__main__":
    main()

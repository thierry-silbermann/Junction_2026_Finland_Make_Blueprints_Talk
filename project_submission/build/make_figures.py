"""Draw real evidence boxes from the per-folder run onto the drawings for the deck."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/per-folder-6-v2"
OUT = Path(__file__).resolve().parents[1] / "assets"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

COLORS = {"counted": (22, 163, 74), "uncertain": (234, 88, 12), "context": (37, 99, 235)}


def load() -> list[dict]:
    return [json.loads(line) for line in (RUN / "predictions.jsonl").open()]


def correct(p: dict) -> bool:
    a, e = p.get("answer"), p.get("expected_answer")
    if a is None or e is None:
        return False
    try:
        a, e = float(a), float(e)
    except (TypeError, ValueError):
        return False
    return a == e if p["unit"] == "count" else abs(a - e) <= 0.02 * abs(e)


def page_image(p: dict) -> Path:
    images = p.get("page_images") or {}
    rel = images.get("1") or next(iter(images.values()))
    return RUN / rel


def draw(p: dict, name: str, max_side: int = 1600, crop: bool = True) -> Path:
    img = Image.open(page_image(p)).convert("RGB")
    if max(img.size) < 1400:
        scale = 1400 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    w, h = img.size
    canvas = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype(FONT, max(14, w // 140))
    boxes = []
    for ev in p["evidence"]:
        if not ev.get("bbox"):
            continue
        x0, y0, x1, y1 = ev["bbox"]
        box = (x0 * w, y0 * h, x1 * w, y1 * h)
        boxes.append(box)
        kind = "uncertain" if ev.get("uncertain") else "counted" if ev.get("counted") else "context"
        color = COLORS[kind]
        canvas.rectangle(box, outline=color + (255,), fill=color + (40,), width=max(3, w // 400))
        label = str(int(ev.get("id", "0").rsplit("-", 1)[-1].lstrip("e") or 0))
        canvas.text((box[0] + 3, box[1] - font.size - 4), label, fill=color + (255,), font=font)
    if crop and boxes:
        pad = 0.06 * max(w, h)
        x0 = max(0, min(b[0] for b in boxes) - pad)
        y0 = max(0, min(b[1] for b in boxes) - pad)
        x1 = min(w, max(b[2] for b in boxes) + pad)
        y1 = min(h, max(b[3] for b in boxes) + pad)
        img = img.crop((int(x0), int(y0), int(x1), int(y1)))
    img.thumbnail((max_side, max_side))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    img.save(path)
    print(name, p["path"], "|", p["question"][:90], "| answer", p["answer"], p["proposed_answer"],
          "expected", p["expected_answer"], p["status"], len(p["evidence"]))
    return path


def pick(preds: list[dict], source: str, category: str, status: str = "answered", ok: bool | None = True,
         min_boxes: int = 2) -> dict:
    for p in preds:
        if (f"/{source}/" in p["path"] and p["category"] == category and p["status"] == status
                and sum(1 for e in p["evidence"] if e.get("bbox")) >= min_boxes
                and (ok is None or correct(p) == ok)):
            return p
    raise LookupError((source, category, status))


def main() -> None:
    preds = load()
    meta = {}
    for name, args in {
        "swiss_doors_dxf": ("swiss_dwellings", "doors"),
        "swiss_rooms_dxf": ("swiss_dwellings", "rooms_named"),
        "cubicasa_toilets_vlm": ("cubicasa5k", "toilets"),
        "cubicasa_sinks_vlm": ("cubicasa5k", "sinks"),
    }.items():
        p = pick(preds, *args)
        draw(p, name, crop=args[0] != "cubicasa5k")
        meta[name] = {k: p[k] for k in ("path", "question", "answer", "expected_answer", "unit", "model_id")}
    review = pick(preds, "cubicasa5k", "doors", status="needs_review", ok=None)
    draw(review, "cubicasa_doors_review", crop=False)
    meta["cubicasa_doors_review"] = {k: review[k] for k in ("path", "question", "proposed_answer",
                                                           "expected_answer", "warnings")}
    ifc = next(p for p in preds if "/ifc_models/" in p["path"] and p["category"] == "room_area" and correct(p))
    meta["ifc_room_area"] = {k: ifc[k] for k in ("path", "question", "answer", "expected_answer", "unit",
                                                "model_id", "rationale")}
    meta["ifc_room_area"]["evidence"] = ifc["evidence"][:3]
    (OUT / "figures.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

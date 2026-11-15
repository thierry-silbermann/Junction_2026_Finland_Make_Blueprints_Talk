"""Native answers from a layered DXF that accompanies a PDF drawing.

The dataset ships some drawings as PDF, DXF and PNG renderings of the same plan.
When a PDF question has a same-named DXF sibling with semantic layers (DOOR,
WINDOW, FIXTURE, ROOM_OUTLINE, ROOM_TEXT, UNIT_OUTLINE), supported counts and
areas are computed from vector entities and printed room labels instead of a
VLM. Evidence boxes are mapped onto the rendered PDF page through an affine fit
between DXF texts and the same texts extracted from the PDF.
"""
from __future__ import annotations

import importlib
import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import Evidence, Prediction, Question, Usage

ADAPTER_VERSION = "dxf-layers-v1"
REQUIRED_LAYERS = {"DOOR", "WINDOW", "FIXTURE", "ROOM_TEXT"}

_COUNT_PATTERNS = {
    "doors": r"^(How many doors|Wie viele Türen|Combien de portes|Quante porte)\b",
    "windows": r"^(How many windows|Wie viele Fenster|Combien de fenêtres|Quante finestre)\b",
    "toilets": r"^(How many toilets|Wie viele WCs|Combien de WC|Quanti WC)\b",
    "washbasins": r"^(How many washbasins|Wie viele Lavabos|Combien de lavabos|Quanti lavabi)\b",
    "bathtubs": r"^(How many bathtubs|Wie viele Badewannen|Combien de baignoires|Quante vasche da bagno)\b",
    "showers": r"^(How many showers|Wie viele Duschen|Combien de douches|Quante docce)\b",
    "apartments": r"^(How many apartments|Wie viele Wohnungen|Combien d'appartements|Combien de logements|Quanti appartamenti)\b",
}
_NAME = r"(?:'(?P<q>.+)'|«(?P<g>.+)»)"
_QUERY_PATTERNS = [
    ("area_named", r"^(What is the total net area in m² of all rooms labelled|Wie gross ist die gesamte Nettofläche in m² aller Räume mit der Bezeichnung|Quelle est la surface nette totale en m² de toutes les pièces désignées|Qual è la superficie netta totale in m² di tutti i locali denominati) " + _NAME),
    ("area_total", r"^(What is the total net floor area in m² of all numbered indoor rooms|Wie gross ist die gesamte Nettogeschossfläche in m² aller nummerierten Innenräume|Quelle est la surface nette totale en m² de toutes les pièces intérieures numérotées|Qual è la superficie netta totale in m² di tutti i locali interni numerati)\b"),
    ("area_room", r"^(What is the net floor area of room|Wie gross ist die Nettofläche von Raum|Quelle est la surface nette de la pièce|Qual è la superficie netta del locale) (?P<number>\S+) \(" + _NAME + r"\)"),
    ("area_apartment", r"^(What is the net floor area of apartment|Wie gross ist die Nettowohnfläche von Wohnung|Quelle est la surface nette de l'appartement|Qual è la superficie netta dell'appartamento) (?P<letter>[A-Z]+)\b"),
    ("count_named", r"^(How many rooms labelled|Wie viele Räume mit der Bezeichnung|Combien de pièces portant la désignation|Quanti locali con la denominazione) " + _NAME),
]
# Rooms that the questions exclude from net living/indoor areas.
_OUTDOOR = {"balkon", "balcon", "balcone", "loggia", "terrasse", "terrazza", "sitzplatz", "veranda",
            "garten", "jardin", "giardino", "wintergarten", "jardin d'hiver", "giardino d'inverno"}
_VOIDS = {"schacht", "luftraum", "gaine", "vide", "cavedio", "vuoto"}
_UNIT_LABEL = re.compile(r"^(?:Whg|Appt|App|Apt|Unit)\.?\s*(?P<letter>[A-Z]+)$")
_ROOM_NUMBER = re.compile(r"^[A-Z]?\d+(?:\.\d+)+$")
_AREA = re.compile(r"^(?P<value>\d+(?:[.,]\d+)?)\s*m²$")


@dataclass(frozen=True)
class DxfQuery:
    kind: str
    name: str | None = None
    number: str | None = None
    letter: str | None = None


def parse_query(question: str) -> DxfQuery | None:
    text = " ".join(question.split())
    for kind, pattern in _QUERY_PATTERNS:
        match = re.match(pattern, text)
        if match:
            groups = match.groupdict()
            return DxfQuery(kind, name=groups.get("q") or groups.get("g"),
                            number=groups.get("number"), letter=groups.get("letter"))
    for category, pattern in _COUNT_PATTERNS.items():
        if re.match(pattern, text):
            return DxfQuery("count_" + category)
    return None


def dxf_category(query: DxfQuery) -> str:
    if query.kind == "count_named":
        return "rooms_named"
    return query.kind.removeprefix("count_")


def sibling_dxf(path: Path) -> Path | None:
    if path.suffix.lower() != ".pdf":
        return None
    candidate = path.with_suffix(".dxf")
    return candidate if candidate.is_file() else None


@dataclass
class Room:
    number: str
    name: str | None
    area: float | None
    printed_area: float | None
    anchor: tuple[float, float]
    box: tuple[float, float, float, float]


@dataclass
class Drawing:
    doors: list[tuple] = field(default_factory=list)
    windows: list[tuple] = field(default_factory=list)
    fixtures: dict[str, list[tuple]] = field(default_factory=dict)
    rooms: list[Room] = field(default_factory=list)
    units: dict[str, tuple] = field(default_factory=dict)
    unit_polygons: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    texts: list[tuple[str, float, float]] = field(default_factory=list)
    missing_layers: list[str] = field(default_factory=list)


def _points(entity) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in entity.get_points()]


def _box(points) -> tuple[float, float, float, float]:
    xs, ys = zip(*points)
    return (min(xs), min(ys), max(xs), max(ys))


def _inside(point, polygon) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _polygon_area(polygon) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]))) / 2


def _contains(outer, inner, tolerance=1e-6) -> bool:
    return (outer[0] - tolerance <= inner[0] and outer[1] - tolerance <= inner[1]
            and inner[2] <= outer[2] + tolerance and inner[3] <= outer[3] + tolerance)


def _anchor(text) -> tuple[float, float]:
    point = text.dxf.align_point if text.dxf.get("halign", 0) or text.dxf.get("valign", 0) else text.dxf.insert
    return (float(point[0]), float(point[1]))


def _rooms(texts, outlines) -> list[Room]:
    """Group the stacked name / number / area labels around each room number.

    The net area is the enclosing ROOM_OUTLINE polygon's area; a printed area
    label, when present, is kept as a cross-check.
    """
    rooms = []
    for text in texts:
        value = text.dxf.text.strip()
        if not _ROOM_NUMBER.match(value):
            continue
        x, y = _anchor(text)
        step = 2.2 * float(text.dxf.height)
        name = printed = None
        for other in texts:
            ox, oy = _anchor(other)
            if other is text or abs(ox - x) > 1e-3:
                continue
            content = other.dxf.text.strip()
            if 0 < oy - y <= step and not _UNIT_LABEL.match(content) and not _AREA.match(content):
                name = content
            elif 0 < y - oy <= step and (match := _AREA.match(content)):
                printed = float(match.group("value").replace(",", "."))
        # Outlines can nest (a lift inside a stairwell); the innermost one is the room.
        polygon = min((p for p in outlines if _inside((x, y), p)), key=_polygon_area, default=None)
        box = _box(polygon) if polygon else (x - step, y - step, x + step, y + step)
        area = round(_polygon_area(polygon), 2) if polygon else printed
        rooms.append(Room(value, name, area, printed, (x, y), box))
    return rooms


def read_drawing(path: Path) -> Drawing:
    return _read_drawing(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=32)
def _read_drawing(path: str, _mtime: int) -> Drawing:
    ezdxf = importlib.import_module("ezdxf")
    msp = ezdxf.readfile(path).modelspace()
    layers: dict[str, list[Any]] = {}
    for entity in msp:
        layers.setdefault(entity.dxf.layer, []).append(entity)
    if not REQUIRED_LAYERS & set(layers):
        raise ValueError("DXF has none of the semantic layers DOOR, WINDOW, FIXTURE or ROOM_TEXT")
    drawing = Drawing()
    drawing.missing_layers = sorted(REQUIRED_LAYERS - set(layers))
    polylines = lambda layer: [e for e in layers.get(layer, []) if e.dxftype() == "LWPOLYLINE"]
    # Each door is drawn as one 5-vertex frame, one leaf line and an optional swing arc.
    drawing.doors = [_box(_points(e)) for e in polylines("DOOR") if len(e) == 5]
    # Each window is an outer frame enclosing two inner glazing rectangles.
    window_boxes = [_box(_points(e)) for e in polylines("WINDOW")]
    drawing.windows = [b for i, b in enumerate(window_boxes)
                       if not any(j != i and _contains(o, b) and o != b for j, o in enumerate(window_boxes))]
    open_fixtures = [e for e in polylines("FIXTURE") if not e.closed]
    seats = [_box(_points(e)) for e in open_fixtures if len(e) == 2]
    toilets, basins = [], []
    for ellipse in (e for e in open_fixtures if len(e) == 48):
        box = _box(_points(ellipse))
        size = min(box[2] - box[0], box[3] - box[1])
        # A toilet bowl has a seat line (possibly rotated) across the ellipse; a basin has none.
        has_seat = any(_contains((box[0] - 0.1, box[1] - 0.1, box[2] + 0.1, box[3] + 0.1), s)
                       and math.hypot(s[2] - s[0], s[3] - s[1]) > 0.6 * size for s in seats)
        (toilets if has_seat else basins).append(box)
    drawing.fixtures = {
        "toilets": toilets, "washbasins": basins,
        "bathtubs": [_box(_points(e)) for e in open_fixtures if len(e) == 69],
        "showers": [_box(_points(e)) for e in open_fixtures if len(e) == 5],
    }
    texts = [e for e in layers.get("ROOM_TEXT", []) if e.dxftype() == "TEXT"]
    outlines = [_points(e) for e in polylines("ROOM_OUTLINE") if len(e) >= 3]
    drawing.rooms = _rooms(texts, outlines)
    unit_outlines = [_points(e) for e in polylines("UNIT_OUTLINE") if len(e) >= 3]
    for text in texts:
        match = _UNIT_LABEL.match(text.dxf.text.strip())
        if match:
            point = _anchor(text)
            polygon = next((p for p in unit_outlines if _inside(point, p)), None)
            drawing.units[match.group("letter")] = _box(polygon) if polygon else (point[0] - 1, point[1] - 0.5, point[0] + 1, point[1] + 0.5)
            if polygon:
                drawing.unit_polygons[match.group("letter")] = polygon
    drawing.texts = [(e.dxf.text.strip(), *_anchor(e)) for e in texts]
    return drawing


def _page_transform(pdf: Path, texts: list[tuple[str, float, float]]):
    """Fit page = a * dxf + b per axis from texts that occur once in both files."""
    fitz = importlib.import_module("fitz")
    with fitz.open(pdf) as document:
        page = document[0]
        width, height = page.rect.width, page.rect.height
        spans: dict[str, list] = {}
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    spans.setdefault(span["text"].strip(), []).append(span["bbox"])
    counts: dict[str, int] = {}
    for value, _, _ in texts:
        counts[value] = counts.get(value, 0) + 1
    pairs = [((x, y), spans[value][0]) for value, x, y in texts
             if counts[value] == 1 and len(spans.get(value, [])) == 1]
    if len(pairs) < 3:
        return None

    def fit(source, target):
        n = len(source)
        mean_s, mean_t = sum(source) / n, sum(target) / n
        var = sum((s - mean_s) ** 2 for s in source)
        if var == 0:
            return None
        scale = sum((s - mean_s) * (t - mean_t) for s, t in zip(source, target)) / var
        return scale, mean_t - scale * mean_s

    fx = fit([p[0][0] for p in pairs], [(b[0] + b[2]) / 2 for _, b in pairs])
    fy = fit([p[0][1] for p in pairs], [(b[1] + b[3]) / 2 for _, b in pairs])
    if fx is None or fy is None:
        return None

    def transform(box):
        xs = sorted(((fx[0] * box[0] + fx[1]) / width, (fx[0] * box[2] + fx[1]) / width))
        ys = sorted(((fy[0] * box[1] + fy[1]) / height, (fy[0] * box[3] + fy[1]) / height))
        clamp = lambda v: min(1.0, max(0.0, v))
        x1, x2, y1, y2 = clamp(xs[0]), clamp(xs[1]), clamp(ys[0]), clamp(ys[1])
        if x2 - x1 < 1e-4:
            x1, x2 = max(0.0, x1 - 0.001), min(1.0, x2 + 0.001)
        if y2 - y1 < 1e-4:
            y1, y2 = max(0.0, y1 - 0.001), min(1.0, y2 + 0.001)
        return (x1, y1, x2, y2)

    return transform


@lru_cache(maxsize=32)
def _cached_transform(pdf: str, _pdf_mtime: int, dxf: str, _dxf_mtime: int):
    return _page_transform(Path(pdf), read_drawing(Path(dxf)).texts)


def _same(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a.casefold().strip() == b.casefold().strip()


def answer_dxf(pdf: Path, dxf: Path, question: Question, query: DxfQuery, category: str,
               model_alias: str = "dxf") -> Prediction:
    started = time.monotonic()
    base = dict(question_id=question.id or "", path=question.path, question=question.question,
                page=question.page, model=model_alias, model_id=ADAPTER_VERSION,
                unit=question.unit, category=category)

    def result(status: str, **values) -> Prediction:
        return Prediction(**base, status=status,
                          usage=Usage(latency_seconds=time.monotonic() - started, estimated_cost_usd=0.0),
                          **values)

    if question.page not in (None, 1):
        return result("unsupported", error="The DXF sibling describes page 1 of this PDF only.")
    try:
        importlib.import_module("ezdxf")
    except ImportError:
        return result("unsupported", error="DXF support is optional. Install it with: python -m pip install '.[dxf]'")
    try:
        drawing = read_drawing(dxf)
        transform = _cached_transform(str(pdf), pdf.stat().st_mtime_ns, str(dxf), dxf.stat().st_mtime_ns)
    except Exception as exc:
        return result("error", error=f"DXF query failed ({type(exc).__name__}). Check the DXF layers.")

    warnings = [f"Answered from the same-named DXF ({dxf.name}) using its semantic layers, not by visual inference."]
    if transform is None:
        warnings.append("Could not align DXF and PDF coordinates; evidence is listed without boxes.")
    layer = {"count_doors": "DOOR", "count_windows": "WINDOW"}.get(query.kind, "FIXTURE" if query.kind in
             {"count_toilets", "count_washbasins", "count_bathtubs", "count_showers"} else "ROOM_TEXT")
    if layer in drawing.missing_layers:
        warnings.append(f"The DXF has no {layer} layer; the answer treats it as containing no entities.")
    evidence: list[Evidence] = []

    def add(label: str, box, note: str = "", counted: bool = True):
        evidence.append(Evidence(id=f"dxf-{len(evidence) + 1:04d}", label=label[:200],
                                 bbox=transform(box) if transform else None, page=1, category=category,
                                 provenance="dxf_entity", note=note[:1200], counted=counted))

    answer: float | int | None = None
    uncertain = False
    kind = query.kind
    if kind.startswith("count_") and kind != "count_named":
        item = kind.removeprefix("count_")
        if item == "doors":
            boxes, label, rule = drawing.doors, "door", "one DOOR frame per door"
        elif item == "windows":
            boxes, label, rule = drawing.windows, "window", "one outer WINDOW frame per window"
        elif item == "apartments":
            boxes, label, rule = list(drawing.units.values()), "apartment", "one apartment label (e.g. Whg. A) per unit"
        else:
            boxes, label, rule = drawing.fixtures[item], item.rstrip("s"), f"FIXTURE symbols classified as {item}"
        if item == "apartments":
            for letter, box in sorted(drawing.units.items()):
                add(f"apartment {letter}", box, "Unit outline containing its apartment label.")
        else:
            for box in boxes:
                add(label, box)
        answer = len(boxes)
        rationale = f"Counted {answer} {item}: {rule}."
    elif kind == "count_named":
        matches = [r for r in drawing.rooms if _same(r.name, query.name)]
        for room in matches:
            add(f"{room.name} {room.number}", room.box, f"Printed label: {room.name}, room {room.number}.")
        answer = len(matches)
        rationale = f"Counted {answer} rooms whose printed name is exactly '{query.name}'."
    else:
        if kind == "area_room":
            selected = [r for r in drawing.rooms if r.number == query.number]
            scope = f"room {query.number}"
            if selected and query.name and not _same(selected[0].name, query.name):
                uncertain = True
                warnings.append(f"Room {query.number} is labelled '{selected[0].name}', not '{query.name}'.")
        elif kind == "area_named":
            selected = [r for r in drawing.rooms if _same(r.name, query.name)]
            scope = f"rooms labelled '{query.name}'"
        elif kind == "area_apartment":
            polygon = drawing.unit_polygons.get(query.letter or "")
            selected = [r for r in drawing.rooms if polygon and _inside(r.anchor, polygon)
                        and (r.name or "").casefold() not in _OUTDOOR]
            scope = f"apartment {query.letter} excluding outdoor spaces"
            if polygon is None:
                uncertain = True
                warnings.append(f"No unit outline contains the label for apartment {query.letter}.")
        else:
            selected = [r for r in drawing.rooms if (r.name or "").casefold() not in _OUTDOOR | _VOIDS]
            scope = "all numbered indoor rooms"
        if not selected:
            uncertain = True
            warnings.append(f"No numbered rooms were found for {scope}.")
        missing = [r.number for r in selected if r.area is None]
        if missing:
            uncertain = True
            warnings.append(f"Rooms without an outline or printed area: {', '.join(missing)}.")
        disagree = [r.number for r in selected if r.area is not None and r.printed_area is not None
                    and abs(r.area - r.printed_area) > 0.01 * max(r.printed_area, 0.01)]
        if disagree:
            uncertain = True
            warnings.append(f"Outline area differs from the printed area for rooms: {', '.join(disagree)}.")
        for room in selected:
            add(f"{room.name or 'room'} {room.number}", room.box,
                f"Outline area {room.area} m²" + (f"; printed {room.printed_area} m²." if room.printed_area is not None else "."))
        answer = round(sum(r.area for r in selected if r.area is not None), 2)
        terms = " + ".join(f"{r.area}" for r in selected if r.area is not None)
        rationale = f"Sum of printed areas for {scope}: {terms} = {answer} m²." if len(selected) > 1 else \
            f"Printed area for {scope}: {answer} m²."
    status = "needs_review" if uncertain else "answered"
    return result(status, answer=None if uncertain else answer, proposed_answer=answer,
                  evidence=evidence, rationale=rationale[:8000], warnings=warnings)

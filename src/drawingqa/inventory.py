"""Per-drawing inventory: one vision request per page, shared by every question about it.

The model lists what is drawn and what is printed; Python selects, counts and sums.
Questions the inventory cannot express return None from `select` and keep the
per-question route, so the per_drawing strategy never answers fewer questions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .dxf import _OUTDOOR, _VOIDS, dxf_category, parse_query
from .ifc_takeoff import parse_takeoff, takeoff_category
from .prompts import BBOX_RULES, SYSTEM
from .schemas import DrawingInventory, Evidence, Intent, InventoryItem, Question

INVENTORY_VERSION = "drawingqa-inventory-v2"

_KIND_RULES = {
    "door": "door: one per doorway fitted with a door leaf; a double-leaf door is one. Not open passages or cabinet doors.",
    "window": "window: one per window unit, not per pane or stroke.",
    "toilet": "toilet: one per toilet bowl (WC seat).",
    "washbasin": "washbasin: one per bathroom wash basin; a double basin unit is one.",
    "kitchen_sink": "kitchen_sink: one per kitchen sink; a double-bowl sink is one.",
    "bathtub": "bathtub: one per drawn bathtub.",
    "shower": "shower: one per drawn shower.",
    "stair": "stair: one per staircase.",
    "column": "column: one per structural column or pillar symbol.",
    "room": ("room: one per room or space. text = the room name exactly as printed (e.g. \"Schlafzimmer\", \"MH\", \"Bad\"); "
             "number = the printed room number (e.g. \"0.10\", \"7.01\"); area = the printed net area in m² for that room, "
             "only if printed, never estimated; room_type = its function from the label or fittings; apartment = the "
             "apartment or unit letter it belongs to when the drawing shows it (e.g. \"A\" for \"Whg. A\"). "
             "bbox = the room's extent."),
    "apartment": "apartment: one per printed apartment or unit label (e.g. \"Whg. A\", \"Appt B\"); text = the label as printed.",
    "floor": "floor: one per distinct storey panel on the sheet; text = its floor label.",
}

# A small model asked for every kind at once cycles through the kinds and invents items until the
# token limit, so each request covers one focused group. A drawing needs at most one request per group.
GROUPS = {
    "openings": ("door", "window"),
    "fixtures": ("toilet", "washbasin", "kitchen_sink", "bathtub", "shower"),
    "spaces": ("room", "apartment", "floor"),
    "structure": ("stair", "column"),
}
_GROUP_OF = {kind: group for group, kinds in GROUPS.items() for kind in kinds}

_TASK = """List every {names} visible in this drawing, each exactly once. Software will answer several
questions from this list, so do not answer any question yourself and do not list any other kind of object.
Kinds:
{kinds}
Leave out fields that do not apply; symbols need only kind and bbox. Never repeat an item.
Keep ambiguous items with uncertain=true instead of dropping them. An empty list with complete=true means
you inspected the whole image and found none of these kinds.
Set complete=true only if you inspected the whole image and listed every relevant item."""

_COUNT_KINDS = {
    "doors": ("door",), "windows": ("window",), "toilets": ("toilet",),
    "sinks": ("washbasin", "kitchen_sink"), "showers_bathtubs": ("shower", "bathtub"), "floors": ("floor",),
    "washbasins": ("washbasin",), "bathtubs": ("bathtub",), "showers": ("shower",),
    "rooms": ("room",), "stairs": ("stair",), "columns": ("column",),
}
_ROOM_TYPES = {"bedrooms": "bedroom", "kitchens": "kitchen", "bathrooms": "bathroom", "wc_rooms": "wc",
               "saunas": "sauna", "balconies": "balcony"}


@dataclass(frozen=True)
class Selector:
    operation: str  # count | count_labels | area | sum_area
    category: str
    kinds: tuple[str, ...]
    room_type: str | None = None
    name: str | None = None
    number: str | None = None
    apartment: str | None = None
    indoor_only: bool = False
    numbered_only: bool = False


def group_of(selector: Selector) -> str:
    groups = {_GROUP_OF[kind] for kind in selector.kinds}
    if len(groups) != 1:
        raise ValueError(f"Selector kinds span several inventory groups: {selector.kinds}")
    return groups.pop()


def build_inventory_prompt(group: str, bbox_scale: int) -> tuple[str, str]:
    kinds = GROUPS[group]
    system = SYSTEM.format(bbox_rule=BBOX_RULES[bbox_scale])
    schema = json.dumps(DrawingInventory.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    task = _TASK.format(names=", ".join(kinds), kinds="\n".join(f"- {_KIND_RULES[k]}" for k in kinds))
    user = f"Prompt version: {INVENTORY_VERSION}\nInventory group: {group}\nTask: {task}\nJSON schema: {schema}"
    return system, user


def select(question: Question, intent: Intent, path: Path) -> Selector | None:
    """Map a question onto the inventory, or None to keep the per-question route."""
    if intent.operation == "count":
        if intent.category in _ROOM_TYPES:
            return Selector("count", intent.category, ("room",), room_type=_ROOM_TYPES[intent.category])
        return Selector("count", intent.category, _COUNT_KINDS[intent.category])
    takeoff = parse_takeoff(question.question, question.unit)
    # "How many doors ... in the model 'X'" spans every storey; one sheet cannot answer it.
    if takeoff is not None and takeoff.scope == "model":
        return None
    query = parse_query(question.question)
    if query is not None:
        category = dxf_category(query)
        if query.kind == "count_apartments":
            return Selector("count_labels", category, ("apartment",))
        if query.kind.startswith("count_") and query.kind != "count_named":
            return Selector("count", category, _COUNT_KINDS[query.kind.removeprefix("count_")])
        if query.kind == "count_named":
            return Selector("count", category, ("room",), name=query.name)
        if query.kind == "area_room":
            return Selector("area", category, ("room",), name=query.name, number=query.number)
        if query.kind == "area_named":
            return Selector("sum_area", category, ("room",), name=query.name)
        if query.kind == "area_total":
            return Selector("sum_area", category, ("room",), indoor_only=True, numbered_only=True)
        if query.kind == "area_apartment":
            return Selector("sum_area", category, ("room",), apartment=query.letter, indoor_only=True)
    if takeoff is not None and path.suffix.lower() != ".ifc":
        category = takeoff_category(takeoff)
        if takeoff.kind == "count" and takeoff.element in _COUNT_KINDS:
            return Selector("count", category, _COUNT_KINDS[takeoff.element])
        if takeoff.kind == "room_area":
            return Selector("area", category, ("room",), name=takeoff.room)
        if takeoff.kind == "storey_area":
            return Selector("sum_area", category, ("room",))
    return None


def _fold(text: str | None) -> str:
    return " ".join((text or "").split()).casefold()


def _room_matches(item: InventoryItem, name: str | None, number: str | None) -> bool:
    if number is not None:
        return _fold(item.number) == _fold(number)
    target = _fold(name)
    labels = {_fold(item.text), _fold(item.number), _fold(f"{item.text} ({item.number})")}
    parts = re.fullmatch(r"(.*) \((.+)\)", target)
    if parts and (_fold(item.number) == parts.group(2) or (not item.number and _fold(item.text) == parts.group(1))):
        return True
    return target in labels - {""}


def _indoor(item: InventoryItem) -> bool:
    return item.room_type not in {"balcony", "terrace"} and _fold(item.text) not in _OUTDOOR | _VOIDS


def _chosen(selector: Selector, entries: list[tuple[int, str, InventoryItem]]):
    chosen = []
    for entry in entries:
        item = entry[2]
        if item.kind not in selector.kinds:
            continue
        if selector.room_type and item.room_type != selector.room_type:
            continue
        if selector.operation in {"count", "sum_area"} and selector.name is not None and _fold(item.text) != _fold(selector.name):
            continue
        if selector.operation == "area" and not _room_matches(item, selector.name, selector.number):
            continue
        if selector.apartment is not None and _fold(item.apartment) != _fold(selector.apartment):
            continue
        if selector.indoor_only and not _indoor(item):
            continue
        if selector.numbered_only and not item.number:
            continue
        chosen.append(entry)
    return chosen


def _evidence(entries, category: str) -> list[Evidence]:
    evidence = []
    for index, (page, view_id, item) in enumerate(entries, 1):
        label = f"{item.kind}: {item.text}" if item.text else item.kind
        note = " · ".join(filter(None, [item.number, f"{item.area:g} m²" if item.area is not None else "",
                                         item.room_type or "", f"apt {item.apartment}" if item.apartment else ""]))
        evidence.append(Evidence(label=label[:200], bbox=item.bbox, uncertain=item.uncertain, note=note, id=f"e{index}",
                                 page=page, category=category, view_ids=[view_id]))
    return evidence


def answer(selector: Selector, entries: list[tuple[int, str, InventoryItem]], complete: bool) -> dict:
    """Deterministic answer from inventory entries whose boxes are already in page coordinates."""
    chosen = _chosen(selector, entries)
    evidence = _evidence(chosen, selector.category)
    warnings = [] if complete else ["Model did not declare a complete inventory"]
    review = not complete or any(item.uncertain for _, _, item in chosen)
    proposed = None
    if selector.operation == "count":
        proposed = len(chosen)
        if proposed == 0:
            warnings.append("Zero means no matching entities were identified; absence has not been independently verified")
        rationale = f"Counted {proposed} '{selector.category}' item(s) in the shared drawing inventory."
    elif selector.operation == "count_labels":
        proposed = len({_fold(item.text) or str(i) for i, (_, _, item) in enumerate(chosen)})
        rationale = f"Counted {proposed} distinct apartment label(s) in the shared drawing inventory."
    elif selector.operation == "area":
        if len(chosen) != 1:
            review = True
            warnings.append(f"{len(chosen)} rooms in the inventory match this room; expected exactly one")
        elif chosen[0][2].area is None:
            review = True
            warnings.append("The matching room has no printed area in the inventory")
        else:
            proposed = chosen[0][2].area
        rationale = "Printed area of the matching room in the shared drawing inventory."
    else:
        areas = [item.area for _, _, item in chosen]
        if not chosen:
            review = True
            warnings.append("No matching rooms in the inventory")
        elif None in areas:
            review = True
            warnings.append(f"{areas.count(None)} of {len(areas)} matching rooms have no printed area")
        if chosen and None not in areas:
            proposed = round(sum(areas), 2)
        rationale = f"Sum of the printed areas of {len(chosen)} matching room(s) in the shared drawing inventory."
    if proposed is None:
        review = True
    return {"proposed": proposed, "review": review, "evidence": evidence, "warnings": warnings, "rationale": rationale}

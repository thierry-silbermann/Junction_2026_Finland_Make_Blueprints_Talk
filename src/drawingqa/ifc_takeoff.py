"""Quantity take-off from an IFC model for storey, room and model questions.

Answers questions that name a model, storey or room in quotes, e.g. "How many
doors are shown on the floor plan of 'Level 1'?" or "What is the net floor
area of the room 'Kitchen (A103)' on 'Level 1', in m²?", in several languages.
A floor plan `<model>_plan-NN-<level>.pdf` is answered from `<model>.ifc`.

Room areas are the footprint of the IfcSpace body geometry, which is defined
for every model; stored quantities and authoring-tool areas vary by exporter.
Element counts exclude parts decomposed from an element of the same class
(e.g. stair risers aggregated into a stair).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
import re
import time
import unicodedata
from typing import Any

from .ifc import _evidence, _sha256
from .schemas import Prediction, Question, Usage

ADAPTER_VERSION = "ifc-takeoff-v1"

# A single-quoted name may contain an apostrophe, so it ends only at a quote before a separator.
_QUOTED = re.compile(r"'(.+?)'(?=[\s,?.:;)]|$)|«([^»]+)»|„([^“”]+)[“”]|“([^”]+)”")
_ELEMENTS = {
    "doors": r"doors?|türen|deuren|døre|drzwi|uks|ust|uste|puertas|uși|usi|portas|porte",
    "windows": r"windows?|fenster|ramen|vinduer|okien|okna|aken|akent|akna|ventanas|ferestre|janelas|finestre",
    "walls": r"walls?|wände|wänden|wanden|vægge|ścian|ściany|seina|seinu|seinte|muros|pereți|pereti|paredes|muri",
    "columns": r"columns?|stützen|kolommen|søjler|słupów|słupy|sammast|sambaid|posti|postid|pilares|stâlpi|stalpi|colunas|pilastri|colonne",
    "stairs": r"stairs?|treppen|trappen|trapper|schodów|schody|treppi|trepid|treppe|escaleras|scări|scari|escadas",
    "beams": r"beams?|träger|balken|bjælker|belek|belki|talasid|tala|vigas|grinzi",
    "rooms": r"rooms?|spaces|räume|räumen|raums|ruimtes|rum|rummet|pomieszczeń|pomieszczenia|ruumi|ruume|ruumide|locales|încăperi|incaperi|camere|espaços",
    "slabs": r"slabs?|decken|bodenplatten|vloeren|dæk|plader|stropów|płyt|plaatide|plaadid|forjados|losas|planșee|plansee|lajes",
    "sanitary": r"sanitary|sanitär\w*|sanitair\w*|sanitet\w*|sanitarn\w*|sanitaar\w*|sanitari\w*|sanitar\w*",
    "lights": r"light|leuchten|verlichting\w*|belysning\w*|oprawy|oświetl\w*|valgusti\w*|luminarias|iluminat|luminárias",
}
_ELEMENT_RES = {name: re.compile(rf"(?<!\w)(?:{words})(?!\w)") for name, words in _ELEMENTS.items()}
_MODEL_SCOPE = re.compile(r"(?<!\w)(?:model|modell|modellen|modelo|modelu|modelul|modello|mudel|mudelis)(?!\w)")
_LENGTH = re.compile(r"length|länge|lang\b|lengte|længde|długość|pikkus|longitud|lungime|comprimento")
_CLASSES = {
    "doors": ("IfcDoor",), "windows": ("IfcWindow",), "walls": ("IfcWall",), "columns": ("IfcColumn",),
    "stairs": ("IfcStair",), "beams": ("IfcBeam",), "rooms": ("IfcSpace",), "slabs": ("IfcSlab",),
}
# IFC2X3 has no occurrence classes for these; occurrences are IfcFlowTerminal with a typed product type.
_TERMINALS = {"sanitary": ("IfcSanitaryTerminal", "IfcSanitaryTerminalType"),
              "lights": ("IfcLightFixture", "IfcLightFixtureType")}


@dataclass(frozen=True)
class TakeoffQuery:
    kind: str  # count | room_area | storey_area | sum
    element: str | None = None
    scope: str = "storey"  # storey | model
    storey: str | None = None
    room: str | None = None
    quantity: str | None = None  # NetVolume | NetSideArea | Length


def sibling_ifc(path: Path) -> Path | None:
    if path.suffix.lower() == ".ifc":
        return path
    stem = re.sub(r"_(plan|elevation|section)-[^/]*$", "", path.stem)
    candidate = path.with_name(stem + ".ifc")
    return candidate if stem != path.stem and candidate.exists() else None


def _unit(question: str, unit: str | None) -> str:
    if unit:
        return {"m²": "m2", "m³": "m3"}.get(unit, unit)
    if "m³" in question or re.search(r"\bm3\b", question):
        return "m3"
    if "m²" in question or re.search(r"\bm2\b", question):
        return "m2"
    return "m" if _LENGTH.search(question.casefold()) else "count"


def parse_takeoff(question: str, unit: str | None = None) -> TakeoffQuery | None:
    names = [next(g for g in match.groups() if g) for match in _QUOTED.finditer(question)]
    if not names or len(names) > 2:
        return None
    rest = unicodedata.normalize("NFC", _QUOTED.sub(" ", question)).casefold()
    found = [name for name, pattern in _ELEMENT_RES.items() if pattern.search(rest)]
    kind_unit = _unit(question, unit)
    if kind_unit == "m2" and len(names) == 2:
        return TakeoffQuery("room_area", "rooms", room=names[0], storey=names[1])
    if len(names) != 1:
        return None
    if kind_unit == "count":
        # "rooms" also appears in sanitary/fixture wording in some languages; prefer the specific class.
        found = [f for f in found if f != "rooms"] or found
        if len(found) != 1:
            return None
        if _MODEL_SCOPE.search(rest):
            return TakeoffQuery("count", found[0], scope="model")
        return TakeoffQuery("count", found[0], storey=names[0])
    if kind_unit == "m2" and "walls" in found:
        return TakeoffQuery("sum", "walls", storey=names[0], quantity="NetSideArea")
    if kind_unit == "m2" and "rooms" in found:
        return TakeoffQuery("storey_area", "rooms", storey=names[0])
    if kind_unit == "m3" and len(set(found) & {"walls", "slabs"}) == 1:
        return TakeoffQuery("sum", next(iter(set(found) & {"walls", "slabs"})), storey=names[0], quantity="NetVolume")
    if kind_unit == "m" and "walls" in found:
        return TakeoffQuery("sum", "walls", storey=names[0], quantity="Length")
    return None


def takeoff_category(query: TakeoffQuery) -> str:
    if query.kind == "count":
        return query.element if query.scope == "storey" else f"model_{query.element}"
    return {"room_area": "room_area", "storey_area": "storey_area"}.get(query.kind, f"{query.element}_{query.quantity}")


def space_label(space: Any) -> str:
    long_name, name = getattr(space, "LongName", None), getattr(space, "Name", None)
    if long_name and name:
        return f"{long_name} ({name})"
    return str(long_name or name or "")


class _Model:
    def __init__(self, path: str):
        ifcopenshell = importlib.import_module("ifcopenshell")
        self.element = importlib.import_module("ifcopenshell.util.element")
        unit = importlib.import_module("ifcopenshell.util.unit")
        self.file = ifcopenshell.open(path)
        self.scale = {kind: unit.calculate_unit_scale(self.file, f"{kind}UNIT") for kind in ("LENGTH", "AREA", "VOLUME")}
        self._storeys: dict[int, Any] = {}
        self._areas: dict[int, float | None] = {}

    def storey(self, entity: Any) -> Any:
        key = entity.id()
        if key not in self._storeys:
            parent = self.element.get_aggregate(entity) or self.element.get_container(entity)
            while parent is not None and not parent.is_a("IfcBuildingStorey"):
                parent = self.element.get_aggregate(parent) or self.element.get_container(parent)
            self._storeys[key] = parent
        return self._storeys[key]

    def elements(self, element: str) -> list[Any]:
        if element in _TERMINALS:
            occurrence, type_class = _TERMINALS[element]
            try:
                return list(self.file.by_type(occurrence))
            except RuntimeError:
                return [e for e in self.file.by_type("IfcFlowTerminal")
                        if (t := self.element.get_type(e)) is not None and t.is_a(type_class)]
        found = [e for c in _CLASSES[element] for e in self.file.by_type(c)]
        classes = _CLASSES[element]
        # Parts aggregated into an element of the same class are not separate elements.
        return [e for e in found if not ((p := self.element.get_aggregate(e)) is not None and any(p.is_a(c) for c in classes))]

    def quantity(self, entity: Any, name: str) -> float | None:
        scale = self.scale["LENGTH" if name == "Length" else "AREA" if name.endswith("Area") else "VOLUME"]
        for values in self.element.get_psets(entity, qtos_only=True).values():
            if isinstance(values.get(name), (int, float)):
                return values[name] * scale
        return None

    def stored_net_area(self, space: Any) -> float | None:
        stored = self.quantity(space, "NetFloorArea")
        if stored is not None:
            return stored
        # Authoring-tool property sets, e.g. ArchiCAD's "Measured Net Area".
        for values in self.element.get_psets(space, psets_only=True).values():
            for name, value in values.items():
                if isinstance(value, (int, float)) and re.search(r"net\w*\s*area", name, re.I):
                    return value * self.scale["AREA"]
        return None

    def footprint(self, space: Any) -> float | None:
        """Plan area of the body: half the total vertical projection of all faces.

        Equals the footprint for closed solids, and also covers surface models
        that have no upward-facing faces.
        """
        key = space.id()
        if key not in self._areas:
            try:
                geom = importlib.import_module("ifcopenshell.geom")
                settings = geom.settings()
                settings.set("use-world-coords", True)
                # Keep the shape alive while its geometry buffer is read.
                created = geom.create_shape(settings, space)
                verts, faces = created.geometry.verts, created.geometry.faces
                projected = 0.0
                for i in range(0, len(faces), 3):
                    a, b, c = (faces[i + j] * 3 for j in range(3))
                    projected += abs((verts[b] - verts[a]) * (verts[c + 1] - verts[a + 1])
                                     - (verts[c] - verts[a]) * (verts[b + 1] - verts[a + 1])) / 2
                self._areas[key] = projected / 2 or None
            except Exception:
                self._areas[key] = None
        return self._areas[key]

    def net_area(self, space: Any) -> tuple[float | None, str]:
        stored = self.stored_net_area(space)
        if stored is not None:
            return stored, "stored net area"
        return self.footprint(space), "geometric footprint"


@lru_cache(maxsize=8)
def _load(path: str, _mtime: int) -> _Model:
    return _Model(path)


def load_model(path: Path) -> _Model:
    return _load(str(path), path.stat().st_mtime_ns)


def answer_takeoff(path: Path, ifc_path: Path, question: Question, query: TakeoffQuery,
                   model_alias: str = "ifc") -> Prediction:
    started = time.monotonic()
    category = takeoff_category(query)
    base = dict(
        question_id=question.id or hashlib.sha256(f"{question.path}\0{question.question}".encode()).hexdigest()[:16],
        path=question.path, question=question.question, model=model_alias, model_id=ADAPTER_VERSION,
        unit=question.unit or ("count" if query.kind == "count" else "m2" if query.kind != "sum" else
                               {"NetVolume": "m3", "NetSideArea": "m2", "Length": "m"}[query.quantity]),
        category=category, expected_answer=question.answer, has_ground_truth="answer" in question.model_fields_set,
    )

    def result(status: str, **values) -> Prediction:
        return Prediction(**base, status=status,
                          usage=Usage(latency_seconds=time.monotonic() - started, estimated_cost_usd=0.0), **values)

    try:
        importlib.import_module("ifcopenshell")
    except ImportError:
        return result("unsupported", error="IFC support is optional. Install it with: python -m pip install '.[ifc]'")
    try:
        model = load_model(ifc_path)
        warnings = [f"Answered from the IFC model {ifc_path.name}, not by reading the drawing."]
        if path.suffix.lower() != ".ifc":
            warnings.append("Evidence lists IFC GlobalIds; boxes are not mapped onto the drawing yet.")
        storeys = {s.Name: s for s in model.file.by_type("IfcBuildingStorey")}
        if query.scope == "storey" and query.storey not in storeys:
            return result("needs_review", error=f"No storey named '{query.storey}' in {ifc_path.name}.",
                          warnings=warnings, source_sha256=_sha256(path))
        on_storey = lambda e: query.scope == "model" or (model.storey(e) is not None and model.storey(e).Name == query.storey)
        uncertain = False
        if query.kind == "count":
            items = [e for e in model.elements(query.element) if on_storey(e)]
            value: float = len(items)
            notes = ["" for _ in items]
            rationale = f"Counted {len(items)} {query.element} " + (f"on storey '{query.storey}'." if query.scope == "storey" else "in the whole model.")
        elif query.kind == "room_area":
            items = [s for s in model.elements("rooms") if space_label(s) == query.room and on_storey(s)]
            if len(items) != 1:
                return result("needs_review", error=f"Found {len(items)} rooms labelled '{query.room}' on '{query.storey}'.",
                              warnings=warnings, source_sha256=_sha256(path))
            area, source = model.net_area(items[0])
            if area is None:
                return result("needs_review", error="The room has neither a stored net area nor evaluable geometry.",
                              warnings=warnings, source_sha256=_sha256(path))
            footprint = model.footprint(items[0]) if source != "geometric footprint" else area
            if footprint is not None and abs(footprint - area) > 0.02 * area:
                warnings.append(f"Stored net area {area:.2f} m² differs from the geometric footprint {footprint:.2f} m².")
            value = round(area, 2)
            notes = [f"{source.capitalize()} {area:.2f} m²."]
            rationale = f"Net floor area of '{query.room}' is the {source} of its IfcSpace."
        else:
            items = [e for e in model.elements(query.element) if on_storey(e)]
            values = []
            for e in items:
                v = model.net_area(e)[0] if query.kind == "storey_area" else model.quantity(e, query.quantity)
                values.append(v)
            missing = sum(v is None for v in values)
            if missing:
                uncertain = True
                warnings.append(f"{missing} of {len(items)} {query.element} have no {'geometry' if query.kind == 'storey_area' else query.quantity}; they are left out of the sum.")
            value = round(sum(v for v in values if v is not None), 3)
            label = "footprint area" if query.kind == "storey_area" else query.quantity
            notes = [f"{label} {v:.3f}." if v is not None else f"No {label}." for v in values]
            rationale = f"Sum of {label} over {len(items)} {query.element} on storey '{query.storey}'."
        evidence = [_evidence(e, category, i + 1, note) for i, (e, note) in enumerate(zip(items, notes))]
        for item in evidence:
            item.provenance = "ifc_entity"
        if value == 0 and query.kind != "count":
            uncertain = True
            warnings.append("No quantities were found; a zero total is not reliable.")
        return result("needs_review" if uncertain else "answered",
                      answer=None if uncertain else value, proposed_answer=value, evidence=evidence,
                      rationale=rationale, warnings=warnings, source_sha256=_sha256(path))
    except Exception as exc:
        return result("error", error=f"IFC take-off failed ({type(exc).__name__}). Check the file schema and validity.")

"""Optional IFC entity queries with source GUIDs, without a VLM or fake geometry.

Counts describe *entities in this IFC file*. They do not prove that an IFC export
contains every real-world component. No geometry renderer is used in this module.
IFC4/IFC2X3 sanitary types follow buildingSMART's IfcSanitaryTerminalTypeEnum:
https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/HTML/lexical/IfcSanitaryTerminalTypeEnum.htm
In particular, TOILETPAN is a toilet fixture; WCSEAT is only its seat.
"""
from __future__ import annotations

import hashlib
import importlib
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from .schemas import Evidence, Intent, Prediction, Question, Usage

ADAPTER_VERSION = "ifc-entities-v1"

# Aliases make the native adapter independently useful to future intent mappers.
_CATEGORIES = {
    "door": "doors", "window": "windows", "floor": "floors",
    "storeys": "floors", "storey": "floors", "bedroom": "bedrooms",
    "kitchen": "kitchens", "bathroom": "bathrooms", "wc_room": "wc_rooms",
    "separate_wc": "wc_rooms", "separate_wc_rooms": "wc_rooms",
    "sauna": "saunas", "balcony": "balconies", "toilet": "toilets",
    "toilet_bowls": "toilets", "sink": "sinks", "shower_bathtub": "showers_bathtubs",
    "showers_and_bathtubs": "showers_bathtubs", "showers_and_baths": "showers_bathtubs",
    "showers_baths": "showers_bathtubs",
}
_DIRECT_CLASSES = {"doors": "IfcDoor", "windows": "IfcWindow", "floors": "IfcBuildingStorey"}
_FIXTURE_TYPES = {
    "toilets": {"TOILETPAN"},
    "sinks": {"SINK", "WASHHANDBASIN"},
    "showers_bathtubs": {"SHOWER", "BATH"},
}
_KNOWN_FIXTURE_TYPES = {
    "BATH", "BIDET", "CISTERN", "SANITARYFOUNTAIN", "SHOWER", "SINK",
    "TOILETPAN", "URINAL", "WASHHANDBASIN", "WCSEAT",
}
_SPACE_LABELS = {
    "bedrooms": {"bedroom", "master bedroom", "guest bedroom", "makuuhuone", "mh"},
    "kitchens": {"kitchen", "kitchenette", "open kitchen", "utility kitchen", "keittiö", "keittokomero", "avokeittiö", "tupakeittiö", "apukeittiö", "k", "kk", "kt"},
    "bathrooms": {"bathroom", "washroom", "shower room", "kylpyhuone", "pesuhuone", "suihkuhuone", "kh", "kph", "ph", "pesuh", "sh"},
    "wc_rooms": {"wc", "separate wc", "toilet room", "water closet", "erillinen wc", "wc tila", "wc huone"},
    "saunas": {"sauna", "s"},
    "balconies": {"balcony", "glazed balcony", "parveke", "lasitettu parveke", "parv"},
    "other": {"living room", "lounge", "hallway", "hall", "corridor", "storage", "garage", "terrace", "porch", "dining room", "utility room", "technical room", "office", "stairwell", "olohuone", "eteinen", "käytävä", "varasto", "autotalli", "terassi", "kuisti", "ruokailutila", "kodinhoitohuone", "tekninen tila", "aula", "oh", "et", "vh", "khh", "rt", "tekn"},
}


def _normal_label(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = re.sub(r"[_\-/.,:;()]", " ", text)
    # Codes such as MH1 / Bedroom 2 explicitly name the use of the space.
    text = re.sub(r"\d+", " ", text)
    return " ".join(text.split())


def _classify_space(entity: Any) -> tuple[str | None, str]:
    matches: set[str] = set()
    labels: list[str] = []
    for key in ("Name", "LongName", "ObjectType"):
        value = getattr(entity, key, None)
        if not isinstance(value, str) or not value.strip():
            continue
        labels.append(f"{key}={value}")
        normalized = _normal_label(value)
        for category, aliases in _SPACE_LABELS.items():
            if normalized in aliases:
                matches.add(category)
    # Conflicting explicit classifications are not silently resolved.
    return (next(iter(matches)) if len(matches) == 1 else None), "; ".join(labels)


def _is_a(entity: Any, ifc_class: str) -> bool:
    try:
        return bool(entity.is_a(ifc_class))
    except (AttributeError, TypeError, RuntimeError):
        return False


def _types(entity: Any) -> list[Any]:
    found = []
    # IFC4 uses IsTypedBy; IFC2X3 uses IsDefinedBy with IfcRelDefinesByType.
    for attribute in ("IsTypedBy", "IsDefinedBy"):
        for relation in getattr(entity, attribute, ()) or ():
            relating_type = getattr(relation, "RelatingType", None)
            if relating_type is not None:
                found.append(relating_type)
    return found


def _predefined_type(entity: Any) -> str | None:
    direct = getattr(entity, "PredefinedType", None)
    if direct and str(direct).upper() not in {"NOTDEFINED", "USERDEFINED"}:
        return str(direct).upper()
    inherited = {
        str(getattr(t, "PredefinedType", "")).upper()
        for t in _types(entity)
        if getattr(t, "PredefinedType", None)
        and str(getattr(t, "PredefinedType")).upper() not in {"NOTDEFINED", "USERDEFINED"}
    }
    return next(iter(inherited)) if len(inherited) == 1 else None


def _entities(model: Any, ifc_class: str) -> list[Any]:
    try:
        return list(model.by_type(ifc_class))
    except RuntimeError as exc:
        # IfcSanitaryTerminal occurrences are absent from IFC2X3's schema.
        if ifc_class == "IfcSanitaryTerminal" and "IFC2X3" in str(getattr(model, "schema", "")).upper():
            return []
        raise exc


def _step_id(entity: Any) -> str:
    try:
        return str(entity.id())
    except (AttributeError, TypeError):
        return str(id(entity))


def _unique_entities(entities: list[Any]) -> tuple[list[Any], list[str], bool]:
    output, seen, warnings = [], {}, []
    ambiguous = False
    for entity in entities:
        guid = getattr(entity, "GlobalId", None)
        step_id = _step_id(entity)
        key = ("guid", str(guid)) if guid else ("step", step_id)
        if key in seen:
            warnings.append(f"Repeated entity identity {key[1]} was counted once.")
            # Repeated references are harmless; duplicated GUIDs on distinct
            # physical entities are malformed and require checking the export.
            if seen[key] != step_id:
                ambiguous = True
                warnings.append(f"Distinct STEP entities share GlobalId={key[1]}; the IFC export needs correction.")
            continue
        seen[key] = step_id
        output.append(entity)
        if not guid:
            warnings.append(f"Entity #{step_id} lacks GlobalId; only its STEP identifier is available.")
            ambiguous = True
    return output, warnings, ambiguous


def _evidence(entity: Any, category: str, index: int, note: str = "") -> Evidence:
    guid = getattr(entity, "GlobalId", None)
    ifc_class = str(entity.is_a())
    name = str(getattr(entity, "Name", "") or "")
    label = f"{ifc_class}: {name}" if name else ifc_class
    details = f"STEP #{_step_id(entity)}; GlobalId={guid or 'missing'}. {note}".strip()
    return Evidence(
        id=f"ifc-{index:04d}", label=label[:200], bbox=None, page=1,
        category=category, provenance="ifc_entity", entity_guid=str(guid) if guid else None,
        verified=False, note=details[:1200], counted=True, uncertain=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def answer_ifc(path: Path, question: Question, intent: Intent, model_alias: str = "ifc") -> Prediction:
    """Run a bounded, deterministic class/attribute query over a local IFC file.

    No arbitrary code or model-generated IFC query is executed. Unknown room or
    sanitary classifications cause an abstention with a proposed partial count.
    Root orchestration may replace question_id with its dataset-wide row ID.
    """
    started = time.monotonic()
    category = _CATEGORIES.get(intent.category, intent.category)
    base = dict(
        question_id=question.id or hashlib.sha256(f"{question.path}\0{question.question}".encode()).hexdigest()[:16],
        path=question.path, question=question.question, model=model_alias,
        model_id=ADAPTER_VERSION, unit=question.unit or "count", category=intent.category,
        expected_answer=question.answer, has_ground_truth="answer" in question.model_fields_set,
    )

    def result(status: str, **values) -> Prediction:
        return Prediction(
            **base, status=status,
            usage=Usage(latency_seconds=time.monotonic() - started, estimated_cost_usd=0.0),
            **values,
        )

    if question.page is not None:
        return result("unsupported", error="IFC has no drawing-page indexing; remove page or query a rendered drawing instead.")
    if intent.operation != "count" or category not in set(_DIRECT_CLASSES) | set(_FIXTURE_TYPES) | (set(_SPACE_LABELS) - {"other"}):
        return result("unsupported", error="This version supports IFC entity counts only; measurements, geometry and arbitrary BIM questions are not implemented.")
    try:
        module = importlib.import_module("ifcopenshell")
    except ImportError:
        return result("unsupported", error="IFC support is optional. Install it with: python -m pip install '.[ifc]'")
    try:
        source_sha256 = _sha256(path)
        model = module.open(str(path))
        warnings = [
            "Counts cover typed entities in this IFC file, not drawing symbols; export completeness is not verified.",
            "Evidence is an IFC GlobalId/STEP reference. This version does not render or highlight 3D geometry.",
        ]
        matches: list[tuple[Any, str]] = []
        uncertain = False
        if category in _DIRECT_CLASSES:
            entities, dedup_notes, ambiguous = _unique_entities(_entities(model, _DIRECT_CLASSES[category]))
            warnings += dedup_notes
            uncertain |= ambiguous
            for entity in entities:
                if category == "floors":
                    composition = getattr(entity, "CompositionType", None)
                    if composition in {"COMPLEX", "PARTIAL"}:
                        uncertain = True
                        warnings.append(f"Storey #{_step_id(entity)} has CompositionType={composition}; excluded from the proposed count of whole storeys.")
                        continue
                matches.append((entity, f"Exact entity class query: {_DIRECT_CLASSES[category]}."))
            if category == "doors":
                warnings.append("One IfcDoor instance counts once; door-leaf geometry and any assembly grouping are not checked.")
            if category == "windows":
                warnings.append("One IfcWindow instance counts once; pane/assembly grouping follows the IFC authoring convention.")
        elif category in _FIXTURE_TYPES:
            terminals = _entities(model, "IfcSanitaryTerminal")
            # IFC2X3 stores occurrences as IfcFlowTerminal with a sanitary type.
            if "IFC2X3" in str(getattr(model, "schema", "")).upper():
                for entity in _entities(model, "IfcFlowTerminal"):
                    related = _types(entity)
                    if any(_is_a(t, "IfcSanitaryTerminalType") for t in related):
                        terminals.append(entity)
                    elif not related:
                        uncertain = True
                        warnings.append(f"Untyped IfcFlowTerminal #{_step_id(entity)} cannot be classified as sanitary or non-sanitary.")
            terminals, dedup_notes, ambiguous = _unique_entities(terminals)
            warnings += dedup_notes
            uncertain |= ambiguous
            for entity in terminals:
                terminal_type = _predefined_type(entity)
                if terminal_type not in _KNOWN_FIXTURE_TYPES:
                    uncertain = True
                    warnings.append(f"Sanitary entity #{_step_id(entity)} has no supported unambiguous PredefinedType.")
                elif terminal_type in _FIXTURE_TYPES[category]:
                    matches.append((entity, f"PredefinedType={terminal_type}."))
            if not terminals:
                uncertain = True
                warnings.append("No classified sanitary inventory is present; absence cannot establish a zero fixture count.")
            if category == "toilets":
                warnings.append("TOILETPAN is counted; WCSEAT, CISTERN, BIDET and URINAL are excluded.")
            if category == "sinks":
                warnings.append("Each SINK or WASHHANDBASIN instance counts once; a double-bowl assembly modeled as two instances needs review.")
        else:
            spaces, dedup_notes, ambiguous = _unique_entities(_entities(model, "IfcSpace"))
            warnings += dedup_notes
            uncertain |= ambiguous
            for entity in spaces:
                space_category, label_evidence = _classify_space(entity)
                if space_category is None:
                    uncertain = True
                    warnings.append(f"IfcSpace #{_step_id(entity)} has missing, conflicting or unrecognized room-use labels.")
                elif space_category == category:
                    matches.append((entity, label_evidence))
            if not spaces:
                uncertain = True
                warnings.append("No IfcSpace inventory is present; absence cannot establish a zero room count.")
            warnings.append("Rooms are classified only from explicit Name/LongName/ObjectType aliases; no geometry or room function is inferred.")

        evidence = [_evidence(entity, intent.category, index + 1, note) for index, (entity, note) in enumerate(matches)]
        count = len(evidence)
        return result(
            "needs_review" if uncertain else "answered",
            answer=None if uncertain else count, proposed_answer=count,
            evidence=evidence,
            rationale=f"Selected {count} distinct modeled entities using the {category} rule. "
            + ("The count is partial because classification or identity is incomplete." if uncertain else "Each counted entity has a source reference for review."),
            warnings=list(dict.fromkeys(warnings)), source_sha256=source_sha256,
        )
    except Exception as exc:
        # Entity/parser errors are isolated to this dataset row. No model, input
        # file bytes, credentials or environment values are included in errors.
        return result("error", error=f"IFC query failed ({type(exc).__name__}). Check the file schema and validity.")

"""Model-independent evidence fusion and deterministic counting."""
from __future__ import annotations

import re
from pathlib import Path

from .cache import ExtractionCache, digest
from .data import SUPPORTED_EXTENSIONS, DatasetIndex, sha256_file, stable_id
from .dxf import answer_dxf, dxf_category, parse_query, sibling_dxf
from .ifc_takeoff import answer_takeoff, parse_takeoff, sibling_ifc
from . import inventory
from .imaging import make_views, render_pages
from .intents import resolve_intent
from .prompts import PROMPT_VERSION, build_prompt
from .providers import OpenAICompatibleProvider, ProviderError, parse_extraction, parse_inventory
from .schemas import AppConfig, DrawingInventory, Evidence, Prediction, Question, Usage


def prediction_fingerprint(config: AppConfig, model_alias: str) -> str:
    return digest({"pipeline": config.pipeline.model_dump(),
                   "model": config.models[model_alias].model_dump(),
                   "prompt_version": PROMPT_VERSION, "schema": "1.0"})


def add_usage(total: Usage, part: Usage) -> None:
    for key in ("input_tokens", "output_tokens", "requests", "retries", "cached_requests", "latency_seconds"):
        setattr(total, key, getattr(total, key) + getattr(part, key))
    total.usage_reported = total.usage_reported and part.usage_reported
    if part.requests:
        if part.estimated_cost_usd is None:
            total.estimated_cost_usd = None
        elif total.estimated_cost_usd is not None:
            total.estimated_cost_usd += part.estimated_cost_usd


def iou(a, b):
    x1, y1, x2, y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    intersection = max(0,x2-x1)*max(0,y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0


def fuse_evidence(items: list[Evidence], threshold: float) -> tuple[list[Evidence], list[str]]:
    def kind(item):
        if item.category not in {"showers_bathtubs", "general"}:
            return item.category
        label = re.sub(r"[^a-z]+", " ", item.label.lower()).strip()
        if "shower" in label and "bath" not in label:
            return "shower"
        if "bath" in label and "shower" not in label:
            return "bathtub"
        return label
    fused = []
    warnings = []
    for item in items:
        matched = next((existing for existing in fused if existing.page == item.page
                        and existing.bbox is not None and item.bbox is not None
                        and not set(existing.view_ids).intersection(item.view_ids)
                        and kind(existing) == kind(item)
                        and iou(existing.bbox,item.bbox) >= threshold), None)
        if matched is None:
            # Moderately overlapping boxes from separate views may be duplicate instances.
            for existing in fused:
                if (existing.page == item.page and existing.bbox and item.bbox
                        and 0.2 < iou(existing.bbox,item.bbox)):
                    existing.uncertain = item.uncertain = True
                    warnings.append("Overlapping entities may be duplicates or different fixtures; boxes were retained for review")
            fused.append(item.model_copy(deep=True))
        else:
            if matched.counted != item.counted:
                matched.uncertain = True
                warnings.append("Views disagree on whether an entity should count")
            matched.counted = matched.counted or item.counted
            matched.uncertain = matched.uncertain or item.uncertain
            matched.view_ids = sorted(set(matched.view_ids + item.view_ids))
    for index, item in enumerate(fused, 1):
        item.id = f"e{index}"
    return fused, list(dict.fromkeys(warnings))


def flag_repeated_boxes(items: list[Evidence], minimum: int = 4) -> list[str]:
    """Mark runs of identical boxes stamped along one line as uncertain.

    Four or more counted boxes with the same label, note and exact size, sharing
    one x or y coordinate, are typical of a model enumerating coordinates rather
    than localizing drawn symbols.
    """
    groups: dict[tuple, list[Evidence]] = {}
    for item in items:
        if item.bbox is None or not item.counted:
            continue
        x1, y1, x2, y2 = item.bbox
        identity = (item.page, item.label.casefold(), item.note.casefold(), round(x2 - x1, 3), round(y2 - y1, 3))
        groups.setdefault(identity + ("column", round(x1, 3)), []).append(item)
        groups.setdefault(identity + ("row", round(y1, 3)), []).append(item)
    flagged = {id(m) for members in groups.values() if len(members) >= minimum for m in members}
    for item in items:
        if id(item) in flagged:
            item.uncertain = True
    if not flagged:
        return []
    return [f"{len(flagged)} identical boxes are stamped along one line; "
            "they may be enumerated coordinates rather than localized objects"]


class Pipeline:
    def __init__(self, config: AppConfig, data_root: Path, output: Path, *, provider_factory=None):
        self.config = config
        self.index = DatasetIndex(data_root)
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache = ExtractionCache(Path(config.pipeline.cache_dir).expanduser())
        self.provider_factory = provider_factory or OpenAICompatibleProvider
        self._failed_inventories: dict[str, str] = {}

    def predict(self, question: Question, model_alias: str) -> Prediction:
        intent = resolve_intent(question)
        model_config = self.config.models[model_alias]
        fingerprint = prediction_fingerprint(self.config, model_alias)
        result = Prediction(question_id=question.id or stable_id(question), path=question.path,
                            question=question.question, page=question.page, model=model_alias,
                            model_id=model_config.model, status="error", unit=question.unit,
                            category=intent.category, config_sha256=fingerprint,
                            usage=Usage(estimated_cost_usd=0))
        provider = None
        try:
            path = self.index.resolve(question.path)
            result.source_sha256 = sha256_file(path)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                result.status = "unsupported"
                result.error = f"Unsupported file format {path.suffix}. Export CAD/vector formats to PDF or PNG first."
                return result
            takeoff = parse_takeoff(question.question, question.unit)
            if path.suffix.lower() == ".ifc":
                if takeoff is not None and question.page is None:
                    native = answer_takeoff(path, path, question, takeoff, model_alias)
                else:
                    from .ifc import answer_ifc
                    native = answer_ifc(path, question, intent, model_alias)
                native.question_id = result.question_id
                native.page = question.page
                native.source_sha256 = result.source_sha256
                native.config_sha256 = fingerprint
                return native
            pages = render_pages(path, self.output / "assets", result.source_sha256,
                                 self.config.pipeline, question.page)
            result.page_images = {str(p.number): str(p.path.relative_to(self.output)) for p in pages}
            ifc = sibling_ifc(path) if takeoff is not None and self.config.pipeline.use_ifc_sibling else None
            if ifc is not None:
                native = answer_takeoff(path, ifc, question, takeoff, model_alias)
                native.question_id = result.question_id
                native.source_sha256 = result.source_sha256
                native.config_sha256 = fingerprint
                native.page_images = result.page_images
                return native
            dxf = sibling_dxf(path) if self.config.pipeline.use_dxf_sibling else None
            query = parse_query(question.question) if dxf else None
            if query is not None:
                native = answer_dxf(path, dxf, question, query, dxf_category(query), model_alias)
                native.question_id = result.question_id
                native.source_sha256 = result.source_sha256
                native.config_sha256 = fingerprint
                native.page_images = result.page_images
                return native
            if self.config.pipeline.strategy == "per_drawing":
                selector = inventory.select(question, intent, path)
                if selector is not None and self._answer_from_inventory(result, pages, selector, model_config):
                    return result
            settings = self.config.pipeline
            if settings.mode == "direct" or intent.operation == "general":
                settings = settings.model_copy(update={"views": "overview"})
            if intent.operation == "general" and len(pages) > 1:
                result.status = "unsupported"
                result.error = "General multi-page synthesis is not implemented; specify a 1-based page in the question"
                return result
            views = make_views(pages, self.output / "views", settings, intent.category)
            system, prompt = build_prompt(intent, settings.mode, question.question, model_config.bbox_scale)
            provider = self.provider_factory(model_config)
            extracted = []
            raw_evidence = []
            stated_count_mismatch = False
            for view in views:
                # Keep the visible question/answer labels out of the canonical count prompt.
                key = digest({"model": model_config.model_dump(), "system": system,
                              "prompt": prompt, "image": sha256_file(view.path),
                              "version": PROMPT_VERSION})
                with self.cache.locked(key):
                    output = self.cache.get(key)
                    if output is not None:
                        result.usage.cached_requests += 1
                    else:
                        response = provider.generate(system, prompt, [view.path])
                        add_usage(result.usage, response.usage)
                        output = parse_extraction(response.content, model_config.bbox_scale)
                        self.cache.put(key, output)
                extracted.append(output)
                if intent.operation == "count" and output.answer is not None:
                    if (isinstance(output.answer, bool) or not isinstance(output.answer, int)
                            or output.answer != sum(c.counted and not c.uncertain for c in output.candidates)):
                        stated_count_mismatch = True
                        result.warnings.append("A view's stated count differs from its localized inventory")
                for candidate in output.candidates:
                    raw_evidence.append(Evidence(**{**candidate.model_dump(), "bbox": view.to_page(candidate.bbox)},
                                                  id="pending", page=view.page, category=intent.category,
                                                  view_ids=[view.id]))
                result.warnings.extend(output.warnings)
            result.evidence, fusion_warnings = fuse_evidence(raw_evidence, settings.dedup_iou)
            result.warnings.extend(fusion_warnings)
            result.warnings.extend(flag_repeated_boxes(result.evidence))
            result.rationale = "\n".join(dict.fromkeys(e.rationale for e in extracted if e.rationale))[:8000]
            complete = all(e.complete for e in extracted)
            ambiguous = stated_count_mismatch or any(e.uncertain for e in result.evidence)
            if intent.operation == "count":
                proposed = sum(e.counted for e in result.evidence)
                if settings.mode == "direct":
                    answers = [e.answer for e in extracted]
                    if any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in answers):
                        raise ValueError("Direct count must return a nonnegative JSON integer")
                    result.proposed_answer = sum(answers)
                    if result.proposed_answer != proposed:
                        ambiguous = True
                        result.warnings.append("Model's stated count differs from the number of localized counted entities")
                else:
                    result.proposed_answer = proposed
                if len(pages) > 1:
                    ambiguous = True
                    result.warnings.append("Multi-page count is a count of depicted occurrences; repeated entities across sheets need review. Select a page for a scoped count.")
                if not complete:
                    result.warnings.append("Model did not declare a complete visual inventory")
                if proposed == 0:
                    result.warnings.append("Zero means no matching entities were identified; absence has not been independently verified")
                result.status = "answered" if complete and not ambiguous else "needs_review"
                result.answer = result.proposed_answer if result.status == "answered" else None
            else:
                result.proposed_answer = extracted[0].answer
                result.status = "answered" if complete and not ambiguous and result.evidence and extracted[0].answer is not None else "needs_review"
                result.answer = result.proposed_answer if result.status == "answered" else None
                if not result.evidence:
                    result.warnings.append("No visual evidence was localized for this answer")
            result.warnings.append("Visual evidence is model-predicted and has not been human verified")
            result.warnings = list(dict.fromkeys(result.warnings))
            return result
        except ProviderError as exc:
            if getattr(exc, "usage", None):
                add_usage(result.usage, exc.usage)
            result.error = str(exc)
            result.status = "error"
            return result
        except (ValueError, OSError, RuntimeError) as exc:
            result.status = "error"
            result.error = str(exc)
            return result
        finally:
            if provider is not None and hasattr(provider, "close"):
                provider.close()

    def _answer_from_inventory(self, result: Prediction, pages, selector: inventory.Selector, model_config) -> bool:
        """Fill `result` from the shared page inventory; False means use the per-question route."""
        group = inventory.group_of(selector)
        settings = self.config.pipeline.model_copy(update={"views": "overview"})
        model_config = model_config.model_copy(update={
            "max_tokens": settings.inventory_max_tokens,
            "extra_body": {**model_config.extra_body, **settings.inventory_extra_body}})
        views = make_views(pages, self.output / "views", settings, "inventory")
        system, prompt = inventory.build_inventory_prompt(group, model_config.bbox_scale)
        entries, complete, provider, warnings = [], True, None, []
        try:
            for view in views:
                key = digest({"model": model_config.model_dump(), "system": system, "prompt": prompt,
                              "image": sha256_file(view.path), "version": inventory.INVENTORY_VERSION})
                # Concurrent questions about the same page wait here and reuse the first response.
                with self.cache.locked(key):
                    if key in self._failed_inventories:
                        result.warnings.append(self._failed_inventories[key])
                        return False
                    found = self.cache.get(key, DrawingInventory)
                    if found is not None:
                        result.usage.cached_requests += 1
                    else:
                        try:
                            provider = provider or self.provider_factory(model_config)
                            response = provider.generate(system, prompt, [view.path])
                            add_usage(result.usage, response.usage)
                            found = parse_inventory(response.content, model_config.bbox_scale)
                        except (ProviderError, ValueError) as exc:
                            if getattr(exc, "usage", None):
                                add_usage(result.usage, exc.usage)
                            # Remembered for this run so the page's other questions do not repeat the request.
                            self._failed_inventories[key] = f"Shared '{group}' inventory failed ({exc}); answered per question"
                            result.warnings.append(self._failed_inventories[key])
                            return False
                        self.cache.put(key, found)
                complete = complete and found.complete
                warnings.extend(found.warnings)
                entries.extend((view.page, view.id, item.model_copy(update={"bbox": view.to_page(item.bbox)}))
                               for item in found.items if item.kind in inventory.GROUPS[group])
        finally:
            if provider is not None and hasattr(provider, "close"):
                provider.close()
        result.route = "vlm-inventory"
        result.category = selector.category
        result.warnings.extend(warnings)
        outcome = inventory.answer(selector, entries, complete)
        result.evidence = outcome["evidence"]
        result.warnings.extend(outcome["warnings"])
        if selector.operation == "count":
            result.warnings.extend(flag_repeated_boxes(result.evidence))
            if len(pages) > 1:
                result.warnings.append("Multi-page count is a count of depicted occurrences; repeated entities across sheets need review. Select a page for a scoped count.")
        review = outcome["review"] or any(e.uncertain for e in result.evidence) or (selector.operation == "count" and len(pages) > 1)
        result.rationale = outcome["rationale"]
        result.proposed_answer = outcome["proposed"]
        result.status = "needs_review" if review else "answered"
        result.answer = result.proposed_answer if result.status == "answered" else None
        result.warnings.append("Visual evidence is model-predicted and has not been human verified")
        result.warnings = list(dict.fromkeys(result.warnings))
        return True


def attach_ground_truth(result: Prediction, question: Question) -> Prediction:
    # This happens strictly after inference; expected answers never enter prompts/cache.
    result.expected_answer = question.answer
    result.has_ground_truth = "answer" in question.model_fields_set and question.answer is not None
    return result

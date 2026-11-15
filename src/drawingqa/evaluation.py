"""Transparent, deterministic metrics for labeled and unlabeled DrawingQA runs.

These metrics score final answers only. They do not infer localization quality
from an answer being correct, and proposed answers never count as abstentions'
final answers. All accuracy values are proportions in [0, 1].
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .schemas import Prediction


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def exact_match(actual: Any, expected: Any) -> bool:
    """Numerically equal numbers match; booleans cannot impersonate 0 or 1."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is bool and type(expected) is bool and actual == expected
    if _number(actual) and _number(expected):
        return actual == expected
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()
    return type(actual) is type(expected) and actual == expected


MEASUREMENT_UNITS = {"m", "m2", "m²", "m3", "m³"}
MEASUREMENT_TOLERANCE = 0.02


def answer_matches(actual: Any, expected: Any, unit: str | None) -> bool:
    """Counts and text match exactly; areas, lengths and volumes within 2%, as the dataset is scored."""
    if unit in MEASUREMENT_UNITS and _number(actual) and _number(expected):
        return abs(actual - expected) <= MEASUREMENT_TOLERANCE * abs(expected)
    return exact_match(actual, expected)


def _answered(p: Prediction) -> bool:
    return p.status == "answered" and p.answer is not None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _proposal_metrics(rows: list[Prediction]) -> dict[str, Any]:
    """Diagnostic only: proposed values are not accepted final answers."""
    eligible = [p for p in rows if p.status in {"answered", "needs_review"}]
    proposed = [p for p in eligible if p.proposed_answer is not None]
    labeled = [p for p in eligible if p.has_ground_truth]
    scored = [p for p in proposed if p.has_ground_truth]
    correct = sum(answer_matches(p.proposed_answer, p.expected_answer, p.unit) for p in scored)
    numeric = [p for p in scored if _number(p.proposed_answer) and _number(p.expected_answer)]
    errors = [abs(p.proposed_answer - p.expected_answer) for p in numeric]
    return {
        "eligible_questions": len(eligible),
        "excluded_error_or_unsupported": len(rows) - len(eligible),
        "proposed_questions": len(proposed),
        "proposal_coverage": _ratio(len(proposed), len(eligible)),
        "eligible_labeled_questions": len(labeled),
        "proposed_labeled_questions": len(scored),
        "correct_proposals": correct,
        "exact_accuracy_all_eligible_labeled": _ratio(correct, len(labeled)),
        "exact_accuracy_proposed_only": _ratio(correct, len(scored)),
        "numeric_mae_proposed_only": mean(errors) if errors else None,
        "numeric_mae_samples": len(errors),
        "note": "Diagnostic proposed values only, not accepted final answers. Error/unsupported rows are excluded; missing proposals remain in the all-eligible-labeled denominator. Read proposal accuracy with proposal coverage and final-answer accuracy.",
    }


def _usage_metrics(rows: list[Prediction]) -> dict[str, Any]:
    usage_known = all(p.usage.usage_reported for p in rows)
    costs = [p.usage.estimated_cost_usd for p in rows]
    latencies = [p.usage.latency_seconds for p in rows]
    return {
        "requests": sum(p.usage.requests for p in rows),
        "retries": sum(p.usage.retries for p in rows),
        "cached_requests": sum(p.usage.cached_requests for p in rows),
        "reported_input_tokens": sum(p.usage.input_tokens for p in rows),
        "reported_output_tokens": sum(p.usage.output_tokens for p in rows),
        "usage_complete": usage_known,
        "estimated_cost_usd": sum(costs) if costs and usage_known and all(c is not None for c in costs) else None,
        "known_cost_subtotal_usd": sum(c for c in costs if c is not None),
        "cost_complete": bool(costs) and usage_known and all(c is not None for c in costs),
        "latency_mean_seconds": mean(latencies) if latencies else None,
        "latency_p50_seconds": _percentile(latencies, 0.5),
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "latency_sum_seconds": sum(latencies),
        "latency_note": "Per-question sum of provider API elapsed time including retry waits. Excludes image preprocessing, report generation and other local work; not concurrent batch wall time. Cache hits contribute zero API seconds.",
    }


def _metrics(rows: list[Prediction]) -> dict[str, Any]:
    labeled = [p for p in rows if p.has_ground_truth]
    answered = [p for p in rows if _answered(p)]
    scored = [p for p in labeled if _answered(p)]
    correct = sum(answer_matches(p.answer, p.expected_answer, p.unit) for p in scored)
    numeric = [p for p in scored if _number(p.answer) and _number(p.expected_answer)]
    errors = [abs(p.answer - p.expected_answer) for p in numeric]
    return {
        "questions": len(rows),
        "labeled_questions": len(labeled),
        "answered_questions": len(answered),
        "answered_labeled_questions": len(scored),
        "correct_answers": correct,
        "answer_coverage": _ratio(len(answered), len(rows)),
        "exact_accuracy_all_labeled": _ratio(correct, len(labeled)),
        "exact_accuracy_answered_only": _ratio(correct, len(scored)),
        "numeric_mae_answered_only": mean(errors) if errors else None,
        "numeric_mae_samples": len(errors),
        "status_counts": dict(sorted(Counter(p.status for p in rows).items())),
        "proposal_diagnostics": _proposal_metrics(rows),
    }


def _source(path: str) -> str:
    parts = Path(path.replace("\\", "/")).parts
    if parts and parts[0] == "data":
        parts = parts[1:]
    return parts[0] if len(parts) > 1 else "root"


def _fact_key(p: Prediction) -> tuple:
    # A generic answer may concern different quantities in the same image.
    # Only canonical count intents may share a visual fact across translations.
    if p.unit == "count" and p.category not in {"general", "unknown", ""}:
        return (p.model, p.source_sha256 or p.path, getattr(p, "page", None), p.category, p.config_sha256)
    return (p.model, p.question_id)


def summarize(predictions: list[Prediction]) -> dict[str, Any]:
    summary = _metrics(predictions)
    grouped: dict[str, dict[str, Any]] = {}
    for dimension, getter in (("category", lambda p: p.category), ("source", lambda p: _source(p.path)), ("model", lambda p: p.model)):
        groups: dict[str, list[Prediction]] = defaultdict(list)
        for p in predictions:
            groups[getter(p)].append(p)
        grouped[dimension] = {key: _metrics(value) for key, value in sorted(groups.items())}
        if dimension == "model":
            for key, rows in groups.items():
                grouped[dimension][key]["usage"] = _usage_metrics(rows)
    summary["by_category"] = grouped["category"]
    summary["by_source"] = grouped["source"]
    summary["by_model"] = grouped["model"]
    counts = [p for p in predictions if p.unit == "count" and p.has_ground_truth and _number(p.expected_answer)]
    summary["count_zero"] = _metrics([p for p in counts if p.expected_answer == 0])
    summary["count_nonzero"] = _metrics([p for p in counts if p.expected_answer != 0])
    summary["usage"] = _usage_metrics(predictions)
    answered = [p for p in predictions if _answered(p)]
    summary["evidence"] = {
        "answered_with_evidence": sum(bool(p.evidence) for p in answered),
        "answered_with_evidence_fraction": _ratio(sum(bool(p.evidence) for p in answered), len(answered)),
        "human_verified_entities": sum(e.verified for p in predictions for e in p.evidence),
        "localization_accuracy": None,
        "note": "Evidence presence is not localization correctness. Zero counts may legitimately have no object boxes. Human review or annotated boxes are required to score evidence accuracy.",
    }
    facts: dict[tuple, list[Prediction]] = defaultdict(list)
    for p in predictions:
        facts[_fact_key(p)].append(p)
    labeled_facts = 0
    correct_facts = 0
    conflicting_labels = 0
    repeated = 0
    consistent = 0
    for rows in facts.values():
        labeled = [p for p in rows if p.has_ground_truth]
        if labeled:
            if not all(exact_match(p.expected_answer, labeled[0].expected_answer) for p in labeled):
                conflicting_labels += 1
            else:
                labeled_facts += 1
                # Conservative: all supplied wordings must produce the right answer.
                correct_facts += all(_answered(p) and answer_matches(p.answer, p.expected_answer, p.unit) for p in labeled)
        if len(rows) > 1:
            repeated += 1
            consistent += all(_answered(p) and exact_match(p.answer, rows[0].answer) for p in rows)
    summary["unique_facts"] = {
        "facts": len(facts),
        "labeled_facts": labeled_facts,
        "correct_in_all_labeled_wordings": correct_facts,
        "exact_accuracy_all_wordings": _ratio(correct_facts, labeled_facts),
        "conflicting_ground_truth_groups": conflicting_labels,
        "repeated_wording_groups": repeated,
        "answered_consistently_groups": consistent,
        "wording_consistency": _ratio(consistent, repeated),
        "note": "Canonical count intents are grouped by model, image content/path, requested page, category and configuration. This includes bilingual pairs; agreement is not correctness and may result from shared cached extraction.",
    }
    summary["metric_notes"] = [
        "Abstentions, errors and unsupported cases remain in the all-labeled accuracy denominator.",
        "No semantic judge is used: text answers match after trimming outer whitespace; counts match exactly; areas, lengths and volumes match within 2%; booleans remain distinct from numbers.",
        "MAE covers only answered numeric targets and must be interpreted with coverage.",
        "Model-proposed answers for needs_review are not scored as final answers; proposal_diagnostics separately measures proposed values and excludes errors/unsupported cases.",
        "Missing pricing or unreported usage is unknown, not free. Provider retries can add unreported charges.",
    ]
    return summary


def read_predictions(path: Path) -> list[Prediction]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(Prediction.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid prediction at {path.name}:{line_number}: {exc}") from exc
    return rows


def _csv_cell(value: Any) -> Any:
    # Keep text originating in datasets from becoming spreadsheet formulas.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "\'" + value
    return value


def compare_runs(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Compare paired, identity-checked questions; keep unmatched rows visible."""
    if len(run_dirs) < 2:
        raise ValueError("compare requires at least two run directories")
    groups: dict[str, dict[str, Prediction]] = {}
    warnings: list[str] = []
    for index, directory in enumerate(run_dirs, 1):
        directory = Path(directory)
        by_model: dict[str, list[Prediction]] = defaultdict(list)
        for row in read_predictions(directory / "predictions.jsonl"):
            by_model[row.model].append(row)
        if not by_model:
            warnings.append(f"Run {directory.name} contains no predictions.")
        for model, rows in sorted(by_model.items()):
            label = f"{index}:{directory.name}/{model}"
            indexed: dict[str, Prediction] = {}
            for row in rows:
                if row.question_id in indexed:
                    raise ValueError(f"Duplicate question_id {row.question_id!r} in {label}")
                indexed[row.question_id] = row
            groups[label] = indexed
    if len(groups) < 2:
        raise ValueError("comparison needs at least two non-empty model/run groups")
    sets = [set(rows) for rows in groups.values()]
    common = set.intersection(*sets)
    all_ids = set.union(*sets)
    if any(ids != sets[0] for ids in sets[1:]):
        warnings.append("Question sets differ. Paired metrics use only the shared question-ID intersection; full-run metrics are also retained.")
    rejected: list[str] = []
    for qid in sorted(common):
        rows = [g[qid] for g in groups.values()]
        first = rows[0]
        # Same ID alone is not sufficient: compare the question, selected page,
        # source bytes if both known, and ground truth when supplied.
        incompatible = any(
            p.question != first.question
            or getattr(p, "page", None) != getattr(first, "page", None)
            or (p.source_sha256 and first.source_sha256 and p.source_sha256 != first.source_sha256)
            or (not (p.source_sha256 and first.source_sha256) and p.path != first.path)
            or (p.has_ground_truth != first.has_ground_truth)
            or (p.has_ground_truth and not exact_match(p.expected_answer, first.expected_answer))
            for p in rows[1:]
        )
        if incompatible:
            rejected.append(qid)
    common.difference_update(rejected)
    if rejected:
        warnings.append(f"Excluded {len(rejected)} shared IDs with conflicting source/question/page/ground-truth identity.")
    if not common:
        warnings.append("No compatible shared questions; paired accuracy is unavailable.")
    result = {
        "schema_version": "1.0",
        "shared_questions": len(common),
        "union_questions": len(all_ids),
        "excluded_identity_conflicts": rejected,
        "warnings": warnings,
        "runs": {
            label: {
                "full": summarize(list(rows.values())),
                "paired": summarize([rows[qid] for qid in sorted(common)]),
                "unpaired_questions": len(set(rows) - common),
            }
            for label, rows in groups.items()
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "scope", "questions", "labeled", "accuracy_all_labeled", "accuracy_answered_only", "coverage", "numeric_mae", "cost_usd", "latency_p50_seconds", "proposal_accuracy_all_eligible_labeled", "proposal_accuracy_proposed_only", "proposal_coverage", "proposal_numeric_mae"])
        for label, metrics in result["runs"].items():
            for scope in ("full", "paired"):
                metric = metrics[scope]
                writer.writerow([label, scope, metric["questions"], metric["labeled_questions"], metric["exact_accuracy_all_labeled"], metric["exact_accuracy_answered_only"], metric["answer_coverage"], metric["numeric_mae_answered_only"], metric["usage"]["estimated_cost_usd"], metric["usage"]["latency_p50_seconds"], metric["proposal_diagnostics"]["exact_accuracy_all_eligible_labeled"], metric["proposal_diagnostics"]["exact_accuracy_proposed_only"], metric["proposal_diagnostics"]["proposal_coverage"], metric["proposal_diagnostics"]["numeric_mae_proposed_only"]])
    with (output_dir / "paired_answers.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        labels = list(groups)
        writer.writerow(["question_id", "path", "question", "expected_answer"] + [f"{label}:{field}" for label in labels for field in ("status", "answer", "correct")])
        for qid in sorted(common):
            first = groups[labels[0]][qid]
            row: list[Any] = [qid, first.path, first.question, first.expected_answer if first.has_ground_truth else None]
            for label in labels:
                p = groups[label][qid]
                row.extend([p.status, p.answer, _answered(p) and answer_matches(p.answer, p.expected_answer, p.unit) if p.has_ground_truth else None])
            writer.writerow([_csv_cell(value) for value in row])
    return result

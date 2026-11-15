"""Dataset loading and unambiguous, sandboxed file resolution."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath

from .schemas import Question

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf", ".ifc"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_id(question: Question) -> str:
    # Labels are deliberately excluded. Editing answers cannot affect inference.
    payload = [question.path, question.question, question.page]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()[:24]


def load_questions(path: Path) -> list[Question]:
    questions = []
    seen = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                question = Question.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid question at {path}:{line_no}: {exc}") from exc
            question.id = question.id or stable_id(question)
            if question.id in seen:
                raise ValueError(f"Duplicate question id {question.id!r} at line {line_no}; give duplicate rows unique ids")
            seen.add(question.id)
            questions.append(question)
    if not questions:
        raise ValueError("The questions file is empty")
    return questions


class DatasetIndex:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Data root does not exist: {self.root}")
        self.by_basename = defaultdict(list)
        self.files = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.resolve().is_relative_to(self.root):
                self.files.append(path)
                self.by_basename[path.name].append(path)

    def resolve(self, reference: str) -> Path:
        # Windows-style relative paths are accepted; absolute paths and traversal are not.
        normalized = reference.replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts or ":" in normalized or not pure.parts:
            raise ValueError("Question path must be relative to data root, without '..' or drive prefixes")
        parts = list(pure.parts)
        if parts and parts[0] == self.root.name and len(parts) > 1:
            parts = parts[1:]
        candidate = (self.root / Path(*parts)).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("Path escapes the data root")
        if candidate.is_file():
            return candidate
        # Only bare basenames get recursive matching. Typos in explicit folders fail.
        if len(parts) == 1:
            matches = self.by_basename.get(parts[0], [])
            if len(matches) == 1:
                matched = matches[0].resolve()
                if not matched.is_relative_to(self.root) or not matched.is_file():
                    raise ValueError("Indexed file changed or escapes the data root")
                return matched
            if len(matches) > 1:
                choices = ", ".join(str(p.relative_to(self.root)) for p in matches[:8])
                raise ValueError(f"Ambiguous basename {parts[0]!r}; use a relative path: {choices}")
        raise FileNotFoundError(f"Drawing not found beneath data root: {reference}")

    def summary(self) -> dict:
        formats = defaultdict(int)
        sources = defaultdict(int)
        for path in self.files:
            formats[path.suffix.lower() or "[no extension]"] += 1
            rel = path.relative_to(self.root)
            sources[rel.parts[0] if len(rel.parts) > 1 else "[root]"] += 1
        return {
            "root": str(self.root), "files": len(self.files),
            "formats": dict(formats), "sources": dict(sources),
            "supported_files": sum(n for ext, n in formats.items() if ext in SUPPORTED_EXTENSIONS),
            "ambiguous_basenames": {k: [str(p.relative_to(self.root)) for p in v]
                                    for k, v in self.by_basename.items() if len(v) > 1},
        }

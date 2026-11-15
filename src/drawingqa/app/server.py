"""PlanSight live demo: a two-tab web app over the DrawingQA pipeline.

Live Blueprint runs the pipeline on any uploaded or dataset drawing. Benchmark reads saved
runs and re-runs single dataset questions. Ground truth is attached after inference only.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from ..data import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS, DatasetIndex, sha256_file
from ..evaluation import answer_matches, read_predictions
from ..imaging import render_pages, rgb
from ..inventory import GROUPS
from ..pipeline import Pipeline, add_usage
from ..schemas import AppConfig, Question, Usage

STATIC = Path(__file__).with_name("static")
UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}
SERVABLE = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_NAME = re.compile(r"^[0-9a-f]{64}\.(png|jpe?g|webp|pdf)$")
PREFERRED_SAMPLES = ["cubicasa5k/high_quality-4017.png", "swiss_dwellings/site1021_bldg2101_floor9477.pdf",
                     "cubicasa5k/colorful-10052.png"]
SUGGESTED_QUESTIONS = [
    "How many doors are shown on this floor plan?",
    "Kuinka monta ikkunaa pohjapiirustuksessa on?",
    "How many toilet bowls (WC seats) are drawn on this floor plan?",
    "Kuinka monta makuuhuonetta (MH) pohjapiirustuksessa on?",
    "How many rooms are on this floor plan?",
]


def route_of(p: dict) -> str:
    model_id = str(p.get("model_id") or "")
    if model_id.startswith("dxf-"):
        return "DXF"
    if model_id.startswith("ifc-takeoff"):
        return "IFC take-off"
    if model_id.startswith("ifc-"):
        return "IFC"
    return "Nemotron" if "nemotron" in model_id.lower() else "Vision model"


def is_correct(p: dict) -> bool | None:
    if not p.get("has_ground_truth"):
        return None
    return p.get("status") == "answered" and p.get("answer") is not None and \
        answer_matches(p["answer"], p.get("expected_answer"), p.get("unit"))


def _inside(base: Path, relative: str, allowed_roots: set[str]) -> Path:
    pure = PurePosixPath(relative)
    if "\\" in relative or ":" in relative or pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise HTTPException(404)
    if pure.parts[0] not in allowed_roots or pure.suffix.lower() not in SERVABLE:
        raise HTTPException(404)
    path = base.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        raise HTTPException(404) from None
    if not path.is_file():
        raise HTTPException(404)
    return path


class QuestionIndex:
    """JSONL questions grouped by drawing, keyed by the path relative to the data root."""

    def __init__(self, path: Path | None, index: DatasetIndex):
        self.by_path: dict[str, list[dict]] = defaultdict(list)
        if path is None or not path.is_file():
            return
        resolved: dict[str, str | None] = {}
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reference, text = row.get("path"), row.get("question")
                if not isinstance(reference, str) or not isinstance(text, str):
                    continue
                if reference not in resolved:
                    try:
                        resolved[reference] = index.resolve(reference).relative_to(index.root).as_posix()
                    except (ValueError, OSError):
                        resolved[reference] = None
                key = resolved[reference]
                if key:
                    self.by_path[key].append({"question": text, "answer": row.get("answer"), "unit": row.get("unit"),
                                              "type": row.get("type"), "page": row.get("page")})

    def find(self, path: str, question: str) -> dict | None:
        wanted = question.strip()
        return next((q for q in self.by_path.get(path, []) if q["question"].strip() == wanted), None)


class RunStore:
    """Saved evaluation runs, loaded lazily and indexed by drawing."""

    def __init__(self, runs_dir: Path, index: DatasetIndex, exclude: set[str]):
        self.runs_dir = runs_dir
        self.index = index
        self.exclude = exclude
        self._loaded: dict[str, dict] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        if not self.runs_dir.is_dir():
            return []
        return sorted(d.name for d in self.runs_dir.iterdir()
                      if d.is_dir() and d.name not in self.exclude
                      and (d / "summary.json").is_file() and (d / "predictions.jsonl").is_file())

    def _relative(self, reference: str) -> str | None:
        try:
            return self.index.resolve(reference).relative_to(self.index.root).as_posix()
        except (ValueError, OSError):
            return None

    def load(self, name: str) -> dict:
        if name not in self.names():
            raise HTTPException(404, f"Unknown run {name!r}")
        with self._lock:
            if name in self._loaded:
                return self._loaded[name]
            directory = self.runs_dir / name
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            by_image: dict[str, list[dict]] = defaultdict(list)
            routes: dict[str, Counter] = defaultdict(Counter)
            relative_of: dict[str, str | None] = {}
            zero_token = 0
            rows = [p.model_dump(mode="json") for p in read_predictions(directory / "predictions.jsonl")]
            for p in rows:
                if p["path"] not in relative_of:
                    relative_of[p["path"]] = self._relative(p["path"])
                key = relative_of[p["path"]]
                p["route_label"] = route_of(p)
                p["correct"] = is_correct(p)
                p["page_urls"] = {page: f"/api/runs/{name}/files/{image}" for page, image in p["page_images"].items()}
                if p["route_label"] not in {"Nemotron", "Vision model"} and p["status"] in {"answered", "needs_review"}:
                    zero_token += 1
                if key:
                    by_image[key].append(p)
                    routes[key.split("/")[0]][p["route_label"]] += 1
            run = {"name": name, "dir": directory, "summary": summary, "by_image": by_image,
                   "routes": {k: dict(v) for k, v in routes.items()},
                   "zero_token_share": zero_token / len(rows) if rows else None,
                   "manifest_models": json.loads((directory / "manifest.json").read_text()).get("models")
                   if (directory / "manifest.json").is_file() else None}
            self._loaded[name] = run
            return run


class ModelProbe:
    def __init__(self, config: AppConfig, enabled: bool):
        self.config = config
        self.enabled = enabled
        self._cache: dict[str, tuple[float, dict]] = {}

    def status(self, alias: str) -> dict:
        model = self.config.models[alias]
        item = {"alias": alias, "model": model.model, "base_url": model.base_url,
                "hosted": not model.base_url.startswith(("http://localhost", "http://127.0.0.1")),
                "credential": bool(os.getenv(model.api_key_env, "")) if model.api_key_env else None}
        if not self.enabled:
            item["reachable"] = None
            return item
        cached = self._cache.get(alias)
        if cached and time.monotonic() - cached[0] < 30:
            return {**item, **cached[1]}
        result = {"reachable": False}
        if item["credential"] is False:
            result["detail"] = f"{model.api_key_env} is not set"
        else:
            headers = {"Authorization": "Bearer " + os.environ[model.api_key_env]} if model.api_key_env else {}
            try:
                response = httpx.get(model.base_url + "/models", headers=headers, timeout=4, follow_redirects=False)
                result = {"reachable": response.status_code == 200, "detail": f"HTTP {response.status_code}"}
            except httpx.HTTPError:
                result["detail"] = "unreachable"
        self._cache[alias] = (time.monotonic(), result)
        return {**item, **result}


class AskBody(BaseModel):
    path: str | None = None
    upload: str | None = None
    question: str = Field(min_length=1, max_length=2000)
    model: str
    page: int | None = Field(default=None, ge=1)


class DetectBody(BaseModel):
    path: str | None = None
    upload: str | None = None
    model: str
    page: int | None = Field(default=None, ge=1)
    groups: list[str] | None = None


def create_app(config: AppConfig, data_root: Path, *, runs_dir: Path, output: Path,
               questions: Path | None = None, default_run: str | None = None,
               provider_factory=None, probe_models: bool = True) -> FastAPI:
    output = output.resolve()
    uploads = output / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    data_pipeline = Pipeline(config, data_root, output, provider_factory=provider_factory)
    upload_pipeline = Pipeline(config, uploads, output, provider_factory=provider_factory)
    index = data_pipeline.index
    question_index = QuestionIndex(questions, index)
    runs = RunStore(runs_dir.resolve(), index, exclude={output.name} if output.parent == runs_dir.resolve() else set())
    probe = ModelProbe(config, probe_models)
    folders: dict[str, list[str]] = defaultdict(list)
    for path in index.files:
        relative = path.relative_to(index.root)
        if len(relative.parts) > 1 and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            folders[relative.parts[0]].append(relative.as_posix())
    default_model = "nemotron_nvidia" if "nemotron_nvidia" in config.models else next(iter(config.models))

    app = FastAPI(title="PlanSight", docs_url="/api/docs", redoc_url=None)

    def target(path: str | None, upload: str | None) -> tuple[Pipeline, str]:
        if bool(path) == bool(upload):
            raise HTTPException(400, "Give exactly one of path or upload")
        if upload:
            if not UPLOAD_NAME.match(upload) or not (uploads / upload).is_file():
                raise HTTPException(404, "Unknown upload")
            return upload_pipeline, upload
        try:
            resolved = index.resolve(path)
        except (ValueError, OSError) as exc:
            raise HTTPException(404, str(exc)) from None
        return data_pipeline, resolved.relative_to(index.root).as_posix()

    def model_alias(alias: str) -> str:
        if alias not in config.models:
            raise HTTPException(400, f"Unknown model {alias!r}")
        return alias

    def live_url(pipeline: Pipeline, path: Path) -> str:
        return "/api/live/" + path.resolve().relative_to(pipeline.output).as_posix()

    @app.get("/api/models")
    def models():
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(config.models))) as pool:
            statuses = list(pool.map(probe.status, config.models))
        return {"default": default_model, "models": statuses}

    @app.get("/api/runs")
    def list_runs():
        names = runs.names()
        chosen = default_run if default_run in names else (names[0] if names else None)
        return {"default": chosen, "runs": names}

    @app.get("/api/runs/{name}")
    def run_detail(name: str):
        run = runs.load(name)
        return {"name": name, "summary": run["summary"], "routes": run["routes"],
                "zero_token_share": run["zero_token_share"], "models": run["manifest_models"]}

    @app.get("/api/runs/{name}/files/{relative:path}")
    def run_file(name: str, relative: str):
        return FileResponse(_inside(runs.load(name)["dir"], relative, {"assets"}))

    @app.get("/api/live/{relative:path}")
    def live_file(relative: str):
        return FileResponse(_inside(output, relative, {"assets", "views", "thumbs"}))

    @app.get("/api/datasets")
    def datasets(run: str | None = None):
        loaded = runs.load(run) if run else None
        by_source = (loaded["summary"].get("by_source") or {}) if loaded else {}
        result = []
        for name, files in sorted(folders.items()):
            asked = [f for f in files if f in question_index.by_path]
            result.append({"name": name, "files": len(files), "images_with_questions": len(asked),
                           "questions": sum(len(question_index.by_path[f]) for f in asked),
                           "run_metrics": by_source.get(name), "routes": loaded["routes"].get(name) if loaded else None})
        result.sort(key=lambda d: (d["run_metrics"] is None, -(d["questions"] or 0), d["name"]))
        return {"datasets": result}

    @app.get("/api/datasets/{name}/images")
    def dataset_images(name: str, run: str | None = None, offset: int = Query(0, ge=0),
                       limit: int = Query(48, ge=1, le=200)):
        if name not in folders:
            raise HTTPException(404, f"Unknown dataset {name!r}")
        saved = runs.load(run)["by_image"] if run else {}
        items = []
        for path in folders[name]:
            questions = question_index.by_path.get(path, [])
            predictions = saved.get(path, [])
            if not questions and not predictions:
                continue
            items.append({"path": path, "name": PurePosixPath(path).name, "questions": len(questions),
                          "saved": len(predictions), "correct": sum(p["correct"] is True for p in predictions),
                          "thumb": "/api/thumb?path=" + quote(path)})
        items.sort(key=lambda i: (-i["saved"], -i["correct"], i["path"]))
        return {"total": len(items), "images": items[offset:offset + limit]}

    @app.get("/api/samples")
    def samples(run: str | None = None):
        chosen: list[str] = [p for p in PREFERRED_SAMPLES if p in question_index.by_path or (index.root / p).is_file()]
        saved = runs.load(run)["by_image"] if run else {}
        best: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for path, predictions in saved.items():
            if PurePosixPath(path).suffix.lower() != ".ifc":
                best[path.split("/")[0]].append((-sum(p["correct"] is True for p in predictions), path))
        for name in sorted(best):
            chosen.extend(path for _, path in sorted(best[name])[:1])
        for name, files in sorted(folders.items()):
            if name not in best:
                drawable = [f for f in files if PurePosixPath(f).suffix.lower() in IMAGE_EXTENSIONS | {".pdf"}]
                chosen.extend(([f for f in drawable if f in question_index.by_path] or drawable)[:1])
        unique = list(dict.fromkeys(p for p in chosen if (index.root / p).is_file()))[:12]
        return {"samples": [{"path": p, "dataset": p.split("/")[0], "name": PurePosixPath(p).name,
                             "thumb": "/api/thumb?path=" + quote(p)} for p in unique],
                "questions": SUGGESTED_QUESTIONS}

    @app.get("/api/thumb")
    def thumb(path: str | None = None, upload: str | None = None):
        pipeline, reference = target(path, upload)
        source = pipeline.index.root / reference
        if source.suffix.lower() == ".ifc":
            raise HTTPException(404, "BIM models have no thumbnail")
        stat = source.stat()
        key = hashlib.sha1(f"{reference}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:24]
        destination = output / "thumbs" / f"{key}.jpg"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() == ".pdf":
                import fitz
                with fitz.open(source) as document:
                    page = document[0]
                    scale = min(1.0, 480 / max(page.rect.width, page.rect.height))
                    pixels = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
                    image = Image.frombytes("RGB", [pixels.width, pixels.height], pixels.samples)
            else:
                with Image.open(source) as opened:
                    opened.draft("RGB", (960, 960))
                    image = rgb(opened)
            image.thumbnail((480, 480), Image.Resampling.LANCZOS)
            descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".jpg")
            os.close(descriptor)
            image.save(temporary, "JPEG", quality=82)
            os.replace(temporary, destination)
        return FileResponse(destination, headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/render")
    def render(path: str | None = None, upload: str | None = None, page: int | None = Query(None, ge=1)):
        pipeline, reference = target(path, upload)
        source = pipeline.index.root / reference
        if source.suffix.lower() == ".ifc":
            return {"path": reference, "kind": "ifc", "pages": {}}
        try:
            pages = render_pages(source, pipeline.output / "assets", sha256_file(source), config.pipeline, page)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None
        return {"path": reference, "kind": "drawing",
                "pages": {str(p.number): live_url(pipeline, p.path) for p in pages}}

    @app.get("/api/image")
    def image_questions(path: str, run: str | None = None):
        _, reference = target(path, None)
        saved = {p["question"].strip(): p for p in (runs.load(run)["by_image"].get(reference, []) if run else [])}
        rows = []
        for q in question_index.by_path.get(reference, []):
            rows.append({**q, "saved": saved.pop(q["question"].strip(), None)})
        rows.extend({"question": p["question"], "answer": p.get("expected_answer"), "unit": p.get("unit"),
                     "type": None, "page": p.get("page"), "saved": p} for p in saved.values())
        return {"path": reference, "dataset": reference.split("/")[0],
                "kind": "ifc" if reference.lower().endswith(".ifc") else "drawing", "questions": rows}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        suffix = PurePosixPath(file.filename or "").suffix.lower()
        if suffix not in UPLOAD_EXTENSIONS:
            raise HTTPException(415, "Upload a PNG, JPG, WEBP or PDF drawing")
        digest, size = hashlib.sha256(), 0
        descriptor, temporary = tempfile.mkstemp(dir=uploads, suffix=".part")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Drawings up to 50 MB are supported")
                    digest.update(chunk)
                    stream.write(chunk)
            name = digest.hexdigest() + (".jpg" if suffix == ".jpeg" else suffix)
            try:
                if suffix == ".pdf":
                    import fitz
                    with fitz.open(temporary) as document:
                        if document.page_count < 1:
                            raise ValueError("empty PDF")
                else:
                    with Image.open(temporary) as image:
                        image.verify()
            except Exception:
                raise HTTPException(422, "The file could not be read as a drawing") from None
            os.replace(temporary, uploads / name)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {"upload": name, "name": file.filename}

    @app.post("/api/ask")
    def ask(body: AskBody):
        pipeline, reference = target(body.path, body.upload)
        alias = model_alias(body.model)
        known = question_index.find(reference, body.question) if body.path else None
        question = Question(path=reference, question=body.question, page=body.page,
                            unit=known["unit"] if known else None, type=known["type"] if known else None)
        pipeline.forget_failures()
        started = time.monotonic()
        result = pipeline.predict(question, alias)
        elapsed = time.monotonic() - started
        if known and known["answer"] is not None:
            result.expected_answer = known["answer"]
            result.has_ground_truth = True
        payload = result.model_dump(mode="json")
        payload["page_urls"] = {page: "/api/live/" + image for page, image in result.page_images.items()}
        payload["route_label"] = route_of(payload)
        payload["correct"] = is_correct(payload)
        payload["live"] = result.usage.requests > 0
        payload["elapsed_seconds"] = elapsed
        return payload

    @app.post("/api/detect")
    def detect(body: DetectBody):
        pipeline, reference = target(body.path, body.upload)
        alias = model_alias(body.model)
        groups = body.groups or list(GROUPS)
        if any(g not in GROUPS for g in groups):
            raise HTTPException(400, f"Groups must be among {', '.join(GROUPS)}")
        pipeline.forget_failures()
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = {g: pool.submit(pipeline.inventory_page, reference, g, alias, body.page) for g in groups}
            outcomes = {}
            for group, future in futures.items():
                try:
                    outcomes[group] = future.result()
                except (ValueError, OSError) as exc:
                    raise HTTPException(422, str(exc)) from None
        usage = Usage(estimated_cost_usd=0)
        items, summary, page_urls = [], {}, {}
        for group, outcome in outcomes.items():
            add_usage(usage, outcome["usage"])
            page_urls = page_urls or {str(p.number): live_url(pipeline, p.path) for p in outcome["pages"]}
            for page, _, item in outcome["entries"]:
                items.append({**item.model_dump(mode="json"), "group": group, "page": page})
            summary[group] = {"failed": outcome["failed"], "complete": outcome["complete"],
                              "items": len(outcome["entries"]), "warnings": outcome["warnings"],
                              "live": outcome["usage"].requests > 0}
        counts = Counter(i["kind"] for i in items)
        uncertain = Counter(i["kind"] for i in items if i["uncertain"])
        room_types = Counter(i["room_type"] for i in items if i["kind"] == "room" and i.get("room_type"))
        return {"path": reference, "model": alias, "page_urls": page_urls, "items": items,
                "counts": {k: {"total": n, "uncertain": uncertain.get(k, 0)} for k, n in counts.most_common()},
                "room_types": dict(room_types.most_common()), "groups": summary,
                "usage": usage.model_dump(mode="json"), "live": usage.requests > 0,
                "elapsed_seconds": time.monotonic() - started}

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Question(BaseModel):
    """Ground truth stays here and must never be passed to a model."""
    model_config = ConfigDict(extra="allow")
    path: str
    question: str
    answer: Any = None
    unit: str | None = None
    type: str | None = None
    id: str | None = None
    page: int | None = Field(default=None, ge=1)

    @field_validator("path", "question")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Intent(StrictModel):
    operation: Literal["count", "general"]
    category: str
    canonical_question: str
    rule: str


def _valid_box(box):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
        raise ValueError("bbox must use finite normalized coordinates in [0,1]")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")
    return box


class Candidate(StrictModel):
    """One visual entity. Box coordinates are normalized to its input view."""
    label: str = Field(min_length=1, max_length=200)
    bbox: tuple[float, float, float, float]
    counted: bool = True
    uncertain: bool = False
    note: str = Field(default="", max_length=1200)

    @field_validator("bbox")
    @classmethod
    def valid_box(cls, box):
        return _valid_box(box)


InventoryKind = Literal["door", "window", "toilet", "washbasin", "kitchen_sink", "bathtub", "shower",
                        "stair", "column", "room", "apartment", "floor"]
RoomType = Literal["bedroom", "kitchen", "bathroom", "wc", "sauna", "balcony", "terrace", "living",
                   "hallway", "storage", "other"]


class InventoryItem(StrictModel):
    """One entity of a whole-drawing inventory; text fields hold printed values only."""
    kind: InventoryKind
    bbox: tuple[float, float, float, float]
    text: str = Field(default="", max_length=200)
    number: str = Field(default="", max_length=40)
    area: float | None = Field(default=None, ge=0)
    room_type: RoomType | None = None
    apartment: str = Field(default="", max_length=40)
    uncertain: bool = False

    @field_validator("bbox")
    @classmethod
    def valid_box(cls, box):
        return _valid_box(box)

    @field_validator("area")
    @classmethod
    def finite_area(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("area must be finite")
        return value


class DrawingInventory(StrictModel):
    """Everything countable on one view, extracted once and shared by all questions about it."""
    items: list[InventoryItem] = Field(default_factory=list, max_length=2000)
    complete: bool = False
    warnings: list[str] = Field(default_factory=list, max_length=50)


class Extraction(StrictModel):
    """Visible evidence and a short justification; never hidden chain-of-thought."""
    answer: str | int | float | bool | None = None
    candidates: list[Candidate] = Field(default_factory=list, max_length=2000)
    complete: bool = False
    rationale: str = Field(default="", max_length=4000)
    warnings: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("answer")
    @classmethod
    def finite_answer(cls, value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("answer must be finite")
        return value


class Evidence(Candidate):
    bbox: tuple[float, float, float, float] | None = None
    id: str
    page: int = Field(ge=1)
    category: str
    view_ids: list[str] = Field(default_factory=list)
    provenance: Literal["model_prediction", "ifc_entity", "dxf_entity"] = "model_prediction"
    verified: bool = False
    entity_guid: str | None = None


class Usage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    cached_requests: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0, ge=0)
    estimated_cost_usd: float | None = None
    usage_reported: bool = True


class ModelResponse(StrictModel):
    content: str
    usage: Usage
    finish_reason: str | None = None


class Prediction(StrictModel):
    schema_version: str = "1.0"
    question_id: str
    path: str
    question: str
    page: int | None = None
    model: str
    model_id: str
    status: Literal["answered", "needs_review", "error", "unsupported"]
    answer: str | int | float | bool | None = None
    proposed_answer: str | int | float | bool | None = None
    unit: str | None = None
    category: str
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: str = ""
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    source_sha256: str = ""
    config_sha256: str = ""
    page_images: dict[str, str] = Field(default_factory=dict)
    expected_answer: Any = None
    has_ground_truth: bool = False
    error: str | None = None
    route: str | None = None


class ModelConfig(StrictModel):
    model: str
    base_url: str = "http://localhost:8001/v1"
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=180, gt=0)
    max_retries: int = Field(default=2, ge=0, le=8)
    max_tokens: int = Field(default=4096, ge=64)
    temperature: float = Field(default=0, ge=0, le=2)
    json_mode: bool = False
    # Native box scale of the model: 1 for [0,1] fractions, 1000 for the 0..1000 grounding convention.
    bbox_scale: Literal[1, 1000] = 1
    extra_body: dict[str, Any] = Field(default_factory=dict)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)
    # Self-hosted servers bill by the hour, not per token; runs report wall time × this rate.
    gpu_cost_per_hour: float | None = Field(default=None, ge=0)

    @field_validator("base_url")
    @classmethod
    def endpoint(cls, value):
        from urllib.parse import urlparse
        p = urlparse(value)
        if p.scheme not in {"http", "https"} or not p.netloc or p.username or p.password or p.query or p.fragment:
            raise ValueError("base_url must be an http(s) endpoint without credentials/query/fragment")
        return value.rstrip("/")

    @field_validator("extra_body")
    @classmethod
    def no_core_overrides(cls, value):
        forbidden = {"model", "messages", "stream", "max_tokens", "temperature", "response_format"}
        if forbidden.intersection(value):
            raise ValueError("extra_body cannot override core request fields")
        return value


class PipelineConfig(StrictModel):
    mode: Literal["inventory", "direct"] = "inventory"
    views: Literal["overview", "tiles"] = "overview"
    max_image_side: int = Field(default=2048, ge=256, le=8192)
    tile_size: int = Field(default=1024, ge=256, le=4096)
    tile_overlap: float = Field(default=0.15, ge=0, lt=0.5)
    max_views: int = Field(default=24, ge=1, le=256)
    max_pages: int = Field(default=12, ge=1, le=1000)
    pdf_dpi: int = Field(default=150, ge=72, le=400)
    max_source_pixels: int = Field(default=80000000, ge=1000000)
    dedup_iou: float = Field(default=0.55, gt=0, le=1)
    max_concurrency: int = Field(default=2, ge=1, le=256)
    # Answer supported PDF questions from a same-named semantic DXF instead of a VLM.
    use_dxf_sibling: bool = True
    # Answer quoted storey/room/model questions about `<model>_plan-*.pdf` from `<model>.ifc`.
    use_ifc_sibling: bool = True
    # per_question: one vision request per question (canonical counts share one per category).
    # per_drawing: one shared inventory request per page answers every recognized question.
    strategy: Literal["per_question", "per_drawing"] = "per_question"
    # Inventory requests only. Greedy decoding loops on long lists (invented grids of rooms or
    # doors); a repetition penalty prevents most loops and the token cap makes the rest fail fast.
    inventory_max_tokens: int = Field(default=4096, ge=256)
    inventory_extra_body: dict[str, Any] = Field(default_factory=lambda: {"repetition_penalty": 1.1})
    cache_dir: str = ".drawingqa/cache"


class Profile(StrictModel):
    """A named way of running: pipeline overrides plus the default models to evaluate."""
    description: str = ""
    models: list[str] | None = None
    pipeline: dict[str, Any] = Field(default_factory=dict)


class AppConfig(StrictModel):
    models: dict[str, ModelConfig]
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    profiles: dict[str, Profile] = Field(default_factory=dict)

    def apply_profile(self, name: str) -> list[str] | None:
        if name not in self.profiles:
            raise ValueError(f"Unknown profile {name!r}; configured: {', '.join(sorted(self.profiles)) or 'none'}")
        profile = self.profiles[name]
        self.pipeline = PipelineConfig.model_validate({**self.pipeline.model_dump(), **profile.pipeline})
        unknown = [m for m in profile.models or [] if m not in self.models]
        if unknown:
            raise ValueError(f"Profile {name!r} names unknown models: {', '.join(unknown)}")
        return profile.models

    @model_validator(mode="after")
    def nonempty_models(self):
        if not self.models:
            raise ValueError("at least one model is required")
        return self

"""Configuration loading and reproducible hashes without credential values."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from .schemas import AppConfig


def load_config(path: Path | str) -> AppConfig:
    path = Path(path)
    # Configuration does not interpolate environment variables into saved files.
    # Explicitly set environment variables take precedence over .env.
    load_dotenv(Path.cwd() / ".env", override=False)
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return AppConfig.model_validate(payload)


def config_fingerprint(config: BaseModel | dict) -> str:
    payload = config.model_dump(mode="json") if isinstance(config, BaseModel) else config
    # ModelConfig stores only the NAME of the credential variable. Values are
    # read solely at request time and are never present in this serialization.
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

"""Seed a classification cache from eval/fixtures/classification_cache.json.

The fixture holds the classifier's recorded predictions (exported with
scripts/export_eval_fixture.py). Seeding a fresh SQLite store with it and pointing
LLM_PROVIDER / *_MODEL at the same provider+model makes every classify_cached()
call a cache hit, so evals and tests run with no API key and no LLM calls.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from src.core import store

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "classification_cache.json"

_MODEL_ENV = {"cerebras": "CEREBRAS_MODEL", "gemini": "GEMINI_MODEL"}


def seed_cache(conn: sqlite3.Connection, path: Path = FIXTURE) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    provider, model = data["provider"], data["model"]
    os.environ["LLM_PROVIDER"] = provider
    os.environ[_MODEL_ENV[provider]] = model
    for p in data["predictions"]:
        store.set_cached(
            conn, provider, model, p["normalized_description"], p["subcategory"],
            p["category"], p["confidence"], p["justification"],
        )
    return provider, model

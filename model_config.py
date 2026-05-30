from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "planner_model": "gpt-5-mini",
    "rerank_model": "gpt-5-mini",
    "answer_model": "gpt-5-mini",
    "embedding_model": "text-embedding-3-large",
    # Keep Qdrant compatible with the existing 1536-dimension collection while
    # using the stronger embedding model.
    "embedding_dimensions": 1536,
}

MODEL_STEPS = {
    "planner_model": {
        "label": "Planning",
        "description": "Turns the user question and recent history into focused retrieval queries, collection choices, and metadata hints.",
    },
    "rerank_model": {
        "label": "Reranking",
        "description": "Reviews candidate chunks and keeps only evidence that directly helps answer the question.",
    },
    "answer_model": {
        "label": "Answer synthesis",
        "description": "Writes the final answer using only retrieved context and inline citations.",
    },
    "embedding_model": {
        "label": "Embeddings",
        "description": "Converts PDF chunks and search queries into vectors for Qdrant similarity search.",
    },
}

MODEL_OPTIONS = {
    "chat": ["gpt-5-mini", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"],
    "embedding": ["text-embedding-3-large", "text-embedding-3-small"],
}

CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", Path(__file__).resolve().parent / "data" / "model_config.json"))


def _env_default(key: str, env_name: str) -> str:
    return os.getenv(env_name, str(DEFAULT_MODEL_CONFIG[key])).strip() or str(DEFAULT_MODEL_CONFIG[key])


def default_model_config() -> dict[str, Any]:
    config = dict(DEFAULT_MODEL_CONFIG)
    config["planner_model"] = _env_default("planner_model", "PLANNER_MODEL")
    config["rerank_model"] = _env_default("rerank_model", "RERANK_MODEL")
    config["answer_model"] = _env_default("answer_model", "ANSWER_MODEL")
    config["embedding_model"] = _env_default("embedding_model", "EMBEDDING_MODEL")
    config["embedding_dimensions"] = int(os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_MODEL_CONFIG["embedding_dimensions"])))
    return config


def normalize_model_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    config = default_model_config()
    raw = raw or {}
    for key in ("planner_model", "rerank_model", "answer_model", "embedding_model"):
        value = str(raw.get(key) or config[key]).strip()
        if value:
            config[key] = value
    try:
        dimensions = int(raw.get("embedding_dimensions") or config["embedding_dimensions"])
    except (TypeError, ValueError):
        dimensions = int(DEFAULT_MODEL_CONFIG["embedding_dimensions"])
    config["embedding_dimensions"] = dimensions
    return config


def get_model_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return normalize_model_config()
    try:
        return normalize_model_config(json.loads(CONFIG_PATH.read_text()))
    except (OSError, json.JSONDecodeError):
        return normalize_model_config()


def save_model_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_model_config(config)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(normalized, indent=2) + "\n")
    return normalized


def model_for_step(step: str) -> str:
    return str(get_model_config()[step])


def embedding_dimensions() -> int:
    return int(get_model_config()["embedding_dimensions"])


def public_model_config() -> dict[str, Any]:
    config = get_model_config()
    return {
        "config": config,
        "defaults": default_model_config(),
        "steps": MODEL_STEPS,
        "options": MODEL_OPTIONS,
        "storage_path": str(CONFIG_PATH),
    }

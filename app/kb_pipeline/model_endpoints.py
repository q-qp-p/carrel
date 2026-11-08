"""Shared endpoint overrides, stored separately from the chat-model registry.

Environment settings remain the baseline. Console values override only the four
configured roles; parsing, graph jobs and search read the same application record.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

CONFIG_KEY = "model_endpoints"
ROLES = ("embedding", "visual_embedding", "reranker", "visual_reranker")


def read_overrides(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='app_config'").fetchone():
            return {}
        row = con.execute("SELECT value FROM app_config WHERE key=?", (CONFIG_KEY,)).fetchone()
        if row is None:
            return {}
        value = json.loads(row[0])
        if not isinstance(value, dict) or any(k not in ROLES or not isinstance(v, dict) for k, v in value.items()):
            raise ValueError("Invalid model endpoint configuration")
        return value
    finally:
        con.close()


def apply_overrides(settings):
    values = {}
    for role, row in read_overrides(settings.state_db).items():
        for field in ("base_url", "model_id", "api_key"):
            if field in row:
                values[f"{role}_{field}"] = row[field]
        if "embedding" in role and "dim" in row:
            values[f"{role}_dim"] = row["dim"]
        if "embedding" in role:
            for field, suffix in (("batch_size", "batch"), ("concurrency", "concurrency")):
                if field in row:
                    values[f"{role}_{suffix}"] = row[field]
        if role == "visual_embedding" and "enabled" in row:
            values["visual_embedding_enabled"] = row["enabled"]
        if "reranker" in role and row.get("enabled") is False:
            values[f"{role}_base_url"] = ""
    return replace(settings, **values) if values else settings


def endpoint_values(settings, overrides: dict | None = None) -> dict:
    """Internal values including secrets; API responses must remove api_key."""
    saved = read_overrides(settings.state_db) if overrides is None else overrides
    out = {}
    for role in ROLES:
        value = {field: str(getattr(settings, f"{role}_{field}", "") or "")
                 for field in ("base_url", "model_id", "api_key")}
        if "embedding" in role:
            value["dim"] = getattr(settings, f"{role}_dim")
            value["batch_size"] = getattr(settings, f"{role}_batch")
            value["concurrency"] = getattr(settings, f"{role}_concurrency")
        value["enabled"] = (True if role == "embedding" else
                            bool(settings.visual_embedding_enabled) if role == "visual_embedding" else
                            bool(value["base_url"]))
        value.update(saved.get(role, {}))
        out[role] = value
    return out

#!/usr/bin/env python3
"""Shared paths, serialization, and value utilities for the benchmark."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def root_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - user environment guard.
        raise SystemExit(
            "PyYAML is required for benchmark configs. Install with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    with root_path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return loaded


def read_json(path: str | Path) -> Any:
    with root_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = root_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = root_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
            count += 1
    return count


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with root_path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield record


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "unknown"


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert value to float: {value!r}") from exc
    if number != number:
        raise ValueError("NaN is not a valid ground-truth value")
    return number


def selected_image_ids(config: dict[str, Any], image_limit: int | None = None) -> list[int]:
    policy = config.get("image_policy", {})
    ids = [int(item) for item in policy.get("selected_image_ids", [])]
    limit = image_limit if image_limit is not None else int(policy.get("default_limit", 15))
    return ids[:limit]

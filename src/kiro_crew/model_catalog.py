"""模型目录的基础解析与可用性判断."""

from __future__ import annotations

from collections.abc import Sequence


def advertised_model_ids(entries: object) -> list[str]:
    if not isinstance(entries, (list, tuple)):
        return []
    ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("modelId") or entry.get("value") or ""
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id)
    return ids


def model_is_unusable(model_id: str, advertised: Sequence[str] | None) -> bool:
    if not advertised:
        return False
    wanted = model_id.strip().lower()
    return wanted not in {model.strip().lower() for model in advertised if model and model.strip()}
"""共享事件的容错 JSON 渲染."""

from __future__ import annotations

import json
from typing import Any

UNSERIALISABLE_SIBLING_VALUE = "[[value omitted: nested too deeply to serialise]]"


def _dumps_degraded(payload: Any, **kwargs: Any) -> str:
    try:
        return json.dumps(payload, **kwargs)
    except (TypeError, ValueError):
        try:
            return str(payload)
        except RecursionError:
            return UNSERIALISABLE_SIBLING_VALUE
    except RecursionError:
        return UNSERIALISABLE_SIBLING_VALUE
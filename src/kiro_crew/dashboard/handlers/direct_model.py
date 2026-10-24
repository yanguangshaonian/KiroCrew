"""内置 Agent 的模型连接设置."""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from aiohttp import web

from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.config.loader_document import update_config_locked


async def api_direct_model(request: web.Request) -> web.Response:
    if request.method == "GET":
        config = await asyncio.to_thread(KiroCrewConfig.load)
        settings = dict(config.agent.direct_model)
        settings["has_api_key"] = bool(settings.pop("api_key", ""))
        settings["model"] = (
            config.agent.model if config.agent.model != "auto" else settings.get("model", "")
        )
        return web.json_response(settings)
    try:
        body = await request.json()
        endpoint = str(body.get("url", "")).strip()
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("模型地址必须是完整的 HTTP 或 HTTPS URL.")
        model = str(body.get("model", "")).strip()
        if not model:
            raise ValueError("模型 id 不能为空.")
        models = body.get("models", [])
        if not isinstance(models, list) or not models:
            raise ValueError("模型列表不能为空.")
        ids = set()
        for row in models:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not row["id"].strip()
            ):
                raise ValueError("每个模型必须有非空 id.")
            if row["id"] in ids:
                raise ValueError("模型 id 不能重复.")
            ids.add(row["id"])
            for key in ("context_window", "max_input_tokens", "max_output_tokens"):
                if key in row and (type(row[key]) is not int or row[key] <= 0):
                    raise ValueError(f"{key} 必须是正整数.")
        if model not in ids:
            raise ValueError("默认模型必须包含在模型列表中.")

        def mutate(data):
            agent = data.setdefault("agent", {})
            previous = agent.get("direct_model", {})
            agent["provider"] = "direct"
            agent["model"] = model
            agent["direct_model"] = {
                "url": endpoint,
                "model": model,
                "models": models,
                "api_key": body.get("api_key") or previous.get("api_key", ""),
            }
            return data

        await asyncio.to_thread(update_config_locked, mutate=mutate, wait_for_lock=False)
        state = request.app.get("state")
        sessions = getattr(state, "sessions", None)
        if sessions is not None:
            config = await asyncio.to_thread(KiroCrewConfig.load)
            sessions._cfg = config
            sessions._provider_factory = config.create_provider_factory()
        return web.json_response({"ok": True})
    except BlockingIOError:
        return web.json_response({"error": "配置正在被其他操作修改, 请稍后重试."}, status=409)
    except (ValueError, TypeError, AttributeError) as error:
        return web.json_response({"error": str(error)}, status=400)

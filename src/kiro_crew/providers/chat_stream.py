"""Chat Completions 流响应与工具调用聚合."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ChatStream:
    content: str = ""
    reasoning: str = ""
    calls: dict[int, dict] = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""
    done: bool = False

    def feed(self, data: str) -> list[tuple[str, str]]:
        if self.done:
            raise ValueError("模型流结束后仍收到数据.")
        if data.strip() == "[DONE]":
            if not self.finish_reason:
                raise ValueError("模型流缺少 finish_reason.")
            self.done = True
            return []
        payload = json.loads(data)
        if payload.get("error"):
            error = payload["error"]
            message = (
                error.get("message", "模型返回错误.") if isinstance(error, dict) else str(error)
            )
            raise ValueError(f"模型请求失败: {message}")
        if payload.get("usage"):
            self.usage = payload["usage"]
        choices = payload.get("choices") or []
        if not choices:
            return []
        choice = choices[0]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        events = []
        text = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if text:
            self.content += text
            events.append(("text_chunk", text))
        if reasoning:
            self.reasoning += reasoning
            events.append(("thinking_chunk", reasoning))
        for fragment in delta.get("tool_calls") or []:
            index = fragment.get("index")
            if index is None:
                if len(self.calls) != 1:
                    raise ValueError("工具调用分片缺少明确的 index.")
                index = next(iter(self.calls))
            call = self.calls.setdefault(
                index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if fragment.get("id"):
                call["id"] += fragment["id"]
            function = fragment.get("function") or {}
            for name in ("name", "arguments"):
                call["function"][name] += function.get(name) or ""
        return events

    def message(self) -> dict:
        if not self.done:
            raise ValueError("模型流意外中断, 缺少 [DONE].")
        if self.finish_reason not in {"stop", "tool_calls"}:
            raise ValueError(f"模型未正常完成: {self.finish_reason}.")
        message = {"role": "assistant", "content": self.content or None}
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        if self.calls:
            if self.finish_reason != "tool_calls":
                raise ValueError("模型工具调用与结束原因不一致.")
            calls = [self.calls[index] for index in sorted(self.calls)]
            for call in calls:
                if not call["id"] or not call["function"]["name"]:
                    raise ValueError("模型工具调用缺少 id 或 name.")
                arguments = json.loads(call["function"]["arguments"])
                if not isinstance(arguments, dict):
                    raise TypeError("模型工具参数必须是 JSON 对象.")
            message["tool_calls"] = calls
        elif self.finish_reason == "tool_calls":
            raise ValueError("模型声明工具调用, 但未返回工具.")
        return message

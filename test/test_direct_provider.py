import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.dashboard.handlers.direct_model import api_direct_model
from kiro_crew.dashboard.token_auth import token_auth_middleware
from kiro_crew.providers.chat_stream import ChatStream
from kiro_crew.providers.direct import DirectProvider
from kiro_crew.providers.direct_mcp import McpConnection


class ChatStreamTests(unittest.TestCase):
    def test_session_manager_identifies_direct_provider(self):
        from kiro_crew.session import (
            _is_acp_provider,
            _is_claude_backend,
            _provider_label,
        )

        provider = DirectProvider(settings={"model": "test"})
        self.assertEqual(_provider_label(provider), "direct")
        self.assertFalse(_is_acp_provider(provider))
        self.assertFalse(_is_claude_backend(provider))

    def test_direct_import_does_not_load_legacy_runtime(self):
        code = (
            "import sys; from kiro_crew.providers.direct import DirectProvider; "
            "assert 'kiro_crew.acp' not in sys.modules; "
            "assert 'kiro_crew.acp.client' not in sys.modules; "
            "assert 'kiro_crew.acp.runtime' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30, capture_output=True)

    def test_usage_type_is_shared_without_legacy_ownership(self):
        from kiro_crew.acp.types import TurnUsage as LegacyUsage
        from kiro_crew.turn_usage import TurnUsage

        self.assertIs(LegacyUsage, TurnUsage)
        self.assertEqual(TurnUsage.__module__, "kiro_crew.turn_usage")

    def test_legacy_package_does_not_load_runtime_on_import(self):
        code = (
            "import sys; import kiro_crew.acp; "
            "assert 'kiro_crew.acp.client' not in sys.modules; "
            "assert 'kiro_crew.acp.runtime' not in sys.modules; "
            "from kiro_crew.acp import AcpEvent; "
            "from kiro_crew.acp.types import AcpEvent as Event; "
            "assert AcpEvent is Event"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30, capture_output=True)

    def test_truncated_stream_is_rejected(self):
        stream = ChatStream()
        stream.feed(json.dumps({"choices": [{"delta": {"content": "partial"}}]}))
        with self.assertRaisesRegex(ValueError, "DONE"):
            stream.message()

    def test_ambiguous_tool_index_is_rejected(self):
        stream = ChatStream()
        with self.assertRaisesRegex(ValueError, "index"):
            stream.feed(
                json.dumps(
                    {"choices": [{"delta": {"tool_calls": [{"function": {"name": "read_file"}}]}}]}
                )
            )

    def test_in_band_error_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "模型请求失败"):
            ChatStream().feed(json.dumps({"error": {"message": "failed"}}))


class DirectProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.env = patch.dict(os.environ, {"KIROCREW_HOME": self.root.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.requests = []

        async def completion(request):
            payload = await request.json()
            self.requests.append(payload)
            if payload.get("stream") is False:
                return web.json_response(
                    {
                        "choices": [{"message": {"content": "保留测试标记 SUMMARY_TOKEN_42."}}],
                        "usage": {"completion_tokens": 8},
                    }
                )
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            if len(self.requests) == 1:
                chunks = [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "write_file",
                                                "arguments": '{"path":"result.txt",',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '"content":"hello"}'},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                ]
            else:
                chunks = [
                    {"choices": [{"delta": {"content": "完成"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
                ]
            for chunk in chunks:
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", completion)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.addAsyncCleanup(self.runner.cleanup)
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.settings = {"url": f"http://127.0.0.1:{port}/v1/chat/completions", "model": "test"}

    async def test_factory_preserves_direct_model_ids_and_agent_priority(self):
        from kiro_crew.config import KiroCrewConfig
        from kiro_crew.config.loader import resolve_effective_model

        config = KiroCrewConfig()
        config.agent.direct_model = {**self.settings, "model": "custom/default-v2"}
        config.agent.model = "auto"
        with patch.object(KiroCrewConfig, "_resolve_named_agent_model", return_value="custom/agent-v3"):
            factory = config.create_provider_factory()
            provider = factory(agent="worker", cwd=self.root.name)
            self.assertEqual(provider.model, "custom/agent-v3")
            self.assertEqual(
                factory(agent="worker", model_override="custom/override", cwd=self.root.name).model,
                "custom/override",
            )
            self.assertEqual(factory(agent="kirocrew", cwd=self.root.name).model, "custom/default-v2")
            self.assertEqual(resolve_effective_model(config), "custom/default-v2")
            self.assertEqual(
                resolve_effective_model(config, template="worker"), "custom/agent-v3"
            )

    async def test_spawn_effort_receipt_uses_configured_levels(self):
        from kiro_crew.config import KiroCrewConfig
        from kiro_crew.subagent import (
            effort_applied_note,
            effort_drop_reason,
            effort_pending_note,
        )

        config = KiroCrewConfig()
        config.agent.direct_model = {
            "model": "custom/worker",
            "models": [{"id": "custom/worker", "reasoning_effort": ["high"]}],
        }
        with (
            patch.object(KiroCrewConfig, "load", return_value=config),
            patch("kiro_crew.subagent._spawn_effective_model", return_value="custom/worker"),
        ):
            self.assertIn("reasoning_effort", effort_applied_note("", "high"))
            self.assertEqual(effort_drop_reason("", "high"), "")
            self.assertEqual(effort_applied_note("", "max"), "")
            self.assertTrue(effort_drop_reason("", "max"))
            self.assertEqual(effort_pending_note("", "high"), "")

    async def test_tool_loop_and_persistent_resume(self):
        provider = DirectProvider(
            settings=self.settings, work_dir=self.root.name, session_key="test"
        )
        self.addAsyncCleanup(provider.shutdown)
        events = [event async for event in provider.stream("写入文件")]
        self.assertEqual(Path(self.root.name, "result.txt").read_text(), "hello")
        self.assertEqual(
            [event.kind for event in events], ["tool_call", "tool_result", "text_chunk", "complete"]
        )
        self.assertEqual(self.requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(provider.context_used_tokens(), 12)
        self.assertEqual(events[0].tool_name, "write_file")
        self.assertTrue(events[0].tool_identity_trusted)
        self.assertFalse(events[0].mcp_identity_trusted)
        self.assertEqual(events[0].raw_tool_params["path"], "result.txt")
        self.assertEqual(events[1].tool_name, "write_file")
        self.assertEqual(events[-1].usage.input_tokens, 10)
        self.assertEqual(events[-1].usage.output_tokens, 2)
        self.assertEqual(events[-1].usage.num_turns, 2)
        resumed = DirectProvider(
            settings=self.settings, work_dir=self.root.name, session_key="test"
        )
        self.addAsyncCleanup(resumed.shutdown)
        resumed.set_resume_session_id(provider.session_id)
        await resumed.start()
        self.assertEqual(resumed._messages, provider._messages)
        self.assertTrue(resumed.resumed)

    async def test_tool_waits_for_approval_before_writing(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name, agent="test")
        self.addAsyncCleanup(provider.shutdown)
        async for event in provider.stream("写入文件"):
            if event.kind == "permission_request":
                self.assertFalse(Path(self.root.name, "result.txt").exists())
                self.assertTrue(await provider.approve_tool(event.request_id))
        self.assertEqual(Path(self.root.name, "result.txt").read_text(), "hello")
        self.assertFalse(provider._permissions)

    async def test_rejected_tool_does_not_write(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name, agent="test")
        self.addAsyncCleanup(provider.shutdown)
        async for event in provider.stream("写入文件"):
            if event.kind == "permission_request":
                await provider.reject_tool(event.request_id)
            if event.kind == "tool_result":
                self.assertEqual(event.tool_status, "failed")
        self.assertFalse(Path(self.root.name, "result.txt").exists())
        self.assertFalse(provider._permissions)

    async def test_cancel_pending_permission_cleans_waiter(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name, agent="test")
        self.addAsyncCleanup(provider.shutdown)
        waiting = asyncio.Event()

        async def consume():
            async for event in provider.stream("写入文件"):
                if event.kind == "permission_request":
                    waiting.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(waiting.wait(), timeout=5)
        self.assertEqual(await provider.cancel(wait_ack_timeout=1), "acked")
        await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(provider._permissions)
        self.assertFalse(Path(self.root.name, "result.txt").exists())

    async def test_permission_event_reaches_existing_file_protection(self):
        from kiro_crew.config.paths import config_dir
        from kiro_crew.hook_runtime.tool_identity import hook_gate_kwargs
        from kiro_crew.hooks import HookManager

        provider = DirectProvider(settings=self.settings, work_dir=self.root.name, agent="test")
        self.addAsyncCleanup(provider.shutdown)
        protected = config_dir() / "security_policy.json"
        async for event in provider.stream("写入文件"):
            if event.kind == "permission_request":
                self.assertEqual(event.tool_kind, "edit")
                event.raw_tool_params["path"] = str(protected)
                event.diff_path = str(protected)
                result = await asyncio.to_thread(
                    HookManager().on_tool_call, event.title, **hook_gate_kwargs(event)
                )
                self.assertEqual(result.action, "deny")
                await provider.reject_tool(event.request_id)
        self.assertFalse(protected.exists())
        self.assertFalse(Path(self.root.name, "result.txt").exists())

    async def test_expired_grant_keeps_permission_pending(self):
        from kiro_crew.trust_patterns import GrantDeadline, GrantEnded

        provider = DirectProvider(settings=self.settings, work_dir=self.root.name, agent="test")
        self.addAsyncCleanup(provider.shutdown)
        async for event in provider.stream("写入文件"):
            if event.kind == "permission_request":
                with self.assertRaises(GrantEnded):
                    await provider.approve_tool(
                        event.request_id, deadline=GrantDeadline(wall=0, mono=0)
                    )
                self.assertFalse(provider._permissions[event.request_id].done())
                await provider.reject_tool(event.request_id)
        self.assertFalse(Path(self.root.name, "result.txt").exists())

    async def test_mcp_event_uses_original_tool_identity(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name)
        self.addAsyncCleanup(provider.shutdown)
        provider._mcp_tools["write_file"] = (SimpleNamespace(name="test-server"), "save_document")
        with patch.object(provider, "_execute", return_value="完成"):
            events = [event async for event in provider.stream("测试工具事件")]
        self.assertEqual(events[0].title, "@test-server/save_document")
        self.assertEqual(events[0].tool_name, "save_document")
        self.assertEqual(events[0].mcp_server_name, "test-server")
        self.assertTrue(events[0].mcp_identity_trusted)
        self.assertEqual(events[1].wire_title, "@test-server/save_document")

    async def test_fresh_session_does_not_restore_previous_key(self):
        provider = DirectProvider(
            settings=self.settings, work_dir=self.root.name, session_key="same-key"
        )
        self.addAsyncCleanup(provider.shutdown)
        await provider.start()
        provider._messages = [{"role": "user", "content": "old history"}]
        await provider._save()
        replacement = DirectProvider(
            settings=self.settings, work_dir=self.root.name, session_key="same-key"
        )
        self.addAsyncCleanup(replacement.shutdown)
        await replacement.start()
        self.assertNotEqual(provider.session_id, replacement.session_id)
        self.assertEqual(replacement._messages, [])
        self.assertFalse(replacement.resumed)

    async def test_failed_start_closes_client_and_can_retry(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name)
        self.addAsyncCleanup(provider.shutdown)
        with (
            patch.object(provider, "_load", side_effect=ValueError("历史无效.")),
            self.assertRaisesRegex(ValueError, "历史无效"),
        ):
            await provider.start()
        self.assertFalse(provider.is_alive())
        self.assertIsNone(provider._client)
        await provider.start()
        self.assertTrue(provider.is_alive())

    async def test_restart_does_not_duplicate_mcp_tools(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name, agent="test")
        self.addAsyncCleanup(provider.shutdown)
        schema = {"type": "function", "function": {"name": "mcp_test"}}

        async def mount():
            provider._tool_schemas.append(schema)

        with patch.object(provider, "_start_mcp", mount):
            await provider.start()
            await provider.shutdown()
            await provider.start()
        self.assertEqual(provider._tool_schemas.count(schema), 1)

    async def test_turn_deadline_includes_startup(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name)
        started = asyncio.Event()

        async def blocked_start():
            started.set()
            await asyncio.Future()

        with patch.object(provider, "start", blocked_start), self.assertRaises(TimeoutError):
            async for event in provider.stream("测试初始化截止时间", timeout=0):
                self.fail("超时请求不应生成事件.")
        self.assertTrue(started.is_set())
        self.assertIsNone(provider._task)

    async def test_closing_stream_clears_active_turn(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name)
        started = asyncio.Event()

        async def blocked_start():
            started.set()
            await asyncio.Future()

        async def consume():
            async for event in provider.stream("测试启动取消"):
                self.fail("启动未完成不应返回事件.")

        with patch.object(provider, "start", blocked_start):
            task = asyncio.create_task(consume())
            try:
                async with asyncio.timeout(30):
                    await started.wait()
                    self.assertTrue(provider.has_active_turn())
                    self.assertEqual(await provider.cancel(), "acked")
                self.assertTrue(task.cancelled())
                self.assertFalse(provider.has_active_turn())
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_closing_tool_stream_clears_active_turn(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name)
        self.addAsyncCleanup(provider.shutdown)
        stream = provider.stream("测试关闭流")
        try:
            async with asyncio.timeout(30):
                event = await anext(stream)
            self.assertEqual(event.kind, "tool_call")
            self.assertTrue(provider.has_active_turn())
        finally:
            await stream.aclose()
        self.assertFalse(provider.has_active_turn())

    async def test_mcp_skips_cancelled_queue_entry(self):
        called = []

        class Session:
            async def initialize(self):
                return None

            async def list_tools(self, cursor=None):
                return SimpleNamespace(tools=[], nextCursor=None)

            async def call_tool(self, name, arguments):
                called.append(name)
                return "完成"

        @asynccontextmanager
        async def transport(*args, **kwargs):
            self.assertEqual(args[0].env["KIROCREW_SESSION_KEY"], "current-session")
            yield (None, None)

        @asynccontextmanager
        async def session_context(*args, **kwargs):
            yield Session()

        connection = McpConnection("test", {"command": "unused", "env": {"KIROCREW_SESSION_KEY": "old-session"}}, {"KIROCREW_SESSION_KEY": "current-session"})
        cancelled = asyncio.get_running_loop().create_future()
        cancelled.cancel()
        await connection.queue.put(("cancelled", {}, cancelled))
        with (
            patch("kiro_crew.providers.direct_mcp.stdio_client", transport),
            patch("kiro_crew.providers.direct_mcp.ClientSession", session_context),
        ):
            try:
                await connection.start()
                async with asyncio.timeout(30):
                    self.assertEqual(await connection.call("active", {}), "完成")
            finally:
                await connection.close()
        self.assertEqual(called, ["active"])

    async def test_mcp_tool_timeout_releases_queue(self):
        await self._check_mcp_recovery(cancel=False)

    async def test_mcp_cancellation_releases_queue(self):
        await self._check_mcp_recovery(cancel=True)

    async def _check_mcp_recovery(self, *, cancel):
        called = []
        started = asyncio.Event()
        stopped = asyncio.Event()

        class Session:
            async def initialize(self):
                return None

            async def list_tools(self, cursor=None):
                return SimpleNamespace(tools=[], nextCursor=None)

            async def call_tool(self, name, arguments):
                called.append(name)
                if name == "blocked":
                    if cancel:
                        started.set()
                        try:
                            await asyncio.Future()
                        finally:
                            stopped.set()
                    raise TimeoutError("工具执行超时.")
                return "完成"

        @asynccontextmanager
        async def transport(*args, **kwargs):
            yield (None, None)

        @asynccontextmanager
        async def session_context(*args, **kwargs):
            yield Session()

        connection = McpConnection("test", {"command": "unused"}, {})
        with (
            patch("kiro_crew.providers.direct_mcp.stdio_client", transport),
            patch("kiro_crew.providers.direct_mcp.ClientSession", session_context),
        ):
            try:
                await connection.start()
                if cancel:
                    caller = asyncio.create_task(connection.call("blocked", {}))
                    try:
                        async with asyncio.timeout(30):
                            await started.wait()
                            caller.cancel()
                            await asyncio.gather(caller, return_exceptions=True)
                            await stopped.wait()
                    finally:
                        caller.cancel()
                        await asyncio.gather(caller, return_exceptions=True)
                else:
                    with (
                        self.assertLogs("kiro_crew.providers.direct_mcp", level="ERROR"),
                        self.assertRaises(TimeoutError),
                    ):
                        await connection.call("blocked", {})
                async with asyncio.timeout(30):
                    self.assertEqual(await connection.call("active", {}), "完成")
            finally:
                await connection.close()
        self.assertEqual(called, ["blocked", "active"])

    async def test_configured_model_switch_and_reasoning_effort(self):
        settings = {
            **self.settings,
            "models": [{"id": "test", "reasoning_effort": ["low", "high"]}],
        }
        provider = DirectProvider(settings=settings, work_dir=self.root.name)
        self.assertIsInstance(provider.available_models(), list)
        self.assertTrue(await provider.change_effort("high"))
        self.assertFalse(await provider.change_effort("unknown"))
        await provider.set_model("other")
        self.assertEqual(provider.served_model, "other")
        self.assertEqual(provider._effort, "")
        await provider.set_model("auto")
        self.assertEqual(provider.served_model, "test")
        self.assertTrue(await provider.change_effort("high"))
        await provider.set_model("test")
        self.assertEqual(provider._effort, "high")
        await provider.clear_effort()
        self.assertEqual(provider._effort, "")

    async def test_initial_effort_matches_configured_model(self):
        settings = {**self.settings, "models": [{"id": "test", "reasoning_effort": ["low"]}]}
        provider = DirectProvider(settings=settings, reasoning_effort="max")
        self.assertEqual(provider._effort, "")
        supported = DirectProvider(settings=settings, reasoning_effort="low")
        self.assertEqual(supported._effort, "low")

    async def test_session_map_roundtrip_restores_direct_history(self):
        from kiro_crew.session import _provider_label
        from kiro_crew.session_map import SessionMap

        provider = DirectProvider(settings=self.settings, work_dir=self.root.name)
        self.addAsyncCleanup(provider.shutdown)
        await provider.start()
        provider._messages = [{"role": "user", "content": "DIRECT_RESTORE_MARKER"}]
        await provider._save()
        mapping = SessionMap()
        mapping.set("dashboard:direct-restore", provider.session_id, provider=_provider_label(provider))
        async with asyncio.timeout(30):
            await mapping.aflush()
        reloaded = SessionMap()
        session_id = reloaded.get("dashboard:direct-restore")
        self.assertEqual(session_id, provider.session_id)
        self.assertEqual(reloaded.get_provider("dashboard:direct-restore"), "direct")
        resumed = DirectProvider(settings=self.settings, work_dir=self.root.name)
        self.addAsyncCleanup(resumed.shutdown)
        resumed.set_resume_session_id(session_id)
        await resumed.start()
        self.assertTrue(resumed.resumed)
        self.assertEqual(resumed._messages, provider._messages)
        await provider.cleanup_session(provider.session_id)
        self.assertIsNone(reloaded.get("dashboard:direct-restore"))

    async def test_manager_shutdown_preserves_direct_resume(self):
        from kiro_crew.config.loader import KiroCrewConfig
        from kiro_crew.session import SessionManager

        config = KiroCrewConfig()
        config.agent.direct_model = self.settings
        config.session.pool_size = 0

        def factory(session_key=None, **kwargs):
            return DirectProvider(settings=self.settings, work_dir=self.root.name, session_key=session_key)

        manager = SessionManager(config, provider_factory=factory)
        self.addAsyncCleanup(manager.close_all)
        async with asyncio.timeout(30):
            provider, _, _ = await manager.get_or_create("dashboard:manager-restore")
            provider._messages = [{"role": "user", "content": "MANAGER_RESTORE_MARKER"}]
            await provider._save()
            session_id = provider.session_id
            manager.release("dashboard:manager-restore")
            await manager.close_all()
            replacement = SessionManager(config, provider_factory=factory)
            self.addAsyncCleanup(replacement.close_all)
            resumed, _, restored = await replacement.get_or_create("dashboard:manager-restore")
            try:
                self.assertTrue(restored)
                self.assertEqual(resumed.session_id, session_id)
                self.assertEqual(resumed._messages[0]["content"], "MANAGER_RESTORE_MARKER")
            finally:
                replacement.release("dashboard:manager-restore")

    async def test_task_step_uses_direct_provider(self):
        from kiro_crew.config.loader import KiroCrewConfig
        from kiro_crew.session import SessionManager

        config = KiroCrewConfig()
        config.session.pool_size = 0
        manager = SessionManager(config, provider_factory=lambda session_key=None, **kwargs: DirectProvider(settings=self.settings, work_dir=self.root.name, session_key=session_key))
        self.addAsyncCleanup(manager.close_all)
        async with asyncio.timeout(30):
            provider, _, _ = await manager.open_task_session("task:parent", "task:step", cwd=self.root.name)
            try:
                self.assertIsInstance(provider, DirectProvider)
                self.assertTrue(provider.is_alive())
            finally:
                manager.release("task:step")

    async def test_background_prompt_deadline_covers_queue(self):
        from kiro_crew.session_background import _ProviderBgSession

        provider = DirectProvider(settings=self.settings)
        entry = SimpleNamespace(provider=provider, semaphore=asyncio.Semaphore(0))
        handle = _ProviderBgSession(entry)
        with self.assertRaises(TimeoutError):
            async for event in handle.prompt("测试后台排队截止时间", timeout=0):
                self.fail("未取得会话锁不应生成事件.")
        self.assertFalse(handle._sem_held)

    async def test_explicit_history_cleanup_removes_only_target(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name)
        self.addAsyncCleanup(provider.shutdown)
        await provider.start()
        provider._messages = [{"role": "user", "content": "test"}]
        await provider._save()
        other = provider._history_path.with_name("b" * 64 + ".json")
        other.write_text("{}")
        await provider.cleanup_session(provider.session_id)
        self.assertFalse(provider._history_path.exists())
        self.assertTrue(other.exists())
        with self.assertRaises(ValueError):
            await provider.cleanup_session("../other")

    async def test_read_only_tools_refuse_mutation(self):
        provider = DirectProvider(
            settings=self.settings, work_dir=self.root.name, side_read_only=True
        )
        with self.assertRaises(ValueError):
            await provider._execute("write_file", {"path": "blocked", "content": "x"})
        with self.assertRaises(ValueError):
            await provider._execute("execute_bash", {"command": "true"})
        self.assertFalse(Path(self.root.name, "blocked").exists())

    async def test_cancel_active_turn(self):
        provider = DirectProvider(settings=self.settings)
        started = asyncio.Event()

        async def wait_forever():
            provider._task = asyncio.current_task()
            started.set()
            await asyncio.Future()

        task = asyncio.create_task(wait_forever())
        await started.wait()
        self.assertEqual(await provider.cancel(), "acked")
        self.assertTrue(task.cancelled())

    async def test_cancel_has_bounded_cleanup_wait(self):
        provider = DirectProvider(settings=self.settings)
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_cleanup():
            started.set()
            try:
                await asyncio.Future()
            finally:
                await release.wait()

        task = asyncio.create_task(blocked_cleanup())
        provider._task = task
        await started.wait()
        try:
            with patch(
                "kiro_crew.providers.direct.asyncio.wait", return_value=(set(), {task})
            ) as wait:
                self.assertEqual(await provider.cancel(wait_ack_timeout=0.5), "timeout")
                wait.assert_called_once_with({task}, timeout=0.5)
        finally:
            release.set()
            async with asyncio.timeout(30):
                await asyncio.gather(task, return_exceptions=True)

    async def test_settings_save_preserves_key_and_validates_models(self):
        app = web.Application()
        app.router.add_get("/config", api_direct_model)
        app.router.add_put("/config", api_direct_model)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        payload = {
            "url": self.settings["url"],
            "model": "test",
            "models": [{"id": "test"}],
            "api_key": "test-secret",
        }
        response = await client.put("/config", json=payload)
        self.assertEqual(response.status, 200)
        payload["api_key"] = ""
        response = await client.put("/config", json=payload)
        self.assertEqual(response.status, 200)
        response = await client.get("/config")
        saved = await response.json()
        self.assertTrue(saved["has_api_key"])
        self.assertNotIn("api_key", saved)
        payload["models"] = [{"id": "other"}]
        response = await client.put("/config", json=payload)
        self.assertEqual(response.status, 400)
        payload["models"] = [{"id": "test"}]
        with patch(
            "kiro_crew.dashboard.handlers.direct_model.update_config_locked",
            side_effect=BlockingIOError,
        ) as update:
            response = await client.put("/config", json=payload)
            self.assertEqual(response.status, 409)
            self.assertIn("稍后重试", (await response.json())["error"])
            self.assertFalse(update.call_args.kwargs["wait_for_lock"])

    async def test_compaction_preserves_summary_and_persists(self):
        provider = DirectProvider(
            settings=self.settings, work_dir=self.root.name, session_key="compact"
        )
        self.addAsyncCleanup(provider.shutdown)
        await provider.start()
        provider._messages = [{"role": "user", "content": "SUMMARY_TOKEN_42"}]
        await provider.compact()
        self.assertIn("SUMMARY_TOKEN_42", provider._messages[0]["content"])
        self.assertEqual((await provider.wait_for_compaction())["type"], "completed")
        saved = json.loads(provider._history_path.read_text())
        self.assertEqual(saved["messages"], provider._messages)

    async def test_tool_cancellation_keeps_paired_history(self):
        provider = DirectProvider(
            settings=self.settings, work_dir=self.root.name, session_key="cancel"
        )
        self.addAsyncCleanup(provider.shutdown)
        executing = asyncio.Event()

        async def blocked_tool(name, args):
            executing.set()
            await asyncio.Future()

        async def consume():
            async for event in provider.stream("执行工具"):
                pass

        with patch.object(provider, "_execute", blocked_tool):
            task = asyncio.create_task(consume())
            await executing.wait()
            self.assertEqual(await provider.cancel(), "acked")
            self.assertTrue(task.cancelled())
        self.assertEqual(provider._messages[-1]["role"], "tool")
        self.assertEqual(provider._messages[-1]["tool_call_id"], "call_1")
        self.assertIn("结果未知", provider._messages[-1]["content"])
        self.assertEqual(
            json.loads(provider._history_path.read_text())["messages"], provider._messages
        )

    async def test_no_token_http_and_websocket(self):
        app = web.Application(middlewares=[token_auth_middleware()])

        async def identity(request):
            return web.json_response(
                {"user": request["user"], "owner": request["is_dashboard_user"]}
            )

        async def socket(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json({"user": request["user"]})
            await ws.close()
            return ws

        app.router.add_get("/identity", identity)
        app.router.add_get("/socket", socket)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        response = await client.get("/identity")
        self.assertEqual(await response.json(), {"user": "local-owner", "owner": True})
        async with client.ws_connect("/socket") as ws:
            self.assertEqual(await ws.receive_json(), {"user": "local-owner"})

    async def test_internal_mcp_identity_is_preserved(self):
        app = web.Application(middlewares=[token_auth_middleware(internal_secret="test-secret")])

        async def identity(request):
            return web.json_response(
                {"internal": request.get("internal_auth", False), "app_present": "app" in request}
            )

        app.router.add_get("/identity", identity)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        response = await client.get("/identity", headers={"X-Internal-Secret": "test-secret"})
        self.assertEqual(await response.json(), {"internal": True, "app_present": False})

    async def test_incognito_mode_does_not_persist_history(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name)
        self.addAsyncCleanup(provider.shutdown)
        provider.memory_mode = "incognito"
        await provider.start()
        provider._messages = [{"role": "user", "content": "private"}]
        await provider._save()
        self.assertFalse(provider._history_path.exists())

    async def test_agent_with_no_tools_rejects_builtin_execution(self):
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name)
        provider._tool_schemas = []
        with self.assertRaisesRegex(ValueError, "未挂载"):
            await provider._execute("execute_bash", {"command": "true"})

    async def test_markdown_agent_spec_and_json_priority(self):
        directory = Path(self.root.name) / ".kiro" / "agents"
        directory.mkdir(parents=True)
        (directory / "custom.md").write_text(
            '---\nname: custom\ntools: []\n---\n测试提示词.\n', encoding="utf-8"
        )
        provider = DirectProvider(settings=self.settings, work_dir=self.root.name, agent="custom")
        self.addAsyncCleanup(provider.shutdown)
        await provider.start()
        self.assertEqual(provider._tool_schemas, [])
        self.assertIn("测试提示词", provider._spec["prompt"])
        await provider.shutdown()
        (directory / "custom.json").write_text('{"tools":["fs_read"]}', encoding="utf-8")
        await provider.start()
        self.assertEqual(
            {tool["function"]["name"] for tool in provider._tool_schemas},
            {"read_file", "list_directory"},
        )


if __name__ == "__main__":
    unittest.main()

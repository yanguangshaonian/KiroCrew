"""内置 Agent 的 MCP 客户端生命周期."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class McpConnection:
    def __init__(self, name: str, spec: dict, env: dict[str, str]):
        self.name = name
        self.spec = spec
        self.env = env
        self.tools = []
        self.queue = asyncio.Queue()
        self.task = None
        self.ready = None

    async def start(self):
        self.ready = asyncio.get_running_loop().create_future()
        self.task = asyncio.create_task(self._run())
        try:
            async with asyncio.timeout(60):
                await self.ready
        except BaseException:
            await self.close()
            raise

    async def _run(self):
        assert self.ready is not None
        pending = None
        try:
            async with AsyncExitStack() as stack:
                if self.spec.get("url"):
                    streams = await stack.enter_async_context(
                        streamablehttp_client(self.spec["url"], headers=self.spec.get("headers"))
                    )
                    read, write = streams[:2]
                else:
                    env = {**os.environ, **self.spec.get("env", {}), **self.env}
                    params = StdioServerParameters(
                        command=self.spec["command"], args=self.spec.get("args", []), env=env
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                cursor = None
                while True:
                    result = await session.list_tools(cursor=cursor)
                    self.tools.extend(result.tools)
                    cursor = result.nextCursor
                    if not cursor:
                        break
                self.ready.set_result(None)
                while True:
                    item = await self.queue.get()
                    if item is None:
                        return
                    name, args, pending = item
                    if pending.done():
                        pending = None
                        continue
                    request_task = asyncio.create_task(session.call_tool(name, arguments=args))

                    def cancel_request(future, task=request_task):
                        if future.cancelled():
                            task.cancel()

                    pending.add_done_callback(cancel_request)
                    try:
                        async with asyncio.timeout(180):
                            result = await request_task
                        if not pending.done():
                            pending.set_result(result)
                    except asyncio.CancelledError:
                        if not pending.cancelled():
                            raise
                    except Exception as error:
                        logger.exception("MCP 工具调用失败: %s/%s.", self.name, name)
                        if not pending.done():
                            pending.set_exception(error.with_traceback(None))
                    finally:
                        pending.remove_done_callback(cancel_request)
                        if not request_task.done():
                            request_task.cancel()
                        await asyncio.gather(request_task, return_exceptions=True)
                    pending = None
        except BaseException as error:
            logger.debug("MCP 连接结束: %s.", self.name, exc_info=True)
            if not self.ready.done():
                self.ready.set_exception(error)
            if pending is not None and not pending.done():
                pending.set_exception(RuntimeError(f"MCP 连接已关闭: {self.name}."))
        finally:
            while not self.queue.empty():
                item = self.queue.get_nowait()
                if item is not None and not item[2].done():
                    item[2].set_exception(RuntimeError(f"MCP 连接已关闭: {self.name}."))

    async def call(self, name: str, args: dict):
        if self.task is None or self.task.done():
            raise ValueError(f"MCP 服务未连接: {self.name}.")
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((name, args, future))
        async with asyncio.timeout(180):
            return await future

    async def close(self):
        if self.task is None:
            return
        await self.queue.put(None)
        try:
            async with asyncio.timeout(10):
                await asyncio.shield(self.task)
        except TimeoutError:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.task = None

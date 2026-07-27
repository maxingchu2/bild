"""MCP 层：外部工具协议。

以 MCP 风格注册外部 server 提供的工具（配置级生命周期），
当前内置一个 "main-service" server：把主业务服务 /api/ai/chat/classify
以 MCP tool 形式暴露给主 agent，结果进入主上下文。
"""

import json
from typing import Any, AsyncGenerator, Dict, List

import httpx

from . import config


class MCPServer:
    """一个 MCP server 的最小抽象：名称 + 工具清单 + 调用"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools: Dict[str, Dict[str, Any]] = {}

    def add_tool(self, name: str, description: str) -> None:
        self.tools[name] = {"name": name, "description": description}

    def list_tools(self) -> List[Dict[str, Any]]:
        return list(self.tools.values())


class MainServiceMCP(MCPServer):
    """内置 MCP server：船检主服务"""

    def __init__(self) -> None:
        super().__init__("main-service", "船检主服务（SSE 动作执行）")
        self.add_tool("classify_stream", "转发请求到主服务并流式返回 SSE")

    async def classify_stream(
        self, payload: Dict[str, Any], auth: str = ""
    ) -> AsyncGenerator[bytes, None]:
        headers = {"Content-Type": "application/json",
                   "Accept": "text/event-stream"}
        if auth:
            headers["Authorization"] = auth
        async with httpx.AsyncClient(
            base_url=config.MAIN_BASE, timeout=180
        ) as client:
            async with client.stream(
                "POST", "/api/ai/chat/classify", json=payload, headers=headers
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk


class MCPManager:
    def __init__(self) -> None:
        self.servers: Dict[str, MCPServer] = {}

    def register(self, server: MCPServer) -> None:
        self.servers[server.name] = server

    def get(self, name: str) -> MCPServer:
        return self.servers[name]

    def prompt_index(self) -> str:
        lines = ["已接入的 MCP server："]
        for s in self.servers.values():
            tool_names = ", ".join(t["name"] for t in s.list_tools())
            lines.append(f"- {s.name}（{s.description}）: {tool_names}")
        return "\n".join(lines)


mcp_manager = MCPManager()
main_service = MainServiceMCP()
mcp_manager.register(main_service)

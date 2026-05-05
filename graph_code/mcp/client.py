"""Plugin manifest loader and MCP client routing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from ..tools.schema import ToolResultEnvelope


class MCPClient(Protocol):
    def connect(self) -> None: ...
    def list_tools(self) -> list[str]: ...
    def call_tool(self, tool: str, args: dict[str, Any]) -> ToolResultEnvelope: ...


class MockMCPClient:
    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def list_tools(self) -> list[str]:
        return list((self.config.get("tools") or {}).keys())

    def call_tool(self, tool: str, args: dict[str, Any]) -> ToolResultEnvelope:
        tools = self.config.get("tools") or {}
        if tool not in tools:
            return ToolResultEnvelope.error(
                f"MCP tool not found: {tool}",
                metadata={"server": self.name, "tool_name": f"mcp__{self.name}__{tool}"},
            )
        spec = tools[tool]
        response = spec.get("response", args)
        return ToolResultEnvelope.success(
            str(response),
            metadata={"server": self.name, "tool_name": f"mcp__{self.name}__{tool}"},
        )


class SDKMCPClient:
    """MCP Python SDK client for stdio and Streamable HTTP transports.

    Imports are lazy so the project remains usable without optional MCP
    dependencies until a real MCP transport is configured.
    """

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self._portal_cm = None
        self._portal = None
        self._stack = None
        self._session = None

    def connect(self) -> None:
        self._validate_auth()
        try:
            from anyio.from_thread import start_blocking_portal
        except ImportError as exc:
            raise RuntimeError("MCP SDK transport requires anyio.") from exc
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("MCP SDK transport requires installing mcp[cli].") from exc

        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._portal.call(self._aconnect)

    def _validate_auth(self) -> None:
        auth = self.config.get("auth") or {}
        if auth.get("type") == "bearer":
            token_env = auth.get("token_env")
            if not token_env or not os.getenv(token_env):
                raise RuntimeError(f"needs-auth: missing bearer token env {token_env}")

    async def _aconnect(self) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession

        self._stack = AsyncExitStack()
        transport = self.config.get("transport")
        if transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self.config["command"],
                args=self.config.get("args", []),
                env=self.config.get("env"),
                cwd=self.config.get("cwd"),
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif transport in {"streamable-http", "http"}:
            from mcp.client.streamable_http import streamable_http_client

            headers = dict(self.config.get("headers") or {})
            auth = self.config.get("auth") or {}
            if auth.get("type") == "bearer":
                headers["Authorization"] = f"Bearer {os.environ[auth['token_env']]}"
            context = streamable_http_client(self.config["url"], headers=headers)
            read, write, _ = await self._stack.enter_async_context(context)
        else:
            raise ValueError(f"Unsupported MCP transport: {transport}")

        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    def list_tools(self) -> list[str]:
        async def _list_tools():
            result = await self._session.list_tools()
            return [tool.name for tool in result.tools]

        return self._portal.call(_list_tools)

    def call_tool(self, tool: str, args: dict[str, Any]) -> ToolResultEnvelope:
        async def _call_tool():
            return await self._session.call_tool(tool, arguments=args)

        try:
            result = self._portal.call(_call_tool)
            return self._to_envelope(tool, result)
        except Exception as exc:
            return ToolResultEnvelope.error(
                f"MCP call failed: {type(exc).__name__}: {exc}",
                metadata={"server": self.name, "tool_name": f"mcp__{self.name}__{tool}"},
            )

    def _to_envelope(self, tool: str, result: Any) -> ToolResultEnvelope:
        is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
        structured = getattr(result, "structuredContent", None) or getattr(
            result, "structured_content", None
        )
        if structured is not None:
            content = json.dumps(structured, ensure_ascii=False)
        else:
            parts = []
            for block in getattr(result, "content", []) or []:
                text = getattr(block, "text", None)
                parts.append(text if text is not None else str(block))
            content = "\n".join(parts)
        return ToolResultEnvelope(
            ok=not is_error,
            is_error=is_error,
            content=content,
            metadata={"server": self.name, "tool_name": f"mcp__{self.name}__{tool}"},
        )

    def close(self) -> None:
        if self._portal and self._stack:
            self._portal.call(self._stack.aclose)
        if self._portal_cm:
            self._portal_cm.__exit__(None, None, None)


class MCPClientRegistry:
    """Routes names shaped as mcp__{server}__{tool}."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.clients: dict[str, MCPClient] = {}
        self.connection_state: dict[str, dict[str, str]] = {}

    def load_manifest(self, manifest: dict[str, Any]) -> None:
        for server, config in (manifest.get("servers") or {}).items():
            if config.get("disabled"):
                self.connection_state[server] = {"status": "disabled"}
                continue
            transport = config.get("transport")
            if transport == "mock":
                client = MockMCPClient(server, config)
            elif transport in {"stdio", "streamable-http", "http"}:
                client = SDKMCPClient(server, config)
            else:
                self.connection_state[server] = {"status": "failed"}
                continue
            try:
                client.connect()
                self.connection_state[server] = {"status": "connected"}
            except RuntimeError as exc:
                status = "needs-auth" if "needs-auth" in str(exc) else "failed"
                self.connection_state[server] = {"status": status, "error": str(exc)}
                continue
            self.clients[server] = client

    def load_manifest_file(self, path: str | Path) -> None:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        self.load_manifest(manifest)

    def list_tools(self) -> list[str]:
        names: list[str] = []
        for server, client in self.clients.items():
            for tool in client.list_tools():
                names.append(f"mcp__{server}__{tool}")
        return names

    def call_tool(self, routed_name: str, args: dict[str, Any]) -> ToolResultEnvelope:
        parts = routed_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            return ToolResultEnvelope.error(f"Invalid MCP tool name: {routed_name}")
        _, server, tool = parts
        client = self.clients.get(server)
        if client is None:
            self.connection_state.setdefault(server, {"status": "failed"})
            return ToolResultEnvelope.error(f"MCP server not connected: {server}")
        return client.call_tool(tool, args)

"""MCP (Model Context Protocol) client integration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class MCPTool:
    """Represents an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class MCPResource:
    """Represents an MCP resource."""

    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None


class MCPClient:
    """Client for connecting to MCP servers."""

    def __init__(self) -> None:
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, tuple[str, MCPTool]] = (
            {}
        )  # tool_name -> (server_name, tool)
        self._cleanup_tasks: list[Any] = []

    @asynccontextmanager
    async def connect_stdio(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[ClientSession]:
        """Connect to an MCP server via stdio.

        Args:
            name: A name for this server connection
            command: The command to run
            args: Command arguments
            env: Environment variables

        Yields:
            The MCP client session
        """
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._sessions[name] = session

                # Discover and cache tools from this server
                await self._discover_tools(name, session)

                try:
                    yield session
                finally:
                    del self._sessions[name]
                    # Remove tools from this server
                    self._tools = {k: v for k, v in self._tools.items() if v[0] != name}

    async def _discover_tools(self, server_name: str, session: ClientSession) -> None:
        """Discover tools from an MCP server."""
        try:
            result = await session.list_tools()
            for tool in result.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                )
                self._tools[tool.name] = (server_name, mcp_tool)
        except Exception:
            # Server might not support tools
            pass

    def get_tools(self) -> list[MCPTool]:
        """Get all available tools from connected servers."""
        return [tool for _, tool in self._tools.values()]

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI function calling format."""
        return [tool.to_openai_tool() for tool in self.get_tools()]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call a tool by name.

        Args:
            name: The tool name
            arguments: The tool arguments

        Returns:
            The tool result
        """
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        server_name, _ = self._tools[name]
        session = self._sessions[server_name]

        result = await session.call_tool(name, arguments)
        return result

    async def list_resources(self, server_name: str | None = None) -> list[MCPResource]:
        """List resources from MCP servers.

        Args:
            server_name: Optional specific server to query, or all if None

        Returns:
            List of available resources
        """
        resources = []
        sessions = (
            {server_name: self._sessions[server_name]}
            if server_name
            else self._sessions
        )

        for session in sessions.values():
            try:
                result = await session.list_resources()
                for resource in result.resources:
                    resources.append(
                        MCPResource(
                            uri=str(resource.uri),
                            name=resource.name,
                            description=resource.description,
                            mime_type=resource.mimeType,
                        )
                    )
            except Exception:
                # Server might not support resources
                pass

        return resources

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI.

        Args:
            uri: The resource URI

        Returns:
            The resource content
        """
        # Try each session until we find one that can handle this URI
        for session in self._sessions.values():
            try:
                result = await session.read_resource(uri)
                if result.contents:
                    content = result.contents[0]
                    if hasattr(content, "text"):
                        return content.text
                    return str(content)
            except Exception:
                continue

        raise ValueError(f"Could not read resource: {uri}")

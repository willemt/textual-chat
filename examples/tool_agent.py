#!/usr/bin/env python
"""Agent with tool calls for testing ACP integration."""

import asyncio
from uuid import uuid4

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import (
    AudioContentBlock,
    AuthenticateResponse,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    ResourceContentBlock,
    SetSessionModeResponse,
    SetSessionModelResponse,
    SseMcpServer,
    TextContentBlock,
)


class ToolAgent(Agent):
    """Agent that demonstrates tool usage."""

    _conn: Client

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: object,
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        **kwargs: object,
    ) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: object,
    ) -> PromptResponse:
        # Get user text
        user_text = ""
        for block in prompt:
            user_text = getattr(block, "text", "")

        # Send initial text
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(text_block("Let me look that up... ")),
        )

        # Simulate a tool call
        tool_call_id = f"call_{uuid4().hex[:8]}"

        # Start tool call
        await self._conn.session_update(
            session_id=session_id,
            update=start_tool_call(
                tool_call_id,
                title="search",
                kind="fetch",
                status="in_progress",
                raw_input={"query": user_text},
            ),
        )

        # Simulate work
        await asyncio.sleep(1)

        # Complete tool call
        await self._conn.session_update(
            session_id=session_id,
            update=update_tool_call(
                tool_call_id,
                status="completed",
                content=[tool_content(text_block(f"Found info about: {user_text}"))],
                raw_output={"result": "success"},
            ),
        )

        # Send final response
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(
                text_block(f"Based on my search, here's what I found about '{user_text}'!")
            ),
        )

        return PromptResponse(stop_reason="end_turn")

    async def authenticate(self, method_id: str, **kwargs: object) -> AuthenticateResponse | None:
        return None

    async def cancel(self, session_id: str, **kwargs: object) -> None:
        pass

    async def list_sessions(
        self, cursor: str | None = None, cwd: str | None = None, **kwargs: object
    ) -> ListSessionsResponse:
        return ListSessionsResponse(sessions=[])

    async def load_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        session_id: str,
        **kwargs: object,
    ) -> LoadSessionResponse | None:
        return None

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs: object
    ) -> SetSessionModeResponse | None:
        return None

    async def set_session_model(
        self, model_id: str, session_id: str, **kwargs: object
    ) -> SetSessionModelResponse | None:
        return None

    async def ext_method(self, method: str, params: dict[str, object]) -> dict[str, object]:
        return {}

    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        pass


async def main() -> None:
    await run_agent(ToolAgent())


if __name__ == "__main__":
    asyncio.run(main())

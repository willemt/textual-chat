#!/usr/bin/env python
"""Simple echo agent for testing ACP integration."""

import asyncio
from uuid import uuid4

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message,
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


class EchoAgent(Agent):
    """Simple agent that echoes back user messages."""

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
        # Echo back each text block
        for block in prompt:
            text = getattr(block, "text", "")

            if text:
                # Stream response word by word for demo
                response = f"You said: {text}"
                for word in response.split():
                    await self._conn.session_update(
                        session_id=session_id,
                        update=update_agent_message(text_block(word + " ")),
                    )
                    await asyncio.sleep(0.05)  # Small delay for streaming effect

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
    await run_agent(EchoAgent())


if __name__ == "__main__":
    asyncio.run(main())

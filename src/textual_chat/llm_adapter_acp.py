"""ACP adapter - provides llm-like interface over Agent Client Protocol.

This adapter makes ACP agents look similar to the litellm adapter,
allowing chat.py to work with either backend.

Usage:
    model = get_async_model("python path/to/agent.py")
    conv = model.conversation()

    async for chunk in conv.chain("Hello"):
        print(chunk)

Requires: pip install agent-client-protocol
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypedDict

from acp import PROTOCOL_VERSION, Client, connect_to_agent, text_block
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AgentPlanUpdate,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CurrentModeUpdate,
    Implementation,
    TextContentBlock,
    ToolCallStart,
    ToolCallProgress,
    UserMessageChunk,
)

log = logging.getLogger(__name__)
# log.propagate = False  # TEMPORARILY ENABLED FOR DEBUGGING


class CacheDetails(TypedDict, total=False):
    """Cache-related token details (not used in ACP but kept for interface compat)."""

    cached_tokens: int


@dataclass
class Usage:
    """Token usage information."""

    input: int = 0
    output: int = 0
    details: CacheDetails = field(default_factory=dict)


@dataclass
class ToolCall:
    """Represents a tool call from the agent."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    output: str


def get_async_model(
    agent_command: str,
    *,
    api_key: str | None = None,  # Not used for ACP, kept for interface compat
    api_base: str | None = None,  # Not used for ACP, kept for interface compat
) -> AsyncModel:
    """Get an async model that connects to an ACP agent.

    Args:
        agent_command: Command to spawn the agent (e.g., "python agent.py" or path to executable)
    """
    return AsyncModel(agent_command)


class AsyncModel:
    """Async model interface for ACP agents."""

    def __init__(self, agent_command: str):
        self.model_id = agent_command
        self.agent_command = agent_command
        self.is_claude = False  # ACP agents handle their own caching

    def conversation(self) -> AsyncConversation:
        """Create a new conversation (ACP session)."""
        return AsyncConversation(self)

    # Alias for ACP terminology
    session = conversation


class ACPClientHandler(Client):
    """Handles ACP client callbacks for session updates."""

    def __init__(self):
        self._text_chunks: asyncio.Queue[str | None] = asyncio.Queue()
        self._tool_calls: dict[str, ToolCall] = {}
        self._before_call: Callable[[ToolCall], Any] | None = None
        self._after_call: Callable[[ToolCall, ToolResult], Any] | None = None

    def reset_for_turn(
        self,
        before_call: Callable[[ToolCall], Any] | None = None,
        after_call: Callable[[ToolCall, ToolResult], Any] | None = None,
    ) -> None:
        """Reset state for a new turn."""
        self._text_chunks = asyncio.Queue()
        self._tool_calls = {}
        self._before_call = before_call
        self._after_call = after_call

    async def session_update(
        self,
        session_id: str,
        update: (
            UserMessageChunk
            | AgentMessageChunk
            | AgentThoughtChunk
            | ToolCallStart
            | ToolCallProgress
            | AgentPlanUpdate
            | AvailableCommandsUpdate
            | CurrentModeUpdate
        ),
        **kwargs: Any,
    ) -> None:
        """Handle session updates from the agent."""
        if isinstance(update, AgentMessageChunk):
            # Text streaming
            content = update.content
            if isinstance(content, TextContentBlock):
                await self._text_chunks.put(content.text)

        elif isinstance(update, (ToolCallStart, ToolCallProgress)):
            # Tool call update
            tool_call_id = update.tool_call_id or ""
            status = update.status or ""

            # Create or get tool call
            if tool_call_id not in self._tool_calls:
                tc = ToolCall(
                    id=tool_call_id,
                    name=update.title or "",
                    arguments=update.raw_input if isinstance(update.raw_input, dict) else {},
                )
                self._tool_calls[tool_call_id] = tc

                # Fire before callback for new tool calls
                if self._before_call:
                    result = self._before_call(tc)
                    if asyncio.iscoroutine(result):
                        await result

            tc = self._tool_calls[tool_call_id]

            # Fire after callback when completed
            if status in ("completed", "failed") and self._after_call:
                output = ""
                if update.raw_output:
                    output = str(update.raw_output)
                result = self._after_call(tc, ToolResult(tc.id, output))
                if asyncio.iscoroutine(result):
                    await result

    async def request_permission(self, options: Any, session_id: str, tool_call: Any, **kwargs: Any):
        """Handle permission requests - auto-approve for now."""
        return {"outcome": {"outcome": "approved"}}

    # File system methods - raise not found for now
    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any):
        from acp import RequestError

        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(
        self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **kwargs: Any
    ):
        from acp import RequestError

        raise RequestError.method_not_found("fs/read_text_file")

    # Terminal methods - raise not found for now
    async def create_terminal(self, command: str, session_id: str, **kwargs: Any):
        from acp import RequestError

        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any):
        from acp import RequestError

        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any):
        from acp import RequestError

        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any):
        from acp import RequestError

        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any):
        from acp import RequestError

        raise RequestError.method_not_found("terminal/kill")

    def signal_done(self) -> None:
        """Signal that streaming is complete."""
        self._text_chunks.put_nowait(None)


class AsyncConversation:
    """Manages conversation with an ACP agent."""

    def __init__(self, model: AsyncModel):
        self.model = model
        self._session_id: str | None = None
        self._conn: Any = None
        self._proc: aio_subprocess.Process | None = None
        self._client: ACPClientHandler | None = None
        self.agent_name: str | None = None  # Agent name from initialization

    def chain(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tools: list[Callable] | None = None,
        before_call: Callable[[ToolCall], Any] | None = None,
        after_call: Callable[[ToolCall, ToolResult], Any] | None = None,
        options: dict | None = None,
    ) -> AsyncChainResponse:
        """Execute prompt with the ACP agent.

        Args:
            prompt: User message
            system: System prompt (may not be supported by all agents)
            tools: Not used for ACP (agent has its own tools)
            before_call: Called when agent starts a tool
            after_call: Called when agent completes a tool
            options: Additional options

        Returns:
            AsyncChainResponse that yields text chunks
        """
        return AsyncChainResponse(
            conversation=self,
            prompt=prompt,
            system=system,
            before_call=before_call,
            after_call=after_call,
        )

    def clear(self) -> None:
        """Clear conversation - creates new session on next prompt."""
        self._session_id = None

    async def close(self) -> None:
        """Close the connection and terminate the agent process."""
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await self._proc.wait()
            except ProcessLookupError:
                pass


class AsyncChainResponse:
    """Async iterator that yields text chunks from ACP agent."""

    def __init__(
        self,
        conversation: AsyncConversation,
        prompt: str,
        system: str | None,
        before_call: Callable | None,
        after_call: Callable | None,
    ):
        self._conversation = conversation
        self._prompt = prompt
        self._system = system
        self._before_call = before_call
        self._after_call = after_call
        self._responses: list[AsyncResponse] = []
        self._current_response: AsyncResponse | None = None
        self._iterated = False

    async def _ensure_connection(self) -> tuple[Any, str, ACPClientHandler]:
        """Ensure we have a connection and session."""
        conv = self._conversation

        log.warning("========== _ensure_connection CALLED ==========")
        log.warning(f"Current session_id: {conv._session_id}")
        log.warning(f"Has _session_loaded attr: {hasattr(conv, '_session_loaded')}")

        if conv._conn is None:
            # Parse agent command
            parts = conv.model.agent_command.split()
            program = parts[0]
            args = parts[1:] if len(parts) > 1 else []

            program_path = Path(program)
            spawn_program = program
            spawn_args = args

            # If it's a Python file, run with interpreter
            if program_path.exists() and not os.access(program_path, os.X_OK):
                spawn_program = sys.executable
                spawn_args = [str(program_path), *args]

            # Spawn agent process
            # Redirect stderr to devnull to avoid TUI interference
            conv._proc = await asyncio.create_subprocess_exec(
                spawn_program,
                *spawn_args,
                stdin=aio_subprocess.PIPE,
                stdout=aio_subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            if conv._proc.stdin is None or conv._proc.stdout is None:
                raise RuntimeError("Agent process does not expose stdio pipes")

            # Create client handler (stored on conversation, reused across turns)
            conv._client = ACPClientHandler()

            # Connect to agent
            conv._conn = connect_to_agent(conv._client, conv._proc.stdin, conv._proc.stdout)

            # Initialize and capture agent info
            init_response = await conv._conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(
                    name="textual-chat", title="Textual Chat", version="0.1.0"
                ),
            )

            # Extract agent name if available (prefer title for display)
            if hasattr(init_response, 'agent_info'):
                if hasattr(init_response.agent_info, 'title') and init_response.agent_info.title:
                    conv.agent_name = init_response.agent_info.title
                    log.warning(f"Agent title: {conv.agent_name}")
                elif hasattr(init_response.agent_info, 'name') and init_response.agent_info.name:
                    conv.agent_name = init_response.agent_info.name
                    log.warning(f"Agent name: {conv.agent_name}")
            else:
                log.warning("No agent_info found in init response")

        if conv._session_id is None:
            # Create new session
            log.warning("========== CREATING NEW SESSION ==========")
            session = await conv._conn.new_session(cwd=os.getcwd(), mcp_servers=[])
            conv._session_id = session.session_id
            log.warning(f"Created new session ID: {conv._session_id}")
        elif conv._session_id and not hasattr(conv, "_session_loaded"):
            # We have a session ID (from storage) but haven't loaded it yet
            # Try fork first (undocumented but implemented), then load, then new
            log.warning("========== ATTEMPTING TO RESTORE EXISTING SESSION ==========")
            log.warning(f"Trying to restore session ID: {conv._session_id}")
            log.warning(f"CWD: {os.getcwd()}")

            # Try 1: session/fork (undocumented but exists in agent code)
            log.warning("🔄 Attempt 1: Trying session/fork via send_request...")
            try:
                result = await conv._conn._conn.send_request(
                    "session/fork",
                    {
                        "sessionId": conv._session_id,
                        "cwd": os.getcwd(),
                        "mcpServers": [],
                    },
                )
                conv._session_loaded = True
                log.warning(f"✅ SUCCESS with session/fork!")
                log.warning(f"Fork response: {result}")
                if hasattr(result, 'session_id'):
                    conv._session_id = result.session_id
                    log.warning(f"New forked session ID: {conv._session_id}")
            except Exception as e:
                log.warning(f"❌ session/fork failed: {type(e).__name__}: {e}")

                # Try 2: session/load
                log.warning("🔄 Attempt 2: Trying session/load...")
                try:
                    session = await conv._conn.load_session(
                        cwd=os.getcwd(), mcp_servers=[], session_id=conv._session_id
                    )
                    conv._session_loaded = True
                    log.warning(f"✅ SUCCESS with session/load!")
                    log.warning(f"Load response: {session}")
                except Exception as e2:
                    log.warning(f"❌ session/load failed: {type(e2).__name__}: {e2}")

                    # Try 3: Create new session
                    log.warning("🔄 Attempt 3: Creating new session...")
                    session = await conv._conn.new_session(cwd=os.getcwd(), mcp_servers=[])
                    conv._session_id = session.session_id
                    conv._session_loaded = True
                    log.warning(f"Created new session ID: {conv._session_id}")

        # Reset client for this turn
        conv._client.reset_for_turn(
            before_call=self._before_call,
            after_call=self._after_call,
        )

        return conv._conn, conv._session_id, conv._client

    async def __aiter__(self):
        """Iterate over text chunks from the agent."""
        self._iterated = True

        log.debug("ACP __aiter__: starting")
        conn, session_id, client = await self._ensure_connection()
        log.debug(f"ACP __aiter__: connection ready, session_id={session_id}")

        # Create response tracker
        self._current_response = AsyncResponse()
        self._responses.append(self._current_response)

        full_text = ""
        chunks_collected: list[str] = []
        prompt_error: Exception | None = None

        async def run_prompt():
            """Run the prompt and signal completion."""
            nonlocal prompt_error
            log.debug("ACP run_prompt: starting")
            try:
                await conn.prompt(session_id=session_id, prompt=[text_block(self._prompt)])
                log.debug("ACP run_prompt: prompt completed")
            except Exception as e:
                log.debug(f"ACP run_prompt: error {e}")
                prompt_error = e
            finally:
                # Signal end of chunks
                await client._text_chunks.put(None)
                log.debug("ACP run_prompt: signaled done")

        async def collect_chunks():
            """Collect chunks from the queue."""
            log.debug("ACP collect_chunks: starting")
            while True:
                chunk = await client._text_chunks.get()
                log.debug(f"ACP collect_chunks: got chunk {repr(chunk)[:50]}")
                if chunk is None:
                    break
                chunks_collected.append(chunk)
            log.debug(f"ACP collect_chunks: done, {len(chunks_collected)} chunks")

        # Run both tasks concurrently
        log.debug("ACP __aiter__: starting gather")
        await asyncio.gather(run_prompt(), collect_chunks())
        log.debug("ACP __aiter__: gather completed")

        # Re-raise prompt errors
        if prompt_error:
            raise prompt_error

        # Yield all collected chunks
        for chunk in chunks_collected:
            full_text += chunk
            yield chunk

        self._current_response._text = full_text
        log.debug(f"ACP __aiter__: done, text={repr(full_text)[:50]}")

    async def responses(self):
        """Iterate over response objects (for usage info)."""
        if not self._iterated:
            async for _ in self:
                pass
        for resp in self._responses:
            yield resp


class AsyncResponse:
    """Response object with text and usage information."""

    def __init__(self):
        self._text: str = ""
        self._usage: Usage | None = None

    async def text(self) -> str:
        """Get the response text."""
        return self._text

    async def usage(self) -> Usage | None:
        """Get token usage information (not available from ACP)."""
        return self._usage

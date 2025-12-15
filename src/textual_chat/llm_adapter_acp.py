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
import logging
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from acp import Client, text_block
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AvailableCommandsUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)

from .agent_manager import get_agent_manager
from .session_storage import get_session_storage

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
    details: CacheDetails = field(default_factory=lambda: CacheDetails())


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

    def __init__(self, agent_command: str) -> None:
        self.model_id = agent_command
        self.agent_command = agent_command
        self.is_claude = False  # ACP agents handle their own caching

    def conversation(self, cwd: str | None = None) -> AsyncConversation:
        """Create a new conversation (ACP session).

        Args:
            cwd: Working directory for the agent session. Defaults to current directory.
        """
        return AsyncConversation(self, cwd=cwd)

    # Alias for ACP terminology
    session = conversation


class ACPClientHandler(Client):
    """Handles ACP client callbacks for session updates."""

    def __init__(self) -> None:
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
        log.debug(f"📨 session_update received: {type(update).__name__}")

        # Log UserMessageChunk to debug concatenation issue
        if isinstance(update, UserMessageChunk):
            log.warning(f"⚠️  RECEIVED UserMessageChunk (should NOT queue this): {update.content}")

        if isinstance(update, AgentMessageChunk):
            # Text streaming
            content = update.content
            if isinstance(content, TextContentBlock):
                await self._text_chunks.put(content.text)

        elif isinstance(update, AgentThoughtChunk):
            # Thinking/reasoning text streaming
            content = update.content
            if isinstance(content, TextContentBlock):
                # Stream thinking chunks just like regular text
                # The UI can decide how to display them (e.g., differently styled)
                await self._text_chunks.put(content.text)

        elif isinstance(update, (ToolCallStart, ToolCallProgress)):
            # Tool call update
            tool_call_id = update.tool_call_id or ""
            status = update.status or ""

            # Create or get tool call
            if tool_call_id not in self._tool_calls:
                # Extract arguments from raw_input
                arguments: dict[str, Any] = {}
                if isinstance(update.raw_input, dict):
                    arguments = update.raw_input
                elif isinstance(update.raw_input, str):
                    # Try to parse JSON string
                    import json

                    try:
                        parsed = json.loads(update.raw_input)
                        if isinstance(parsed, dict):
                            arguments = parsed
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, just note it as raw string
                        arguments = {"input": update.raw_input}
                elif update.raw_input is not None:
                    # Some other type - just note it
                    arguments = {"value": str(update.raw_input)}

                log.debug(
                    f"Tool {update.title}: raw_input type={type(update.raw_input)}, parsed args={arguments}"
                )

                tc = ToolCall(
                    id=tool_call_id,
                    name=update.title or "",
                    arguments=arguments,
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

    async def request_permission(
        self, options: Any, session_id: str, tool_call: Any, **kwargs: Any
    ) -> RequestPermissionResponse:
        """Handle permission requests - auto-approve for now."""
        # Auto-approve by selecting the first option if available
        option_id = options[0].id if options and len(options) > 0 else "approved"
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=option_id, outcome="selected")
        )

    # File system methods - raise not found for now
    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("fs/read_text_file")

    # Terminal methods - raise not found for now
    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[Any] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        from acp import RequestError

        raise RequestError.method_not_found("terminal/kill")

    def on_connect(self, conn: Any) -> None:
        """Called when the connection is established."""
        pass

    async def ext_method(self, method: str, params: Any) -> Any:
        """Handle extension methods."""
        from acp import RequestError

        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: Any) -> None:
        """Handle extension notifications."""
        return None

    def signal_done(self) -> None:
        """Signal that streaming is complete."""
        self._text_chunks.put_nowait(None)


class AsyncConversation:
    """Manages conversation with an ACP agent."""

    def __init__(self, model: AsyncModel, cwd: str | None = None) -> None:
        self.model = model
        self._conn: Any = None  # Shared connection (managed by AgentManager)
        self._client: ACPClientHandler | None = None  # Shared client handler
        self.agent_name: str | None = None  # Agent name from initialization
        self._cwd: str = cwd or os.getcwd()  # Working directory for agent sessions
        self.init_response: Any = None  # Store initialization response with capabilities
        self._session_loaded: bool = False  # Whether session has been loaded/forked

        # Check if we have an existing session for this working directory + agent
        storage = get_session_storage()
        self._session_id: str | None = storage.get_session_id(self._cwd, model.agent_command)

        if self._session_id:
            log.warning(
                f"🔍 AsyncConversation.__init__: Found existing session {self._session_id} for {self._cwd} + {model.agent_command}"
            )
        else:
            log.warning(
                f"🔍 AsyncConversation.__init__: No existing session for {self._cwd} + {model.agent_command}"
            )

        # If we loaded a session ID, don't mark it as loaded yet
        # This will trigger the fork/restore logic in _ensure_connection

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

    async def ensure_connected(self) -> None:
        """Ensure connection is established (for getting init info before first prompt)."""
        if self._conn is None:
            # Get shared connection from agent manager
            agent_manager = get_agent_manager()
            shared_conn = agent_manager.get_connection(self.model.agent_command)

            # Ensure the shared connection is initialized
            self._conn = await shared_conn.ensure_connected()

            # Use the shared client handler
            self._client = shared_conn._client

            # Copy agent info from shared connection
            self.init_response = shared_conn.init_response
            self.agent_name = shared_conn.agent_name

    def clear(self) -> None:
        """Clear conversation - creates new session on next prompt."""
        self._session_id = None

    async def close(self) -> None:
        """Close this conversation (but keep shared agent connection alive)."""
        # Don't terminate the shared agent process - it's managed by AgentManager
        # Just clear our reference to the session
        log.info(f"Closing conversation for session: {self._session_id}")
        # Session remains alive in the agent for potential forking
        self._session_id = None


class AsyncChainResponse:
    """Async iterator that yields text chunks from ACP agent."""

    def __init__(
        self,
        conversation: AsyncConversation,
        prompt: str,
        system: str | None,
        before_call: Callable | None,
        after_call: Callable | None,
    ) -> None:
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
            # Get shared connection from agent manager
            agent_manager = get_agent_manager()
            shared_conn = agent_manager.get_connection(conv.model.agent_command)

            # Ensure the shared connection is initialized
            conv._conn = await shared_conn.ensure_connected()

            # Use the shared client handler
            conv._client = shared_conn._client

            # Copy agent info from shared connection
            conv.init_response = shared_conn.init_response
            conv.agent_name = shared_conn.agent_name

            log.warning(f"Using shared connection for: {conv.model.agent_command}")
            log.warning(f"Agent name: {conv.agent_name}")

        if conv._session_id is None:
            # Create new session
            log.warning("========== CREATING NEW SESSION ==========")
            session = await conv._conn.new_session(cwd=conv._cwd, mcp_servers=[])
            conv._session_id = session.session_id
            conv._session_loaded = True  # Mark as loaded so we don't try to restore it
            log.warning(f"Created new session ID: {conv._session_id}")
            log.warning(f"Session CWD: {conv._cwd}")

            # Store in session storage for reuse
            storage = get_session_storage()
            if conv._session_id:
                storage.store_session_id(conv._cwd, conv.model.agent_command, conv._session_id)
        elif conv._session_id and not conv._session_loaded:
            # We have a session ID (from storage) but haven't loaded it yet
            # Try fork first (undocumented but implemented), then load, then new
            log.warning("========== ATTEMPTING TO RESTORE EXISTING SESSION ==========")
            log.warning(f"Trying to restore session ID: {conv._session_id}")
            log.warning(f"CWD: {conv._cwd}")

            # Try 1: session/fork (undocumented but exists in agent code)
            log.warning("🔄 Attempt 1: Trying session/fork via send_request...")
            try:
                result = await conv._conn._conn.send_request(
                    "session/fork",
                    {
                        "sessionId": conv._session_id,
                        "cwd": conv._cwd,
                        "mcpServers": [],
                    },
                )
                conv._session_loaded = True
                log.info("✅ Session forked successfully")

                # Extract new session ID from fork response (could be dict or object)
                new_session_id = None
                if isinstance(result, dict):
                    new_session_id = result.get("sessionId") or result.get("session_id")
                elif hasattr(result, "session_id"):
                    new_session_id = result.session_id
                elif hasattr(result, "sessionId"):
                    new_session_id = result.sessionId

                if new_session_id and new_session_id != conv._session_id:
                    log.info(f"   Forked session ID: {conv._session_id} → {new_session_id}")
                    conv._session_id = new_session_id

                # Update session storage with (possibly new) forked session ID
                storage = get_session_storage()
                if conv._session_id:
                    storage.store_session_id(conv._cwd, conv.model.agent_command, conv._session_id)
            except Exception as e:
                log.warning(f"❌ session/fork failed: {type(e).__name__}: {e}")

                # Try 2: session/load
                log.warning("🔄 Attempt 2: Trying session/load...")
                try:
                    session = await conv._conn.load_session(
                        cwd=conv._cwd, mcp_servers=[], session_id=conv._session_id
                    )
                    conv._session_loaded = True
                    log.warning("✅ SUCCESS with session/load!")
                    log.warning(f"Load response: {session}")
                    # Session ID stays the same after load
                except Exception as e2:
                    log.warning(f"❌ session/load failed: {type(e2).__name__}: {e2}")

                    # Try 3: Create new session (fallback after restore failed)
                    log.warning("🔄 Attempt 3: Creating new session...")
                    session = await conv._conn.new_session(cwd=conv._cwd, mcp_servers=[])
                    conv._session_id = session.session_id
                    # Don't set _session_loaded = True here, since we failed to restore
                    # and created a new session instead
                    log.warning(f"Created new session ID: {conv._session_id}")
                    log.warning(f"Session CWD: {conv._cwd}")

                    # Update session storage with new session ID
                    storage = get_session_storage()
                    if conv._session_id:
                        storage.store_session_id(
                            conv._cwd, conv.model.agent_command, conv._session_id
                        )

        # Reset client for this turn
        if conv._client:
            conv._client.reset_for_turn(
                before_call=self._before_call,
                after_call=self._after_call,
            )

        # Ensure types are correct for return
        assert conv._session_id is not None
        assert conv._client is not None
        return conv._conn, conv._session_id, conv._client

    async def __aiter__(self) -> AsyncIterator[str]:
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

        async def run_prompt() -> None:
            """Run the prompt and signal completion."""
            nonlocal prompt_error
            log.debug("ACP run_prompt: starting")
            log.warning(f"🚀 SENDING PROMPT TO AGENT: {repr(self._prompt)}")
            log.warning(f"   Session ID: {session_id}")
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

        async def collect_chunks() -> None:
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

    async def responses(self) -> AsyncIterator[AsyncResponse]:
        """Iterate over response objects (for usage info)."""
        if not self._iterated:
            async for _ in self:
                pass
        for resp in self._responses:
            yield resp


class AsyncResponse:
    """Response object with text and usage information."""

    def __init__(self) -> None:
        self._text: str = ""
        self._usage: Usage | None = None

    async def text(self) -> str:
        """Get the response text."""
        return self._text

    async def usage(self) -> Usage | None:
        """Get token usage information (not available from ACP)."""
        return self._usage

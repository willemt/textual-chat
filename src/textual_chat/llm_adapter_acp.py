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
import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import singledispatch
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
    ToolCallProgress as ACPToolCallProgress,
    ToolCallStart as ACPToolCallStart,
    UserMessageChunk,
)

from .agent_manager import get_agent_manager
from .events import MessageChunk, ThoughtChunk, ToolCallComplete, ToolCallStart, ToolCallProgress
from .session_storage import get_session_storage

log = logging.getLogger(__name__)
# log.propagate = False  # TEMPORARILY ENABLED FOR DEBUGGING


# Singledispatch handlers for session updates
@singledispatch
async def _handle_update(update: Any, client: Any) -> None:
    """Default handler for unknown update types."""
    log.warning(f"⚠️  Unhandled update type: {type(update).__name__} - {update}")


@_handle_update.register
async def _handle_user_message(update: UserMessageChunk, client: Any) -> None:
    """Handle user message chunks (debugging only)."""
    log.warning(f"⚠️  RECEIVED UserMessageChunk (should NOT queue this): {update.content}")


@_handle_update.register
async def _handle_agent_message(update: AgentMessageChunk, client: Any) -> None:
    """Handle agent message chunks - stream text."""
    content = update.content
    if isinstance(content, TextContentBlock):
        await client._events.put(MessageChunk(content.text))


@_handle_update.register
async def _handle_agent_thought(update: AgentThoughtChunk, client: Any) -> None:
    """Handle agent thinking chunks - stream thinking text."""

    content = update.content
    if isinstance(content, TextContentBlock):
        await client._events.put(ThoughtChunk(content.text))


@_handle_update.register(ACPToolCallStart)
@_handle_update.register(ACPToolCallProgress)
async def _handle_tool_call(update: ACPToolCallStart | ACPToolCallProgress, client: Any) -> None:
    """Handle tool call start and progress updates."""

    tool_call_id = update.tool_call_id or ""
    status = update.status or ""

    # Create or get tool call
    if tool_call_id not in client._tool_calls:
        # Extract arguments from raw_input
        arguments: dict[str, Any] = {}
        if isinstance(update.raw_input, dict):
            arguments = update.raw_input
        elif isinstance(update.raw_input, str):
            # Try to parse JSON string
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

        log.info(
            f"Tool {update.title}: raw_input type={type(update.raw_input)}, parsed args={arguments} {update}"
        )

        tool_name = update.title or ""

        # Track the tool call
        tc = ToolCall(
            id=tool_call_id,
            name=tool_name,
            arguments=arguments,
        )
        client._tool_calls[tool_call_id] = tc

        # Emit ToolCallStart event
        await client._events.put(
            ToolCallStart(id=tool_call_id, name=tool_name, arguments=arguments)
        )

    # Emit progress/completion events
    if status == "in_progress":
        await client._events.put(ToolCallProgress(id=tool_call_id, status=status))
    elif status in ("completed", "failed"):
        output = ""
        if update.raw_output:
            output = str(update.raw_output)
        await client._events.put(ToolCallComplete(id=tool_call_id, output=output))


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
        from .events import StreamEvent

        self._events: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._tool_calls: dict[str, ToolCall] = {}

    def reset_for_turn(self) -> None:
        """Reset state for a new turn."""
        from .events import StreamEvent

        self._events = asyncio.Queue()
        self._tool_calls = {}

    async def session_update(
        self,
        session_id: str,
        update: (
            UserMessageChunk
            | AgentMessageChunk
            | AgentThoughtChunk
            | ACPToolCallStart
            | ACPToolCallProgress
            | AgentPlanUpdate
            | AvailableCommandsUpdate
            | CurrentModeUpdate
        ),
        **kwargs: Any,
    ) -> None:
        """Handle session updates from the agent using singledispatch."""
        log.debug(f"📨 session_update received: {type(update).__name__} {update} {kwargs}")
        await _handle_update(update, self)

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
        self._events.put_nowait(None)


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
        self._session_capabilities: dict[str, bool] = {}  # Available session methods

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
        options: dict | None = None,
    ) -> AsyncChainResponse:
        """Execute prompt with the ACP agent.

        Args:
            prompt: User message
            system: System prompt (may not be supported by all agents)
            tools: Not used for ACP (agent has its own tools)
            options: Additional options

        Returns:
            AsyncChainResponse that yields events (MessageChunk, ToolCallStart, etc.)
        """
        return AsyncChainResponse(
            conversation=self,
            prompt=prompt,
            system=system,
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


async def create_connection(conv: AsyncConversation) -> None:
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


class AsyncChainResponse:
    """Async iterator that yields text chunks from ACP agent."""

    def __init__(
        self,
        conversation: AsyncConversation,
        prompt: str,
        system: str | None,
    ) -> None:
        self._conversation = conversation
        self._prompt = prompt
        self._system = system
        self._responses: list[AsyncResponse] = []
        self._current_response: AsyncResponse | None = None
        self._iterated = False

    async def _create_new_session(self, conv: AsyncConversation) -> None:
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

    async def _load_session(self, conv: AsyncConversation) -> None:
        log.warning("🔄 Attempt: Trying session/load...")
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
            raise

    async def _new_session(self, conv: AsyncConversation) -> None:
        log.warning("🔄 Attempt: Creating new session...")
        session = await conv._conn.new_session(cwd=conv._cwd, mcp_servers=[])
        conv._session_id = session.session_id
        # Don't set _session_loaded = True here, since we failed to restore
        # and created a new session instead
        log.warning(f"Created new session ID: {conv._session_id}")
        log.warning(f"Session CWD: {conv._cwd}")

        # Update session storage with new session ID
        storage = get_session_storage()
        if conv._session_id:
            storage.store_session_id(conv._cwd, conv.model.agent_command, conv._session_id)

    async def _fork_session(self, conv: AsyncConversation) -> None:
        log.warning("🔄: Trying session/fork via send_request...")
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
            raise

    async def _ensure_connection(self) -> tuple[Any, str, ACPClientHandler]:
        """Ensure we have a connection and session."""
        conv = self._conversation

        log.warning("========== _ensure_connection CALLED ==========")
        log.warning(f"Current session_id: {conv._session_id}")
        log.warning(f"Has _session_loaded attr: {hasattr(conv, '_session_loaded')}")

        if conv._conn is None:
            await create_connection(conv)

        # We have a session ID (from storage) but haven't loaded it yet
        # Try fork first (undocumented but implemented), then load, then new
        log.warning("========== ATTEMPTING TO RESTORE EXISTING SESSION ==========")
        log.warning(f"Trying to restore session ID: {conv._session_id}")
        log.warning(f"CWD: {conv._cwd}")

        if conv._session_id is None:
            await self._create_new_session(conv)
        elif conv._session_id and not conv._session_loaded:
            try:
                await self._load_session(conv)
            except:
                try:
                    await self._fork_session(conv)
                except:
                    await self._new_session(conv)

        # Reset client for this turn
        if conv._client:
            conv._client.reset_for_turn()

        # Ensure types are correct for return
        assert conv._session_id is not None
        assert conv._client is not None
        return conv._conn, conv._session_id, conv._client

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        """Iterate over events from the agent."""
        from .events import MessageChunk, StreamEvent

        self._iterated = True

        log.debug("ACP __aiter__: starting")
        conn, session_id, client = await self._ensure_connection()
        log.debug(f"ACP __aiter__: connection ready, session_id={session_id}")

        # Reset client for new turn
        client.reset_for_turn()

        # Create response tracker
        self._current_response = AsyncResponse()
        self._responses.append(self._current_response)

        full_text = ""
        prompt_error: Exception | None = None
        prompt_task: asyncio.Task | None = None

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
                # Signal end of events
                await client._events.put(None)
                log.debug("ACP run_prompt: signaled done")

        # Start prompt task in background
        log.debug("ACP __aiter__: starting prompt task")
        prompt_task = asyncio.create_task(run_prompt())

        # Yield events as they arrive in real-time
        log.debug("ACP __aiter__: starting event loop")
        while True:
            event = await client._events.get()
            log.debug(f"ACP __aiter__: got event {type(event).__name__ if event else None}")
            if event is None:
                break
            if isinstance(event, MessageChunk):
                full_text += event.text
            yield event

        log.debug("ACP __aiter__: event loop done")

        # Wait for prompt task to complete and check for errors
        await prompt_task
        if prompt_error:
            raise prompt_error

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

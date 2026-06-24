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
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import singledispatch
from pathlib import Path
from typing import TypedDict, Union, cast

# JSON type for proper typing of JSON values
JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]

from acp import Client, text_block
from acp.client.connection import ClientSideConnection
from acp.interfaces import Agent
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AvailableCommandsUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    EnvVariable,
    InitializeResponse,
    KillTerminalCommandResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalExitStatus,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallUpdate,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from acp.schema import (
    ToolCallProgress as ACPToolCallProgress,
)
from acp.schema import (
    ToolCallStart as ACPToolCallStart,
)
from acp.schema import (
    UsageUpdate as ACPSchemaUsageUpdate,
)
from acp.schema import (
    ConfigOptionUpdate,
    SessionInfoUpdate,
)

from .agent_manager import get_agent_manager
from .events import (
    Cost,
    MessageChunk,
    PermissionRequest,
    PermissionTimeout,
    PlanChunk,
    StreamEvent,
    ThoughtChunk,
    TokenUsage,
    ToolCallComplete,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessage,
)
from .session_storage import get_session_storage

log = logging.getLogger(__name__)
# log.propagate = False  # TEMPORARILY ENABLED FOR DEBUGGING


def _is_acp_client_capability(update: ACPToolCallStart | ACPToolCallProgress) -> bool:
    """Check if a tool call is an ACP client capability (Read/Write/Edit File, Terminal).

    These calls will emit their own events from the capability methods with proper arguments,
    so we should mute the events that come from the agent's ACPToolCallStart updates.
    """
    if hasattr(update, "field_meta") and update.field_meta:
        claude_meta = update.field_meta.get("claudeCode", {})
        tool_name_meta = claude_meta.get("toolName", "")
        return tool_name_meta in (
            "mcp__acp__Read",
            "mcp__acp__Write",
            "mcp__acp__Edit",
            "mcp__acp__Bash",
        )
    return False


# Singledispatch handlers for session updates
@singledispatch
async def _handle_update(update: object, client: object, session_id: str) -> None:
    """Default handler for unknown update types.

    Also handles dynamically created types like UsageUpdate from newer ACP schemas.
    """
    # Check if this is a usage_update (handles dynamically created UsageUpdate class)
    session_update_type = getattr(update, "session_update", None)
    if session_update_type == "usage_update":
        # Extract usage data from the object
        used = getattr(update, "used", 0)
        size = getattr(update, "size", 0)
        cost_obj = getattr(update, "cost", None)

        handler = cast("ACPClientHandler", client)
        queue = handler.get_session_queue(session_id)

        log.info(
            f"📊 UsageUpdate: {used}/{size} tokens "
            f"({used / size * 100:.1f}% used)"
            + (f", cost: {cost_obj.amount} {cost_obj.currency}" if cost_obj else "")
        )

        # Convert to event
        cost = None
        if cost_obj and hasattr(cost_obj, "amount") and hasattr(cost_obj, "currency"):
            cost = Cost(amount=cost_obj.amount, currency=cost_obj.currency)

        await queue.put(UsageUpdate(used=used, size=size, cost=cost))
        return

    log.warning(f"⚠️  Unhandled update type: {type(update).__name__} - {update}")


@_handle_update.register
async def _handle_user_message(update: UserMessageChunk, client: object, session_id: str) -> None:
    """Handle user message chunks.

    During a live turn these are echoes of what the user just typed (already
    shown locally), so they're dropped. While replaying a loaded session,
    however, they're the only record of the human side of the transcript, so
    surface them as UserMessage events.
    """
    handler = cast("ACPClientHandler", client)
    if not handler.is_replaying(session_id):
        log.debug(f"Dropping live UserMessageChunk: {update.content}")
        return

    content = update.content
    if isinstance(content, TextContentBlock):
        queue = handler.get_session_queue(session_id)
        await queue.put(UserMessage(content.text))


@_handle_update.register
async def _handle_agent_message(update: AgentMessageChunk, client: object, session_id: str) -> None:
    """Handle agent message chunks - stream text."""
    handler = cast("ACPClientHandler", client)
    queue = handler.get_session_queue(session_id)
    content = update.content
    if isinstance(content, TextContentBlock):
        await queue.put(MessageChunk(content.text))


@_handle_update.register
async def _handle_agent_thought(update: AgentThoughtChunk, client: object, session_id: str) -> None:
    """Handle agent thinking chunks - stream thinking text."""
    handler = cast("ACPClientHandler", client)
    queue = handler.get_session_queue(session_id)
    content = update.content
    if isinstance(content, TextContentBlock):
        await queue.put(ThoughtChunk(content.text))


@_handle_update.register
async def _handle_agent_plan(update: AgentPlanUpdate, client: object, session_id: str) -> None:
    """Handle agent planning updates - send entries to UI."""
    handler = cast("ACPClientHandler", client)
    queue = handler.get_session_queue(session_id)

    log.info(f"📋 AgentPlanUpdate received with {len(update.entries)} entries")

    # Log the raw update for debugging
    log.info(f"📋 Raw AgentPlanUpdate: {update}")
    for i, entry in enumerate(update.entries):
        log.info(
            f"📋   Entry {i}: content='{entry.content}', status='{entry.status}', priority={entry.priority}"
        )

    # Convert entries to dicts for the event
    entries: list[dict[str, JSON]] = [
        {
            "content": entry.content,
            "status": entry.status,
            "priority": entry.priority,
        }
        for entry in update.entries
    ]

    log.info(f"📋 Emitting PlanChunk with {len(entries)} entries: {entries}")
    await queue.put(PlanChunk(entries=entries))


@_handle_update.register(ACPToolCallStart)
@_handle_update.register(ACPToolCallProgress)
async def _handle_tool_call(
    update: ACPToolCallStart | ACPToolCallProgress, client: object, session_id: str
) -> None:
    """Handle tool call start and progress updates."""
    handler = cast("ACPClientHandler", client)
    queue = handler.get_session_queue(session_id)

    tool_call_id = update.tool_call_id or ""
    status = update.status or ""

    # Create or get tool call
    if tool_call_id not in handler._tool_calls:
        # Extract arguments from raw_input
        arguments: dict[str, JSON] = {}
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
        handler._tool_calls[tool_call_id] = tc

        # Check if this is an MCP ACP client capability call (Read/Write)
        # These will emit their own events with proper arguments from the capability methods
        if _is_acp_client_capability(update):
            handler._acp_client_capabilities.add(tool_call_id)
            log.info(f"Muting ACP client capability tool: {tool_name}")
        else:
            # Emit ToolCallStart event (unless it's an ACP client capability)
            await queue.put(ToolCallStart(id=tool_call_id, name=tool_name, arguments=arguments))

    # Emit progress/completion events (skip for ACP client capabilities)
    if tool_call_id not in handler._acp_client_capabilities:
        if status == "in_progress":
            await queue.put(ToolCallProgress(id=tool_call_id, status=status))
        elif status in ("completed", "failed"):
            output = ""
            if update.raw_output:
                output = str(update.raw_output)
            await queue.put(ToolCallComplete(id=tool_call_id, output=output))


@_handle_update.register(ACPSchemaUsageUpdate)
async def _handle_schema_usage_update(
    update: ACPSchemaUsageUpdate, client: object, session_id: str
) -> None:
    """Handle usage update from ACP schema - context window and cost."""
    handler = cast("ACPClientHandler", client)
    queue = handler.get_session_queue(session_id)

    log.info(
        f"📊 UsageUpdate: {update.used}/{update.size} tokens "
        f"({update.used / update.size * 100:.1f}% used)"
        + (f", cost: {update.cost.amount} {update.cost.currency}" if update.cost else "")
    )

    # Convert to event
    cost = None
    if update.cost:
        cost = Cost(amount=update.cost.amount, currency=update.cost.currency)

    await queue.put(UsageUpdate(used=update.used, size=update.size, cost=cost))


class CacheDetails(TypedDict, total=False):
    """Cache-related token details (not used in ACP but kept for interface compat)."""

    cached_tokens: int


# ============================================================================
# Local ACP types for session/update usage_update notification
# These are defined locally since the ACP schema may not have them yet.
# See: https://agentclientprotocol.com/rfds/0002-session-usage
# ============================================================================


def _json_to_int(value: JSON, default: int = 0) -> int:
    """Safely convert a JSON value to int."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _json_to_float(value: JSON, default: float = 0.0) -> float:
    """Safely convert a JSON value to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _json_to_str(value: JSON, default: str = "") -> str:
    """Safely convert a JSON value to str."""
    if value is None:
        return default
    return str(value)


@dataclass
class ACPCost:
    """Cost information for cumulative session cost."""

    amount: float
    currency: str  # ISO 4217 currency code

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> "ACPCost":
        """Parse from a dictionary."""
        return cls(
            amount=_json_to_float(data.get("amount"), 0.0),
            currency=_json_to_str(data.get("currency"), "USD"),
        )


@dataclass
class ACPUsageUpdate:
    """Usage update notification sent via session/update.

    Reports context window utilization and optional cumulative session cost.
    """

    session_update: str  # Should be "usage_update"
    used: int  # Tokens currently in context
    size: int  # Total context window size in tokens
    cost: ACPCost | None = None

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> "ACPUsageUpdate":
        """Parse from a dictionary (handles both camelCase and snake_case)."""
        session_update = _json_to_str(data.get("sessionUpdate") or data.get("session_update"), "")
        used = _json_to_int(data.get("used"), 0)
        size = _json_to_int(data.get("size"), 0)
        cost_data = data.get("cost")
        cost = ACPCost.from_dict(cost_data) if isinstance(cost_data, dict) else None
        return cls(session_update=session_update, used=used, size=size, cost=cost)


@dataclass
class ACPPromptUsage:
    """Token usage information in PromptResponse."""

    total_tokens: int
    input_tokens: int
    output_tokens: int
    thought_tokens: int | None = None
    cached_read_tokens: int | None = None
    cached_write_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> "ACPPromptUsage":
        """Parse from a dictionary."""
        thought = data.get("thought_tokens")
        cached_read = data.get("cached_read_tokens")
        cached_write = data.get("cached_write_tokens")
        return cls(
            total_tokens=_json_to_int(data.get("total_tokens"), 0),
            input_tokens=_json_to_int(data.get("input_tokens"), 0),
            output_tokens=_json_to_int(data.get("output_tokens"), 0),
            thought_tokens=_json_to_int(thought) if thought is not None else None,
            cached_read_tokens=_json_to_int(cached_read) if cached_read is not None else None,
            cached_write_tokens=_json_to_int(cached_write) if cached_write is not None else None,
        )


# Register the handler for usage_update
@_handle_update.register(ACPUsageUpdate)
async def _handle_usage_update(update: ACPUsageUpdate, client: object, session_id: str) -> None:
    """Handle usage update notifications - context window and cost."""
    handler = cast("ACPClientHandler", client)
    queue = handler.get_session_queue(session_id)

    log.info(
        f"📊 UsageUpdate: {update.used}/{update.size} tokens "
        f"({update.used / update.size * 100:.1f}%)"
        + (f", cost: {update.cost.amount} {update.cost.currency}" if update.cost else "")
    )

    # Convert to event
    cost = None
    if update.cost:
        cost = Cost(amount=update.cost.amount, currency=update.cost.currency)

    await queue.put(UsageUpdate(used=update.used, size=update.size, cost=cost))


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
    arguments: dict[str, JSON]


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
        self._session_events: dict[str, asyncio.Queue[StreamEvent | None]] = (
            {}
        )  # session_id -> queue
        self._tool_calls: dict[str, ToolCall] = {}
        self._sessions: dict[str, str] = {}  # session_id -> cwd mapping
        self._terminals: dict[str, asyncio.subprocess.Process] = {}  # terminal_id -> process
        self._acp_client_capabilities: set[str] = set()  # tool_call_ids for Read/Write File
        self._permission_responses: dict[str, asyncio.Future[RequestPermissionResponse]] = (
            {}
        )  # request_id -> future
        self._replaying_sessions: set[str] = set()  # session_ids currently replaying history

    def begin_replay(self, session_id: str) -> None:
        """Mark a session as replaying prior history (during session/load).

        While replaying, user message chunks are surfaced as events so the UI
        can reconstruct the transcript; during normal turns they are dropped.
        """
        self._replaying_sessions.add(session_id)

    def end_replay(self, session_id: str) -> None:
        """Stop treating a session's updates as history replay."""
        self._replaying_sessions.discard(session_id)

    def is_replaying(self, session_id: str) -> bool:
        """Whether the given session is currently replaying history."""
        return session_id in self._replaying_sessions

    def register_session(self, session_id: str, cwd: str) -> None:
        """Register a session's working directory and create its event queue."""

        self._sessions[session_id] = cwd
        if session_id not in self._session_events:
            self._session_events[session_id] = asyncio.Queue()
        log.debug(f"Registered session {session_id} with cwd: {cwd}")

    def get_session_queue(self, session_id: str) -> asyncio.Queue[StreamEvent | None]:
        """Get the event queue for a session."""

        if session_id not in self._session_events:
            self._session_events[session_id] = asyncio.Queue()
        return self._session_events[session_id]

    def unregister_session(self, session_id: str) -> None:
        """Unregister a session and clean up its resources."""
        self._sessions.pop(session_id, None)
        self._session_events.pop(session_id, None)
        log.debug(f"Unregistered session {session_id}")

    def reset_for_turn(self) -> None:
        """Reset state for a new turn.

        Note: We don't clear session_events since they're per-session queues.
        """
        self._tool_calls = {}
        self._acp_client_capabilities.clear()
        self._permission_responses = {}

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
            | ConfigOptionUpdate
            | SessionInfoUpdate
            | ACPSchemaUsageUpdate
        ),
        **kwargs: JSON,
    ) -> None:
        """Handle session updates from the agent using singledispatch."""
        log.debug(f"📨 session_update received: {type(update).__name__} {update} {kwargs}")
        await _handle_update(update, self, session_id)

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: JSON,
    ) -> RequestPermissionResponse:
        """Handle permission requests - prompt user for approval.

        Creates a PermissionRequest event and waits for user response.
        """
        from acp.schema import AllowedOutcome

        # Validate options
        if not options:
            log.warning("⚠️ Permission request with no options, cannot proceed")
            raise ValueError("Permission request must have at least one option")

        # Generate unique request ID
        request_id = f"perm_{uuid.uuid4().hex[:8]}"

        # Create future for user response
        response_future: asyncio.Future[RequestPermissionResponse] = asyncio.Future()
        self._permission_responses[request_id] = response_future

        # Convert options and tool_call to dicts for the event
        options_dicts = [
            {
                "option_id": opt.option_id,
                "name": opt.name,
                "description": opt.description if hasattr(opt, "description") else None,
            }
            for opt in options
        ]

        tool_call_dict: dict[str, JSON] = {
            "tool_call_id": tool_call.tool_call_id if hasattr(tool_call, "tool_call_id") else None,
            "title": tool_call.title if hasattr(tool_call, "title") else None,
            "status": tool_call.status if hasattr(tool_call, "status") else None,
        }

        log.info(f"🔐 Requesting permission for: {tool_call.title} (request_id: {request_id})")

        # Queue permission request event for UI (route to session-specific queue)
        queue = self.get_session_queue(session_id)
        await queue.put(
            PermissionRequest(
                request_id=request_id,
                session_id=session_id,
                tool_call=tool_call_dict,
                options=options_dicts,
            )
        )

        # Wait for user response (with timeout)
        try:
            response = await asyncio.wait_for(response_future, timeout=300.0)  # 5 minute timeout
            log.info(f"✅ Received permission response for {request_id}")
            return response
        except asyncio.TimeoutError:
            log.warning(f"⏱️ Permission request {request_id} timed out, denying")
            # Clean up
            del self._permission_responses[request_id]
            # Notify UI to remove the permission prompt
            await queue.put(PermissionTimeout(request_id=request_id))
            # Return first option as denial (or we could raise an error)
            # For now, deny by returning the first option (could be "deny" or "allow")
            return RequestPermissionResponse(
                outcome=AllowedOutcome(option_id=options[0].option_id, outcome="selected")
            )

    # File system methods - implement with cwd support
    async def write_text_file(
        self, content: str = "", path: str = "", session_id: str = "", **kwargs: JSON
    ) -> WriteTextFileResponse:
        """Write text file relative to session's cwd."""
        from acp import RequestError
        from acp.schema import WriteTextFileResponse

        # Validate required parameters
        if not path:
            raise RequestError.invalid_params({"error": "Missing required parameter: path"})
        if not session_id:
            raise RequestError.invalid_params({"error": "Missing required parameter: session_id"})

        # Get session cwd
        cwd = self._sessions.get(session_id, os.getcwd())

        # Resolve path relative to cwd
        full_path = Path(cwd) / path
        log.info(f"📝 Writing file: {full_path}")

        # Emit ToolCallStart event for UI display
        tool_call_id = str(uuid.uuid4())
        arguments: dict[str, JSON] = {"path": path}
        queue = self.get_session_queue(session_id)
        await queue.put(ToolCallStart(id=tool_call_id, name="Write File", arguments=arguments))

        try:
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            full_path.write_text(content, encoding="utf-8")

            # Emit ToolCallComplete event
            await queue.put(ToolCallComplete(id=tool_call_id, output=""))

            return WriteTextFileResponse()
        except Exception as e:
            log.error(f"Failed to write file {full_path}: {e}")
            raise RequestError.internal_error({"error": f"Failed to write file: {e}"})

    async def read_text_file(
        self,
        path: str = "",
        session_id: str = "",
        limit: int | None = None,
        line: int | None = None,
        **kwargs: JSON,
    ) -> ReadTextFileResponse:
        """Read text file relative to session's cwd."""
        from acp import RequestError
        from acp.schema import ReadTextFileResponse

        # Validate required parameters
        if not path:
            raise RequestError.invalid_params({"error": "Missing required parameter: path"})
        if not session_id:
            raise RequestError.invalid_params({"error": "Missing required parameter: session_id"})

        # Get session cwd
        cwd = self._sessions.get(session_id, os.getcwd())

        # Resolve path relative to cwd
        full_path = Path(cwd) / path
        log.info(f"📖 Reading file: {full_path}")

        # Emit ToolCallStart event for UI display
        tool_call_id = str(uuid.uuid4())
        arguments: dict[str, JSON] = {"path": path}
        if limit is not None:
            arguments["limit"] = limit
        if line is not None:
            arguments["line"] = line
        queue = self.get_session_queue(session_id)
        await queue.put(ToolCallStart(id=tool_call_id, name="Read File", arguments=arguments))

        try:
            # Check if file exists and is actually a file
            if not full_path.exists():
                log.warning(f"File not found: {full_path}")
                raise RequestError.invalid_params({"error": f"File not found: {path}"})
            if not full_path.is_file():
                log.warning(f"Path is not a file: {full_path}")
                raise RequestError.invalid_params({"error": f"Path is not a file: {path}"})

            content = full_path.read_text(encoding="utf-8")

            # Handle line parameter (1-based line number to start from)
            if line is not None:
                lines = content.splitlines(keepends=True)
                if line > 0 and line <= len(lines):
                    content = "".join(lines[line - 1 :])

            # Handle limit parameter (max number of lines)
            if limit is not None and limit > 0:
                lines = content.splitlines(keepends=True)
                content = "".join(lines[:limit])

            # Emit ToolCallComplete event
            await queue.put(ToolCallComplete(id=tool_call_id, output=""))

            return ReadTextFileResponse(content=content)
        except RequestError:
            raise
        except Exception as e:
            log.error(f"Failed to read file {full_path}: {e}")
            raise RequestError.internal_error({"error": f"Failed to read file: {e}"})

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: JSON,
    ) -> CreateTerminalResponse:
        """Create a terminal and execute command with session's cwd."""
        from acp import RequestError
        from acp.schema import CreateTerminalResponse

        # Validate required parameters
        if not command:
            raise RequestError.invalid_params({"error": "Missing required parameter: command"})
        if not session_id:
            raise RequestError.invalid_params({"error": "Missing required parameter: session_id"})

        # Get session cwd or use provided cwd
        session_cwd = self._sessions.get(session_id, os.getcwd())
        working_dir = cwd or session_cwd

        # Build command line - combine command and args for shell execution
        if args:
            # Join command and args with spaces
            full_command = command + " " + " ".join(args)
        else:
            full_command = command

        log.info(f"🖥️  Creating terminal in {working_dir}: {full_command}")

        # Emit ToolCallStart event for UI display
        tool_call_id = str(uuid.uuid4())
        arguments: dict[str, JSON] = {"command": command}
        if args:
            arguments["args"] = cast(JSON, args)
        queue = self.get_session_queue(session_id)
        await queue.put(ToolCallStart(id=tool_call_id, name="Terminal", arguments=arguments))

        # Build environment dict if provided
        env_dict = None
        if env:
            env_dict = os.environ.copy()
            for env_var in env:
                env_dict[env_var.name] = env_var.value

        try:
            # Start process using shell (to support operators like &&, |, etc.)
            process = await asyncio.create_subprocess_shell(
                full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                cwd=working_dir,
                env=env_dict,
            )

            # Generate terminal ID
            terminal_id = f"term_{uuid.uuid4().hex[:8]}"

            # Store process
            self._terminals[terminal_id] = process

            # Emit ToolCallComplete event
            await queue.put(ToolCallComplete(id=tool_call_id, output=""))

            return CreateTerminalResponse(terminal_id=terminal_id)
        except Exception as e:
            from acp import RequestError

            log.error(f"Failed to create terminal: {e}")
            raise RequestError.internal_error({"error": f"Failed to create terminal: {e}"})

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: JSON
    ) -> TerminalOutputResponse:
        """Get output from terminal."""
        from acp import RequestError
        from acp.schema import TerminalOutputResponse

        process = self._terminals.get(terminal_id)
        if not process:
            raise RequestError.invalid_params({"error": f"Terminal {terminal_id} not found"})

        try:
            # Read available output (non-blocking)
            output = ""
            if process.stdout:
                try:
                    # Try to read with timeout
                    data = await asyncio.wait_for(process.stdout.read(8192), timeout=0.1)
                    output = data.decode("utf-8", errors="replace")
                except asyncio.TimeoutError:
                    pass

            # Check if process has exited
            exit_status = None
            if process.returncode is not None:
                exit_status = TerminalExitStatus(exit_code=process.returncode)

            return TerminalOutputResponse(output=output, exit_status=exit_status, truncated=False)
        except RequestError:
            raise
        except Exception as e:
            log.error(f"Failed to get terminal output: {e}")
            raise RequestError.internal_error({"error": f"Failed to get terminal output: {e}"})

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: JSON
    ) -> WaitForTerminalExitResponse:
        """Wait for terminal to exit."""
        from acp import RequestError

        process = self._terminals.get(terminal_id)
        if not process:
            raise RequestError.invalid_params({"error": f"Terminal {terminal_id} not found"})

        try:
            # Wait for process to finish
            await process.wait()
            return WaitForTerminalExitResponse(exit_code=process.returncode or 0)
        except Exception as e:
            log.error(f"Failed to wait for terminal: {e}")
            raise RequestError.internal_error({"error": f"Failed to wait for terminal: {e}"})

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: JSON
    ) -> ReleaseTerminalResponse | None:
        """Release terminal resources."""
        if terminal_id in self._terminals:
            process = self._terminals[terminal_id]

            # If still running, terminate it
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

            del self._terminals[terminal_id]
            log.info(f"Released terminal {terminal_id}")

        return None

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: JSON
    ) -> KillTerminalCommandResponse | None:
        """Kill a running terminal."""
        from acp import RequestError

        process = self._terminals.get(terminal_id)
        if not process:
            raise RequestError.invalid_params({"error": f"Terminal {terminal_id} not found"})

        try:
            if process.returncode is None:
                process.kill()
                await process.wait()
                log.info(f"Killed terminal {terminal_id}")

            return None
        except Exception as e:
            log.error(f"Failed to kill terminal: {e}")
            raise RequestError.internal_error({"error": f"Failed to kill terminal: {e}"})

    def on_connect(self, conn: Agent) -> None:
        """Called when the connection is established."""
        pass

    async def ext_method(self, method: str, params: dict[str, JSON]) -> dict[str, JSON]:
        """Handle extension methods."""
        from acp import RequestError

        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, JSON]) -> None:
        """Handle extension notifications."""
        return None

    # Note: signal_done is no longer used with session-specific queues
    # Completion is signaled by putting None into the session's queue

    def respond_to_permission(self, request_id: str, option_id: str) -> None:
        """Respond to a permission request.

        Args:
            request_id: The ID of the permission request
            option_id: The option_id to select from the permission options
        """
        if request_id not in self._permission_responses:
            log.warning(f"⚠️ Unknown permission request ID: {request_id}")
            return

        future = self._permission_responses[request_id]
        if not future.done():
            response = RequestPermissionResponse(
                outcome=AllowedOutcome(option_id=option_id, outcome="selected")
            )
            future.set_result(response)
            log.info(f"✅ Set permission response for {request_id}: {option_id}")
        else:
            log.warning(f"⚠️ Permission request {request_id} already responded to")


class AsyncConversation:
    """Manages conversation with an ACP agent."""

    def __init__(self, model: AsyncModel, cwd: str | None = None) -> None:
        self.model = model
        self._conn: ClientSideConnection | None = (
            None  # Shared connection (managed by AgentManager)
        )
        self._client: ACPClientHandler | None = None  # Shared client handler
        self.agent_name: str | None = None  # Agent name from initialization
        self._cwd: str = cwd or os.getcwd()  # Working directory for agent sessions
        self.init_response: InitializeResponse | None = (
            None  # Store initialization response with capabilities
        )
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

    async def load_history(self) -> list[StreamEvent]:
        """Eagerly load an existing session and return its replayed history.

        Performs ``session/load`` now (instead of lazily on the first prompt) and
        captures the replay ``session/update``s the agent emits, returning them as
        StreamEvents (UserMessage / MessageChunk / ToolCall*) so the UI can render
        the prior transcript. Returns an empty list if there is no session to
        resume, it was already loaded, or the agent doesn't support loading.
        """
        await self.ensure_connected()

        if self._session_id is None or self._session_loaded or self._client is None:
            return []
        if self._conn is None:
            return []

        session_id = self._session_id
        queue = self._client.get_session_queue(session_id)

        # Discard anything queued before load so it can't be mistaken for history.
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._client.begin_replay(session_id)
        try:
            await self._conn.load_session(cwd=self._cwd, mcp_servers=[], session_id=session_id)
            self._session_loaded = True
            self._client.register_session(session_id, self._cwd)
            log.warning(f"✅ Loaded session for history replay: {session_id}")
        except Exception as e:
            # Couldn't load - leave _session_loaded False so the normal
            # fork/new fallback still runs on the first prompt.
            log.warning(f"❌ session/load for history failed: {type(e).__name__}: {e}")
            return []
        finally:
            self._client.end_replay(session_id)

        # Replay updates are enqueued by the connection's read loop as the load
        # request is serviced; yield once so any still-pending handlers run, then
        # drain everything that accumulated.
        await asyncio.sleep(0)
        history: list[StreamEvent] = []
        while not queue.empty():
            try:
                event = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if event is not None:
                history.append(event)
        log.warning(f"📜 Replayed {len(history)} history events for {session_id}")
        return history

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

    def respond_to_permission(self, request_id: str, option_id: str) -> None:
        """Respond to a permission request.

        Args:
            request_id: The ID of the permission request (from PermissionRequest event)
            option_id: The option_id to select (from the options in PermissionRequest event)
        """
        if self._client is None:
            log.warning("⚠️ Cannot respond to permission - no client available")
            return

        self._client.respond_to_permission(request_id, option_id)


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
        assert conv._conn is not None, "Connection must be established"
        session = await conv._conn.new_session(cwd=conv._cwd, mcp_servers=[])
        conv._session_id = session.session_id
        conv._session_loaded = True  # Mark as loaded so we don't try to restore it
        log.warning(f"Created new session ID: {conv._session_id}")
        log.warning(f"Session CWD: {conv._cwd}")

        # Register session cwd with client handler
        if conv._client and conv._session_id:
            conv._client.register_session(conv._session_id, conv._cwd)

        # Store in session storage for reuse
        storage = get_session_storage()
        if conv._session_id:
            storage.store_session_id(conv._cwd, conv.model.agent_command, conv._session_id)

    async def _load_session(self, conv: AsyncConversation) -> None:
        log.warning("🔄 Attempt: Trying session/load...")
        assert conv._conn is not None, "Connection must be established"
        assert conv._session_id is not None, "Session ID must exist for load"
        try:
            session = await conv._conn.load_session(
                cwd=conv._cwd, mcp_servers=[], session_id=conv._session_id
            )
            conv._session_loaded = True
            log.warning("✅ SUCCESS with session/load!")
            log.warning(f"Load response: {session}")

            # Register session cwd with client handler
            if conv._client and conv._session_id:
                conv._client.register_session(conv._session_id, conv._cwd)

            # Session ID stays the same after load
        except Exception as e2:
            log.warning(f"❌ session/load failed: {type(e2).__name__}: {e2}")
            raise

    async def _new_session(self, conv: AsyncConversation) -> None:
        log.warning("🔄 Attempt: Creating new session...")
        assert conv._conn is not None, "Connection must be established"
        session = await conv._conn.new_session(cwd=conv._cwd, mcp_servers=[])
        conv._session_id = session.session_id
        # Don't set _session_loaded = True here, since we failed to restore
        # and created a new session instead
        log.warning(f"Created new session ID: {conv._session_id}")
        log.warning(f"Session CWD: {conv._cwd}")

        # Register session cwd with client handler
        if conv._client and conv._session_id:
            conv._client.register_session(conv._session_id, conv._cwd)

        # Update session storage with new session ID
        storage = get_session_storage()
        if conv._session_id:
            storage.store_session_id(conv._cwd, conv.model.agent_command, conv._session_id)

    async def _fork_session(self, conv: AsyncConversation) -> None:
        log.warning("🔄: Trying session/fork via send_request...")
        assert conv._conn is not None, "Connection must be established"
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

            # Register session cwd with client handler
            if conv._client and conv._session_id:
                conv._client.register_session(conv._session_id, conv._cwd)

            # Update session storage with (possibly new) forked session ID
            storage = get_session_storage()
            if conv._session_id:
                storage.store_session_id(conv._cwd, conv.model.agent_command, conv._session_id)
        except Exception as e:
            log.warning(f"❌ session/fork failed: {type(e).__name__}: {e}")
            raise

    async def _ensure_connection(self) -> tuple[ClientSideConnection, str, ACPClientHandler]:
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
        assert conv._conn is not None
        assert conv._session_id is not None
        assert conv._client is not None
        return conv._conn, conv._session_id, conv._client

    async def __aiter__(self) -> AsyncGenerator[StreamEvent, None]:
        """Iterate over events from the agent."""
        from .events import MessageChunk

        self._iterated = True

        log.debug("ACP __aiter__: starting")
        conn, session_id, client = await self._ensure_connection()
        log.debug(f"ACP __aiter__: connection ready, session_id={session_id}")

        # Reset client for new turn
        client.reset_for_turn()

        # Get the session-specific event queue
        queue = client.get_session_queue(session_id)

        # Clear any stale events from previous interrupted responses
        cleared_count = 0
        while not queue.empty():
            try:
                queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        if cleared_count > 0:
            log.info(
                f"🧹 Cleared {cleared_count} stale events from queue before starting new prompt"
            )

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
                response = await conn.prompt(
                    session_id=session_id, prompt=[text_block(self._prompt)]
                )
                log.debug("ACP run_prompt: prompt completed")

                # Extract token usage from PromptResponse if available
                if response and response.usage:
                    usage = response.usage
                    log.info(
                        f"📊 TokenUsage from PromptResponse: "
                        f"in={usage.input_tokens} out={usage.output_tokens} "
                        f"total={usage.total_tokens}"
                    )
                    await queue.put(
                        TokenUsage(
                            prompt_tokens=usage.input_tokens,
                            completion_tokens=usage.output_tokens,
                            cached_tokens=usage.cached_read_tokens or 0,
                            total_tokens=usage.total_tokens,
                            thought_tokens=usage.thought_tokens,
                            cached_read_tokens=usage.cached_read_tokens,
                            cached_write_tokens=usage.cached_write_tokens,
                        )
                    )
            except Exception as e:
                log.debug(f"ACP run_prompt: error {e}")
                prompt_error = e
            finally:
                # Signal end of events
                await queue.put(None)
                log.debug("ACP run_prompt: signaled done")

        # Start prompt task in background
        log.debug("ACP __aiter__: starting prompt task")
        prompt_task = asyncio.create_task(run_prompt())

        # Yield events as they arrive in real-time
        log.debug("ACP __aiter__: starting event loop")
        while True:
            event = await queue.get()
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

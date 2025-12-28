"""Event types for streaming LLM responses.

These events are yielded by chain() to represent the chronological stream
of what's happening during LLM response generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# JSON type for tool call arguments and outputs
JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]


@dataclass
class MessageChunk:
    """A chunk of assistant message text."""

    text: str


@dataclass
class ThoughtChunk:
    """A chunk of thinking/reasoning text (extended thinking)."""

    text: str


@dataclass
class PlanChunk:
    """A chunk of agent planning/reasoning text (ACP agents)."""

    text: str = ""
    entries: list[dict[str, JSON]] | None = None


@dataclass
class ToolCallStart:
    """Tool call is starting."""

    id: str
    name: str
    arguments: dict[str, JSON]


@dataclass
class ToolCallProgress:
    """Tool call progress update."""

    id: str
    status: str


@dataclass
class ToolCallComplete:
    """Tool call finished."""

    id: str
    output: str


@dataclass
class TokenUsage:
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0


@dataclass
class PermissionRequest:
    """Request for user permission to execute a tool."""

    request_id: str
    session_id: str
    tool_call: dict[str, JSON]  # ToolCallUpdate as dict
    options: list[dict[str, JSON]]  # List of PermissionOption as dicts


@dataclass
class PermissionTimeout:
    """Permission request timed out."""

    request_id: str


# Union type for all possible events
StreamEvent = (
    MessageChunk
    | ThoughtChunk
    | PlanChunk
    | ToolCallStart
    | ToolCallProgress
    | ToolCallComplete
    | TokenUsage
    | PermissionRequest
    | PermissionTimeout
)

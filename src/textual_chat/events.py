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
class UserMessage:
    """A user message, replayed from session history on resume.

    Only emitted while loading/replaying a prior ACP session so the UI can
    reconstruct the human side of the transcript. Live turns never emit this
    (the user's message is already shown locally when they send it).
    """

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
    """Token usage information from PromptResponse.

    Per-turn token breakdown as specified in the ACP usage tracking RFC.
    """

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    # Extended fields from ACP RFC
    total_tokens: int | None = None
    thought_tokens: int | None = None
    cached_read_tokens: int | None = None
    cached_write_tokens: int | None = None


@dataclass
class Cost:
    """Cost information for a session."""

    amount: float
    currency: str  # ISO 4217 currency code (e.g., "USD", "EUR")


@dataclass
class UsageUpdate:
    """Context window and cost update notification.

    Sent via session/update with sessionUpdate: "usage_update" to report
    context window utilization and cumulative session cost.
    """

    # Context window (required)
    used: int  # Tokens currently in context
    size: int  # Total context window size in tokens

    # Cost (optional)
    cost: Cost | None = None

    @property
    def remaining(self) -> int:
        """Compute remaining tokens in context window."""
        return self.size - self.used

    @property
    def percentage(self) -> float:
        """Compute percentage of context window used."""
        if self.size <= 0:
            return 0.0
        return (self.used / self.size) * 100


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
    | UserMessage
    | ThoughtChunk
    | PlanChunk
    | ToolCallStart
    | ToolCallProgress
    | ToolCallComplete
    | TokenUsage
    | UsageUpdate
    | PermissionRequest
    | PermissionTimeout
)

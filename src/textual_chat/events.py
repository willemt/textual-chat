"""Event types for streaming LLM responses.

These events are yielded by chain() to represent the chronological stream
of what's happening during LLM response generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MessageChunk:
    """A chunk of assistant message text."""

    text: str


@dataclass
class ThoughtChunk:
    """A chunk of thinking/reasoning text (extended thinking)."""

    text: str


@dataclass
class ToolCallStart:
    """Tool call is starting."""

    id: str
    name: str
    arguments: dict[str, Any]


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


# Union type for all possible events
StreamEvent = (
    MessageChunk | ThoughtChunk | ToolCallStart | ToolCallProgress | ToolCallComplete | TokenUsage
)

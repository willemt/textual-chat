"""LiteLLM integration for chat completions."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import acompletion


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class LLMClient:
    """Client for LLM interactions using LiteLLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM client.

        Args:
            model: The model identifier (e.g., "gpt-4o-mini", "claude-3-sonnet", "ollama/llama2")
            api_key: Optional API key (can also be set via environment variables)
            api_base: Optional API base URL for custom endpoints
            system_prompt: Optional system prompt to prepend to all conversations
            **kwargs: Additional arguments passed to litellm
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.system_prompt = system_prompt
        self.extra_kwargs = kwargs
        self._tools: list[dict[str, Any]] = []

    def set_tools(self, tools: list[dict[str, Any]]) -> None:
        """Set available tools for the LLM.

        Args:
            tools: List of tool definitions in OpenAI function calling format
        """
        self._tools = tools

    def _build_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build the full message list including system prompt."""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(messages)
        return result

    async def complete(
        self,
        messages: list[dict[str, Any]],
    ) -> LLMResponse:
        """Get a completion from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            LLMResponse with content and/or tool calls
        """
        full_messages = self._build_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            **self.extra_kwargs,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self._tools:
            kwargs["tools"] = self._tools

        response = await acompletion(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream a completion from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Yields:
            Content chunks as they arrive
        """
        full_messages = self._build_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "stream": True,
            **self.extra_kwargs,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = await acompletion(**kwargs)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

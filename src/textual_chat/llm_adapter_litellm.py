"""LLM adapter - provides llm-like interface over litellm.

This adapter makes litellm code look similar to Simon Willison's llm library,
allowing for cleaner, more maintainable chat.py code.

Usage:
    model = get_async_model("claude-sonnet-4-20250514")
    conv = model.conversation()

    async for chunk in conv.chain("Hello", tools=[my_func], system="Be helpful"):
        print(chunk)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, Union, get_type_hints

import litellm
from litellm import acompletion

# JSON type for LLM messages and tool arguments
JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]

if TYPE_CHECKING:
    from .events import StreamEvent

# Suppress litellm's noisy logging
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

# Don't propagate to root logger (would interfere with TUI)
log = logging.getLogger(__name__)
log.propagate = False


class CacheDetails(TypedDict, total=False):
    """Cache-related token details."""

    cached_tokens: int  # OpenAI format
    cache_read_input_tokens: int  # Anthropic format
    cache_creation_input_tokens: int  # Anthropic format


def _extract_cache_details(usage: object) -> CacheDetails:
    """Extract cache-related details from usage object."""
    details: CacheDetails = {}
    # OpenAI format
    if hasattr(usage, "prompt_tokens_details"):
        ptd = usage.prompt_tokens_details
        if ptd and hasattr(ptd, "cached_tokens"):
            details["cached_tokens"] = ptd.cached_tokens or 0
    # Anthropic format
    if hasattr(usage, "cache_read_input_tokens"):
        details["cache_read_input_tokens"] = usage.cache_read_input_tokens or 0
    if hasattr(usage, "cache_creation_input_tokens"):
        details["cache_creation_input_tokens"] = usage.cache_creation_input_tokens or 0
    return details


def _empty_cache_details() -> CacheDetails:
    """Factory for empty CacheDetails."""
    return {}


@dataclass
class Usage:
    """Token usage information."""

    input: int = 0
    output: int = 0
    details: CacheDetails = field(default_factory=_empty_cache_details)


@dataclass
class ToolCall:
    """Represents a tool call from the model."""

    id: str
    name: str
    arguments: dict[str, JSON]


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    output: str


def _python_type_to_json(py_type: type | str) -> dict[str, JSON]:
    """Convert Python type to JSON schema."""
    # Handle string annotations (forward references)
    if isinstance(py_type, str):
        type_lower = py_type.lower()
        if "int" in type_lower:
            return {"type": "integer"}
        if "float" in type_lower:
            return {"type": "number"}
        if "bool" in type_lower:
            return {"type": "boolean"}
        if "list" in type_lower:
            return {"type": "array"}
        if "dict" in type_lower:
            return {"type": "object"}
        return {"type": "string"}

    mapping: dict[type, dict[str, JSON]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    return mapping.get(py_type, {"type": "string"})


def _func_to_tool_schema(func: Callable) -> dict[str, JSON]:
    """Convert a function to OpenAI tool schema."""
    # Try to get type hints, but fall back to raw annotations if evaluation fails
    # (e.g., when annotations reference types not in scope like 'App')
    try:
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    except Exception:
        # Fall back to raw annotations without evaluation
        hints = getattr(func, "__annotations__", {})
    sig = inspect.signature(func)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        properties[name] = _python_type_to_json(hints.get(name, str))
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip() or f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,  # type: ignore[dict-item]
                "required": required,  # type: ignore[dict-item]
            },
        },
    }


def _detect_model() -> tuple[str | None, str | None]:
    """Auto-detect the best available model. Returns (model, detected_from)."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini", "OPENAI_API_KEY"
    if os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_KEY"):
        return "github/gpt-4o-mini", "GITHUB_TOKEN"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini/gemini-1.5-flash", "GEMINI_API_KEY"
    if os.getenv("GROQ_API_KEY"):
        return "groq/llama-3.1-8b-instant", "GROQ_API_KEY"
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek/deepseek-chat", "DEEPSEEK_API_KEY"
    if os.getenv("ZAI_API_KEY"):
        # Use known working model - user can select others via model selector
        return "openai/GLM-4.5-air", "ZAI_API_KEY"

    # Check for Ollama
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=0.5)
        if resp.status_code == 200:
            tags = resp.json().get("models", [])
            if tags:
                return f"ollama/{tags[0]['name'].split(':')[0]}", "Ollama (localhost)"
    except Exception:
        pass

    return None, None


def get_async_model(
    model_id: str | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> AsyncModel:
    """Get an async model by ID, or auto-detect if not specified."""
    source = None
    if model_id is None:
        detected, source = _detect_model()
        if detected is None:
            raise ValueError("No model configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.")
        model_id = detected
        log.info(f"Auto-detected model: {model_id} from {source}")

    # Auto-configure credentials for Z.AI models
    # This allows Z.AI models to work even when OPENAI_API_KEY is set
    if os.getenv("ZAI_API_KEY") and _is_zai_model(model_id):
        if api_key is None:
            api_key = os.environ["ZAI_API_KEY"]
        if api_base is None:
            api_base = "https://api.z.ai/api/coding/paas/v4"

    return AsyncModel(model_id, api_key=api_key, api_base=api_base)


def _is_zai_model(model_id: str) -> bool:
    """Check if a model ID is a Z.AI model."""
    # Z.AI models are GLM models accessed via openai/ prefix
    model_lower = model_id.lower()
    return model_lower.startswith("openai/glm-") or model_lower.startswith("glm-")


def get_default_model() -> str:
    """Get the default model ID."""
    model, _ = _detect_model()
    return model or "gpt-4o-mini"


class AsyncModel:
    """Async model interface similar to llm library."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.api_base = api_base
        # Detect if this is a Claude model for cache control
        self.is_claude = "claude" in model_id.lower()

    def conversation(self) -> AsyncConversation:
        """Create a new conversation."""
        return AsyncConversation(self)


class AsyncConversation:
    """Manages conversation history and provides chain() for tool execution."""

    def __init__(self, model: AsyncModel):
        self.model = model
        self._messages: list[dict[str, JSON]] = []

    def chain(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tools: list[Callable] | None = None,
        options: dict | None = None,
    ) -> AsyncChainResponse:
        """Execute prompt with automatic tool calling loop.

        Args:
            prompt: User message
            system: System prompt
            tools: List of callable functions to use as tools
            options: Additional options (e.g., {"cache": True})

        Returns:
            AsyncChainResponse that yields events (MessageChunk, ToolCallStart, etc.)
        """
        return AsyncChainResponse(
            conversation=self,
            prompt=prompt,
            system=system,
            tools=tools,
            options=options or {},
        )

    def clear(self) -> None:
        """Clear conversation history."""
        self._messages.clear()


class AsyncChainResponse:
    """Async iterator that yields events (MessageChunk, ToolCallStart, etc.)."""

    def __init__(
        self,
        conversation: AsyncConversation,
        prompt: str,
        system: str | None,
        tools: list[Callable] | None,
        options: dict,
    ):
        self._conversation = conversation
        self._prompt = prompt
        self._system = system
        self._tools = tools or []
        self._options = options

        # Build tool schemas and lookup
        self._tool_schemas: list[dict] = []
        self._tool_lookup: dict[str, Callable] = {}
        for func in self._tools:
            schema = _func_to_tool_schema(func)
            self._tool_schemas.append(schema)
            self._tool_lookup[func.__name__] = func

        # Response tracking
        self._responses: list[AsyncResponse] = []
        self._current_response: AsyncResponse | None = None
        self._iterated = False

    def _build_kwargs(self) -> dict[str, JSON]:
        """Build kwargs for litellm acompletion."""
        model = self._conversation.model
        is_claude = model.is_claude
        cache = self._options.get("cache", False)

        # Build system message with cache_control for Anthropic
        if self._system:
            if is_claude and cache:
                system_msg = {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self._system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            else:
                system_msg = {"role": "system", "content": self._system}
            messages = [system_msg] + self._conversation._messages
        else:
            messages = list(self._conversation._messages)

        kwargs: dict[str, JSON] = {
            "model": model.model_id,
            "messages": messages,  # type: ignore[dict-item]
        }

        if model.api_key:
            kwargs["api_key"] = model.api_key
        if model.api_base:
            kwargs["api_base"] = model.api_base

        if self._tool_schemas:
            tools = self._tool_schemas.copy()
            # Add cache_control to last tool for Anthropic
            if tools and is_claude and cache:
                tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
            kwargs["tools"] = tools  # type: ignore[assignment]

        return kwargs

    async def __aiter__(self) -> AsyncGenerator[StreamEvent, None]:
        """Iterate over events from the model."""
        from .events import MessageChunk, TokenUsage

        self._iterated = True

        # Add user message to history
        self._conversation._messages.append({"role": "user", "content": self._prompt})

        total_usage: Usage | None = None

        while True:
            kwargs = self._build_kwargs()
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}

            response = await acompletion(**kwargs)

            # Create response tracker
            self._current_response = AsyncResponse()
            self._responses.append(self._current_response)

            # Collect streamed content and tool calls
            full_content = ""
            tool_calls_data: list[dict] = []
            last_chunk = None

            async for chunk in response:
                last_chunk = chunk
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta

                    # Yield text content as MessageChunk events
                    if delta.content:
                        full_content += delta.content
                        yield MessageChunk(delta.content)

                    # Collect tool calls
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            while len(tool_calls_data) <= idx:
                                tool_calls_data.append({"id": "", "name": "", "arguments": ""})
                            if tc.id:
                                tool_calls_data[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_data[idx]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls_data[idx]["arguments"] += tc.function.arguments

            # Store usage from last chunk
            if last_chunk and hasattr(last_chunk, "usage") and last_chunk.usage:
                usage = last_chunk.usage
                self._current_response._usage = Usage(
                    input=getattr(usage, "prompt_tokens", 0) or 0,
                    output=getattr(usage, "completion_tokens", 0) or 0,
                    details=_extract_cache_details(usage),
                )
                total_usage = self._current_response._usage

            self._current_response._text = full_content

            # If no tool calls, we're done
            if not tool_calls_data or not any(tc["name"] for tc in tool_calls_data):
                # Record assistant message
                if full_content:
                    self._conversation._messages.append(
                        {
                            "role": "assistant",
                            "content": full_content,
                        }
                    )
                break

            # Record assistant message with tool calls
            self._conversation._messages.append(
                {
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls_data
                        if tc["name"]
                    ],
                }
            )

            # Handle tool calls and emit events
            async for event in self._handle_tool_calls(tool_calls_data):
                yield event

        # Emit final token usage
        if total_usage:
            yield TokenUsage(
                prompt_tokens=total_usage.input,
                completion_tokens=total_usage.output,
                cached_tokens=total_usage.details.get("cached_tokens", 0)
                or total_usage.details.get("cache_read_input_tokens", 0),
            )

    async def _handle_tool_calls(
        self, tool_calls_data: list[dict[str, JSON]]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Parse tool calls, execute them, and emit events."""
        from .events import ToolCallComplete, ToolCallStart

        # Parse all tool calls and emit ToolCallStart events
        tool_calls: list[ToolCall] = []
        for tc_data in tool_calls_data:
            if not tc_data["name"]:
                continue
            tool_call = ToolCall(
                id=str(tc_data["id"]),
                name=str(tc_data["name"]),
                arguments=(json.loads(str(tc_data["arguments"])) if tc_data["arguments"] else {}),
            )
            tool_calls.append(tool_call)

            # Emit ToolCallStart event
            yield ToolCallStart(
                id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
            )

        # Separate sync vs async tools
        sync_calls: list[tuple[ToolCall, Callable]] = []
        async_calls: list[tuple[ToolCall, Callable]] = []
        unknown_calls: list[ToolCall] = []

        for tc in tool_calls:
            func = self._tool_lookup.get(tc.name)
            if func is None:
                unknown_calls.append(tc)
            elif asyncio.iscoroutinefunction(func):
                async_calls.append((tc, func))
            else:
                sync_calls.append((tc, func))

        # Results dict keyed by tool_call.id to maintain order
        results: dict[str, str] = {}

        # Execute sync tools
        for tc, func in sync_calls:
            try:
                output = str(func(**tc.arguments))
            except Exception as e:
                output = f"Error: {e}"
            results[tc.id] = output

        # Execute async tools in parallel
        async def run_async_tool(tc: ToolCall, func: Callable) -> tuple[str, str]:
            try:
                output = str(await func(**tc.arguments))
            except Exception as e:
                output = f"Error: {e}"
            return tc.id, output

        if async_calls:
            async_results = await asyncio.gather(
                *(run_async_tool(tc, func) for tc, func in async_calls)
            )
            for tc_id, output in async_results:
                results[tc_id] = output

        # Handle unknown tools
        for tc in unknown_calls:
            results[tc.id] = f"Unknown tool: {tc.name}"

        # Emit ToolCallComplete events and add to conversation history
        for tc in tool_calls:
            output = results[tc.id]

            # Emit ToolCallComplete event
            yield ToolCallComplete(id=tc.id, output=output)

            # Add to conversation history
            self._conversation._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                }
            )

    async def responses(self) -> AsyncGenerator[AsyncResponse, None]:
        """Iterate over response objects (for usage info)."""
        if not self._iterated:
            # Consume the iterator first
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
        """Get token usage information."""
        return self._usage

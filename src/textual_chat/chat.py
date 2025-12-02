"""LLM chat for humans.

The simplest way to add AI chat to your terminal app.

    from textual_chat import Chat

    chat = Chat()

    @chat.tool
    def search(query: str) -> str:
        '''Search the web.'''
        return results

    # In your Textual app:
    yield chat

That's it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys

# Setup logging to file (controlled by TEXTUAL_CHAT_LOGGING_LEVEL env var)
_log_level = os.environ.get("TEXTUAL_CHAT_LOGGING_LEVEL", "").upper()
if _log_level:
    logging.basicConfig(
        filename="textual_chat.log",
        level=getattr(logging, _log_level, logging.DEBUG),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
log = logging.getLogger(__name__)

# Separate logger for raw LLM content (controlled by TEXTUAL_CHAT_LOG_LLM env var)
llm_log = logging.getLogger("textual_chat.llm")
llm_log.setLevel(logging.DEBUG)
llm_log.propagate = False  # Don't propagate to root logger
if os.environ.get("TEXTUAL_CHAT_LOG_LLM"):
    _llm_handler = logging.FileHandler("llm_content.log", mode="w")
    _llm_handler.setLevel(logging.DEBUG)
    _llm_handler.setFormatter(logging.Formatter("%(asctime)s\n%(message)s\n"))
    llm_log.addHandler(_llm_handler)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, get_type_hints

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable, Markdown, Static, TextArea

from textual_golden import Golden, BLUE, PURPLE

from .tools.datatable import create_datatable_tools

import litellm
from litellm import acompletion

# Suppress litellm's noisy logging
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

Role = Literal["user", "assistant", "system", "tool"]

# Show thinking modes
INLINE = "inline"  # Show animated thinking inside assistant block
SEPARATE = "separate"  # Show thinking in separate block before response
ShowThinkingMode = Literal["inline", "separate"]


class ConfigurationError(Exception):
    """Raised when Chat is misconfigured."""

    pass


def _detect_model() -> tuple[str, str | None]:
    """Auto-detect the best available model. Returns (model, detected_from)."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini", "OPENAI_API_KEY"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini/gemini-1.5-flash", "GEMINI_API_KEY"
    if os.getenv("GROQ_API_KEY"):
        return "groq/llama-3.1-8b-instant", "GROQ_API_KEY"
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek/deepseek-chat", "DEEPSEEK_API_KEY"

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


def _friendly_error_message() -> str:
    """Generate a helpful error message when no model is configured."""
    return """
No LLM configured! Set up one of these:

  # OpenAI
  export OPENAI_API_KEY=sk-...

  # Anthropic
  export ANTHROPIC_API_KEY=sk-ant-...

  # Local with Ollama (free!)
  brew install ollama && ollama run llama3.2

  # Or specify a model directly:
  Chat(model="gpt-4o-mini", api_key="sk-...")

Docs: https://github.com/your/textual-chat
""".strip()


def _python_type_to_json(py_type: type) -> dict[str, Any]:
    """Convert Python type to JSON schema."""
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    return mapping.get(py_type, {"type": "string"})


def _func_to_tool(func: Callable) -> dict[str, Any]:
    """Convert a function to OpenAI tool schema."""
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
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
                "properties": properties,
                "required": required,
            },
        },
    }


class _ChatInput(TextArea):
    """Multiline input with Enter to submit, Shift+Enter for newlines."""

    class Submitted(Message):
        """User submitted their message."""

        def __init__(self, content: str) -> None:
            super().__init__()
            self.content = content

    def __init__(
        self,
        placeholder: str = "Message...",
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.show_line_numbers = False
        self._placeholder = placeholder

    def on_mount(self) -> None:
        """Set placeholder after mount."""
        self.placeholder = self._placeholder

    async def _on_key(self, event) -> None:
        """Handle key presses."""
        # Shift+Enter (comes through as ctrl+j) - insert newline
        if event.key == "ctrl+j":
            self.insert("\n")
            event.prevent_default()
            event.stop()
            return

        # Enter - submit message
        if event.key in ("enter", "ctrl+m"):
            content = self.text.strip()
            if content:
                self.post_message(self.Submitted(content))
                self.clear()
            event.prevent_default()
            event.stop()
            return

        # Page Up/Down - scroll chat messages
        if event.key in ("pageup", "pagedown"):
            try:
                container = self.app.query_one("#chat-messages")
                if event.key == "pageup":
                    container.scroll_page_up()
                else:
                    container.scroll_page_down()
                event.prevent_default()
                event.stop()
                return
            except NoMatches:
                pass

        # Let TextArea handle everything else
        await super()._on_key(event)


class Chat(Widget):
    """LLM chat widget for Textual apps.

    The simplest way to add AI chat to your TUI:

        from textual_chat import Chat

        chat = Chat()

        class MyApp(App):
            def compose(self):
                yield chat

    Add tools with a decorator:

        @chat.tool
        def get_weather(city: str) -> str:
            '''Get the weather for a city.'''
            return f"Sunny in {city}"

    Customize as needed:

        chat = Chat(
            model="claude-sonnet-4-20250514",
            system="You are a helpful pirate.",
            temperature=0.7,
        )
    """

    DEFAULT_CSS = """
    Chat {
        width: 100%;
        height: 100%;
        layout: vertical;
    }
    Chat #chat-messages {
        height: 1fr;
        padding: 1;
    }
    Chat #chat-input-area {
        height: auto;
        padding: 0 1 1 1;
        dock: bottom;
    }
    Chat #chat-input {
        width: 100%;
        height: auto;
        min-height: 3;
        max-height: 12;
        background: transparent;
        border: round $surface-lighten-1;
    }
    Chat #chat-input:focus {
        background: transparent;
        border: round $primary;
    }
    Chat #chat-status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    Chat .message {
        width: 100%;
        padding: 0 1;
        margin: 0;
        border: round $primary-darken-2;
    }
    Chat .message.user {
        border: round $primary;
    }
    Chat .message.assistant {
        border: round $accent;
    }
    Chat .message.system {
        border: round $warning-darken-2;
        color: $text-muted;
    }
    Chat .message.tool {
        border: round $success;
        color: $text-muted;
    }
    Chat .message.error {
        border: round $error;
        color: $error;
    }
    Chat .message.thinking {
        border: round gray;
        color: $text-muted;
        text-style: italic;
    }
    Chat .content {
        width: 100%;
        margin: 0;
        padding: 0;
    }
    Chat .content > * {
        margin: 0;
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    class Sent(Message):
        """User sent a message."""

        def __init__(self, content: str) -> None:
            super().__init__()
            self.content = content

    class Responded(Message):
        """Assistant responded."""

        def __init__(self, content: str) -> None:
            super().__init__()
            self.content = content

    class ToolCalled(Message):
        """A tool was called."""

        def __init__(self, name: str, args: dict, result: str) -> None:
            super().__init__()
            self.name = name
            self.args = args
            self.result = result

    class Thinking(Message):
        """Model is thinking (extended thinking)."""

        def __init__(self, content: str) -> None:
            super().__init__()
            self.content = content

    def __init__(
        self,
        model: str | None = None,
        *,
        system: str | None = None,
        placeholder: str = "Message...",
        tools: list[Any] | dict[str, Callable] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        thinking: bool | int = False,
        show_thinking: ShowThinkingMode | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        **llm_kwargs: Any,
    ) -> None:
        """Create a chat widget.

        Args:
            model: LLM model (auto-detected if not set)
            system: System prompt
            placeholder: Input placeholder text
            tools: List of functions and/or MCP servers, or dict of name -> function
            api_key: API key (usually from environment)
            api_base: Custom API endpoint
            temperature: Model temperature (0-1)
            thinking: Enable extended thinking. True for default budget, or int for token budget.
            show_thinking: How to display thinking - INLINE (animated in assistant) or SEPARATE (own block).
            **llm_kwargs: Extra args passed to LiteLLM
        """
        super().__init__(name=name, id=id, classes=classes)

        # Auto-detect model if needed
        if model is None:
            detected_model, detected_from = _detect_model()
            if detected_model is None:
                self._config_error = _friendly_error_message()
                self.model = "none"
            else:
                self._config_error = None
                self.model = detected_model
                self._detected_from = detected_from
        else:
            self._config_error = None
            self.model = model
            self._detected_from = None

        self.system = system or "You are a helpful assistant."
        self.placeholder = placeholder
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.thinking = thinking
        self.show_thinking = show_thinking
        self.llm_kwargs = llm_kwargs

        # State
        self._messages: list[dict[str, Any]] = []
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: list[dict[str, Any]] = []
        self._mcp_servers: list[Any] = []
        self._mcp_clients: list[Any] = []
        self._mcp_tools: dict[str, tuple[Any, str]] = (
            {}
        )  # tool_name -> (client, tool_name)
        self._is_responding = False
        self._cancel_requested = False

        # Register initial tools (smart detection)
        if tools:
            if isinstance(tools, dict):
                for tool_name, func in tools.items():
                    self._register_tool(func, tool_name)
            else:
                for item in tools:
                    if callable(item):
                        # It's a function
                        self._register_tool(item)
                    elif isinstance(item, DataTable):
                        # It's a DataTable - register query tools
                        self._register_datatable(item)
                    elif (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and isinstance(item[0], DataTable)
                    ):
                        # It's a (DataTable, name) tuple
                        self._register_datatable(item[0], item[1])
                    else:
                        # Assume it's an MCP server
                        self._mcp_servers.append(item)

    def _register_tool(self, func: Callable, name: str | None = None) -> None:
        """Register a tool function internally."""
        tool_name = name or func.__name__
        self._tools[tool_name] = func
        schema = _func_to_tool(func)
        if name:
            schema["function"]["name"] = name
        self._tool_schemas.append(schema)

    def _register_datatable(self, table: DataTable, name: str | None = None) -> None:
        """Register a DataTable as queryable tools for the LLM."""
        tools = create_datatable_tools(table, name or "table")
        for tool_name, func in tools.items():
            self._register_tool(func, tool_name)

    def tool(self, func: Callable) -> Callable:
        """Register a tool function.

        Example:
            @chat.tool
            def search(query: str) -> str:
                '''Search for information.'''
                return "results..."
        """
        self._register_tool(func)
        return func

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="chat-messages")
        with Vertical(id="chat-input-area"):
            yield Static("", id="chat-status")
            yield _ChatInput(placeholder=self.placeholder, id="chat-input")

    async def on_mount(self) -> None:
        """Show config error or model info on mount."""
        if self._config_error:
            self._show_error(self._config_error)
        elif self._detected_from:
            self._set_status(f"Using {self.model} (from {self._detected_from})")

        # Connect to MCP servers
        if self._mcp_servers:
            await self._connect_mcp_servers()

    async def _connect_mcp_servers(self) -> None:
        """Connect to MCP servers and register their tools."""
        try:
            from fastmcp import Client
        except ImportError:
            self._set_status("fastmcp not installed - MCP tools unavailable")
            return

        for server in self._mcp_servers:
            try:
                client = Client(server)
                await client.__aenter__()
                self._mcp_clients.append(client)

                # Get tools from this server
                tools = await client.list_tools()
                for tool in tools:
                    # Register MCP tool schema
                    schema = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or f"Call {tool.name}",
                            "parameters": (
                                tool.inputSchema
                                if hasattr(tool, "inputSchema")
                                else {"type": "object", "properties": {}}
                            ),
                        },
                    }
                    self._tool_schemas.append(schema)
                    self._mcp_tools[tool.name] = (client, tool.name)

            except Exception as e:
                self._set_status(f"MCP error: {e}")

    def _set_status(self, text: str) -> None:
        """Update the status line."""
        try:
            self.query_one("#chat-status", Static).update(text)
        except Exception:
            pass

    def _show_error(self, error: str) -> None:
        """Show an error message in the chat."""
        container = self.query_one("#chat-messages", ScrollableContainer)
        widget = _MessageWidget("error", error)
        container.mount(widget)

    def _add_message(
        self,
        role: str,
        content: str = "",
        loading: bool = False,
        title: str | None = None,
        before: "_MessageWidget | None" = None,
    ) -> "_MessageWidget":
        """Add a message to the UI."""
        container = self.query_one("#chat-messages", ScrollableContainer)
        widget = _MessageWidget(role, content, loading=loading, title=title)
        if before:
            # Insert before the specified widget using mount with before parameter
            container.mount(widget, before=before)
        else:
            container.mount(widget)
        container.scroll_end(animate=False)
        return widget

    async def on__chat_input_submitted(self, event: _ChatInput.Submitted) -> None:
        """Handle message submission."""
        content = event.content
        if not content or self._is_responding:
            return

        if self._config_error:
            self.notify("Please configure an LLM first", severity="error")
            return

        await self._send(content)

    async def _send(self, content: str) -> None:
        """Send a message and get a response."""
        self._is_responding = True
        self._cancel_requested = False

        # Add user message
        self._add_message("user", content)
        self._messages.append({"role": "user", "content": content})
        self.post_message(self.Sent(content))

        # Show responding indicator with animation
        assistant_widget = self._add_message("assistant", loading=True)
        self._set_status("Responding...")

        try:
            response = await self._get_response(assistant_widget)
            self._messages.append({"role": "assistant", "content": response})
            self._set_status("")
            self.post_message(self.Responded(response))

        except asyncio.CancelledError:
            assistant_widget.update_content("*Cancelled*")
            self._set_status("Cancelled")
        except Exception as e:
            error_msg = f"Error: {e}"
            assistant_widget.update_error(str(e))
            self._set_status(error_msg)
        finally:
            self._is_responding = False

    def _build_kwargs(self) -> dict[str, Any]:
        """Build kwargs for LiteLLM calls."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}] + self._messages,
            **self.llm_kwargs,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self._tool_schemas:
            kwargs["tools"] = self._tool_schemas

        # Extended thinking support (Claude models)
        if self.thinking:
            # Check for int but not bool (bool is subclass of int in Python)
            budget = (
                self.thinking
                if isinstance(self.thinking, int)
                and not isinstance(self.thinking, bool)
                else 1024
            )
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        return kwargs

    async def _get_response(self, widget: "_MessageWidget") -> str:
        """Get response from LLM with streaming, handling tool calls and thinking."""
        kwargs = self._build_kwargs()

        # If we have tools or thinking enabled, need non-streaming
        if self._tool_schemas or self.thinking:
            return await self._get_response_with_tools(widget, kwargs)

        # No tools, no thinking - stream directly
        return await self._stream_response(widget, kwargs)

    async def _stream_response(self, widget: "_MessageWidget", kwargs: dict) -> str:
        """Stream a response, updating the widget as chunks arrive."""
        kwargs["stream"] = True
        response = await acompletion(**kwargs)

        full_content = ""
        async for chunk in response:
            if self._cancel_requested:
                break
            if chunk.choices and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                widget.update_content(full_content)

        return full_content

    def _extract_thinking_and_text(self, message: Any) -> tuple[str | None, str]:
        """Extract thinking and text content from a response message.

        Returns:
            Tuple of (thinking_text, response_text)
        """
        # Litellm normalizes thinking into reasoning_content attribute
        thinking_text = getattr(message, "reasoning_content", None)
        response_text = message.content if isinstance(message.content, str) else ""

        log.debug(f"Extracted reasoning_content: {thinking_text}")
        log.debug(f"Extracted content: {response_text}")

        return thinking_text, response_text

    async def _get_response_with_tools(
        self, widget: "_MessageWidget", kwargs: dict
    ) -> str:
        """Get response handling tool calls and thinking."""
        response = await acompletion(**kwargs)
        message = response.choices[0].message

        # Log raw LLM response
        llm_log.debug(f"=== LLM Response ===\n{message}")

        # Log raw response for debugging
        log.debug(f"Response message: {message}")
        log.debug(f"Message content type: {type(message.content)}")
        log.debug(f"Message content: {message.content}")

        # Extract and display thinking if present
        thinking_text, text_content = self._extract_thinking_and_text(message)
        log.debug(f"Extracted thinking: {thinking_text}")
        log.debug(f"Extracted text: {text_content}")

        # Display thinking based on show_thinking mode
        if thinking_text and self.show_thinking:
            self.post_message(self.Thinking(thinking_text))
            if self.show_thinking == INLINE:
                # Show animated purple thinking in assistant widget, then update
                widget.show_thinking_animated(thinking_text)
                await asyncio.sleep(2)  # Show thinking briefly
            elif self.show_thinking == SEPARATE:
                # Add separate thinking block BEFORE the assistant response
                self._add_message(
                    "thinking", thinking_text, title="Thinking", before=widget
                )

        # Handle tool calls
        while message.tool_calls and not self._cancel_requested:
            # Record assistant's tool call request
            self._messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            # Execute each tool
            for tc in message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                self._set_status(f"Using {name}...")
                widget.show_tool_running(name, args)

                result = await self._call_tool(name, args)

                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

                self.post_message(self.ToolCalled(name, args, result))

            # Update kwargs with new messages and get next response
            kwargs["messages"] = [
                {"role": "system", "content": self.system}
            ] + self._messages

            self._set_status("Responding...")
            response = await acompletion(**kwargs)
            message = response.choices[0].message

            # Log raw LLM response
            llm_log.debug(f"=== LLM Response (after tools) ===\n{message}")

            # Extract thinking from new response
            thinking_text, text_content = self._extract_thinking_and_text(message)
            if thinking_text and self.show_thinking:
                self.post_message(self.Thinking(thinking_text))
                if self.show_thinking == INLINE:
                    widget.show_thinking_animated(thinking_text)
                    await asyncio.sleep(2)
                elif self.show_thinking == SEPARATE:
                    self._add_message(
                        "thinking", thinking_text, title="Thinking", before=widget
                    )

        # Final response
        if text_content:
            widget.update_content(text_content)
            return text_content

        # If no content after tools, stream a new response
        kwargs["messages"] = [
            {"role": "system", "content": self.system}
        ] + self._messages
        return await self._stream_response(widget, kwargs)

    async def _call_tool(self, name: str, args: dict) -> str:
        """Execute a tool (local or MCP)."""
        # Check local tools first
        if name in self._tools:
            try:
                func = self._tools[name]
                result = func(**args)
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
            except Exception as e:
                return f"Tool error: {e}"

        # Check MCP tools
        if name in self._mcp_tools:
            try:
                client, tool_name = self._mcp_tools[name]
                result = await client.call_tool(tool_name, args)
                # Handle different result types
                if hasattr(result, "content"):
                    # MCP result with content blocks
                    if isinstance(result.content, list):
                        texts = [c.text for c in result.content if hasattr(c, "text")]
                        return "\n".join(texts) if texts else str(result.content)
                    return str(result.content)
                return str(result)
            except Exception as e:
                return f"MCP tool error: {e}"

        return f"Unknown tool: {name}"

    def action_clear(self) -> None:
        """Clear the chat."""
        self._messages.clear()
        container = self.query_one("#chat-messages", ScrollableContainer)
        container.remove_children()
        self._set_status("Cleared")

    def action_cancel(self) -> None:
        """Cancel current response."""
        if self._is_responding:
            self._cancel_requested = True
            self._set_status("Cancelling...")

    # Convenience methods for programmatic use

    def say(self, message: str) -> None:
        """Send a message programmatically."""
        asyncio.create_task(self._send(message))

    def add_system_message(self, content: str) -> None:
        """Add a system message to the chat."""
        self._add_message("system", content)


class _MessageWidget(Static):
    """A chat message."""

    def __init__(
        self, role: str, content: str, loading: bool = False, title: str | None = None
    ) -> None:
        super().__init__(classes=f"message {role}")
        self.role = role
        self._content = content
        self._loading = loading
        self.border_title = title or role.title()

    def compose(self) -> ComposeResult:
        if self._loading:
            yield Golden("Responding...", classes="content")
        else:
            yield Markdown(self._content, classes="content")

    def _scroll_parent(self) -> None:
        """Scroll parent container to show this message."""
        try:
            if self.parent:
                self.parent.scroll_end(animate=False)
        except Exception:
            pass

    def update_content(self, content: str) -> None:
        self._content = content
        self._loading = False
        try:
            # Remove any existing content widget
            old_content = self.query_one(".content")
            old_content.remove()
        except NoMatches:
            pass
        # Mount new Markdown content
        self.mount(Markdown(content, classes="content"))
        # Scroll after refresh to ensure content is rendered
        self.call_after_refresh(self._scroll_parent)

    def show_tool_running(self, tool_name: str, args: dict) -> None:
        """Show animated indicator while tool is running."""
        try:
            # Remove current content
            content = self.query_one(".content")
            content.remove()
        except NoMatches:
            pass
        # Format args for display
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        # Mount "Using" in blue wave, rest in regular text
        label = Golden("Using ", colors=BLUE)
        label.styles.width = "auto"
        text = Static(f"{tool_name}({args_str})")
        text.styles.width = "1fr"
        container = Horizontal(label, text, classes="content")
        container.styles.height = "auto"
        container.styles.width = "100%"
        self.mount(container)
        self.call_after_refresh(self._scroll_parent)

    def show_thinking_animated(self, thinking_text: str) -> None:
        """Show animated purple 'Thinking:' label with regular text."""
        try:
            content = self.query_one(".content")
            content.remove()
        except NoMatches:
            pass
        text = Static(f"Thinking: {thinking_text}")
        text.styles.width = "1fr"
        text.styles.text_style = "italic"
        container = Horizontal(text, classes="content")
        container.styles.height = "auto"
        container.styles.width = "100%"
        self.mount(container)
        self.call_after_refresh(self._scroll_parent)

    def update_error(self, error: str) -> None:
        """Show error message in red."""
        self._content = f"Error: {error}"
        self._loading = False
        try:
            content = self.query_one(".content")
            content.remove()
        except NoMatches:
            pass
        # Use Static with Rich markup for colored error
        self.mount(Static(f"[red]Error: {error}[/red]", classes="content"))
        self.call_after_refresh(self._scroll_parent)

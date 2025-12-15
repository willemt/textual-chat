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
import logging
import os

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

import types
from collections.abc import Callable
from typing import Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import DataTable, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from . import llm_adapter_litellm
from .session_storage import SessionStorage
from .tools.datatable import create_datatable_tools
from .tools.introspection import introspect_app
from .widgets import MessageWidget, SessionPromptInput, ToolUse

# Default adapter
_default_adapter = llm_adapter_litellm

# Re-export for backwards compatibility
get_async_model = llm_adapter_litellm.get_async_model
AsyncModel = llm_adapter_litellm.AsyncModel
AsyncConversation = llm_adapter_litellm.AsyncConversation
ToolCall = llm_adapter_litellm.ToolCall
ToolResult = llm_adapter_litellm.ToolResult

Role = Literal["user", "assistant", "system", "tool"]

# Show thinking modes
INLINE: Literal["inline"] = "inline"  # Show animated thinking inside assistant block
SEPARATE: Literal["separate"] = "separate"  # Show thinking in separate block before response
ShowThinkingMode = Literal["inline", "separate"]


class ConfigurationError(Exception):
    """Raised when Chat is misconfigured."""

    pass


class ModelSelectModal(ModalScreen[str | None]):
    """Modal for selecting an LLM model."""

    DEFAULT_CSS = """
    ModelSelectModal {
        align: center middle;
    }
    ModelSelectModal > Vertical {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    ModelSelectModal #title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }
    ModelSelectModal OptionList {
        height: auto;
        max-height: 20;
    }
    ModelSelectModal #model-info {
        text-align: center;
        color: $text-muted;
        height: 1;
        margin-top: 1;
    }
    ModelSelectModal #hint {
        text-align: center;
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, models: list[tuple[str, str, str]], current: str | None = None) -> None:
        """Initialize modal.

        Args:
            models: List of (display_name, model_id, provider) tuples
            current: Currently selected model_id
        """
        super().__init__()
        self.models = models
        self.current = current
        self._model_info: dict[str, str] = {mid: provider for _, mid, provider in models}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Select Model", id="title")
            options = [
                Option(f"{'● ' if mid == self.current else '  '}{name}", id=mid)
                for name, mid, _ in self.models
            ]
            yield OptionList(*options)
            yield Static("", id="model-info")
            yield Static("[i]Enter to select, Escape to cancel[/i]", id="hint")

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Update info when option is highlighted."""
        model_id = event.option.id
        if model_id and model_id in self._model_info:
            info = self._model_info[model_id]
            self.query_one("#model-info", Static).update(f"[i]{info}[/i]")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AgentSelectModal(ModalScreen[str | None]):
    """Modal for selecting an ACP agent."""

    DEFAULT_CSS = """
    AgentSelectModal {
        align: center middle;
    }
    AgentSelectModal > Vertical {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    AgentSelectModal #title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }
    AgentSelectModal OptionList {
        height: auto;
        max-height: 20;
    }
    AgentSelectModal #agent-info {
        text-align: center;
        color: $text-muted;
        height: 1;
        margin-top: 1;
    }
    AgentSelectModal #hint {
        text-align: center;
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, agents: list[tuple[str, str, str]], current: str | None = None) -> None:
        """Initialize modal.

        Args:
            agents: List of (display_name, agent_command, description) tuples
            current: Currently selected agent_command
        """
        super().__init__()
        self.agents = agents
        self.current = current
        self._agent_info: dict[str, str] = {cmd: desc for _, cmd, desc in agents}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Select Agent", id="title")
            options = [
                Option(f"{'● ' if cmd == self.current else '  '}{name}", id=cmd)
                for name, cmd, _ in self.agents
            ]
            yield OptionList(*options)
            yield Static("", id="agent-info")
            yield Static("[i]Enter to select, Escape to cancel[/i]", id="hint")

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Update info when option is highlighted."""
        agent_cmd = event.option.id
        if agent_cmd and agent_cmd in self._agent_info:
            info = self._agent_info[agent_cmd]
            self.query_one("#agent-info", Static).update(f"[i]{info}[/i]")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


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

    async def _on_key(self, event: Any) -> None:
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
        margin: 0 0 1 0;
        padding: 0;
    }
    Chat .content > *:last-child {
        margin-bottom: 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("ctrl+m", "select_model", "Models", show=False),
        Binding("ctrl+a", "select_agent", "Agents", show=False),
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

    class ModelChanged(Message):
        """Model was changed."""

        def __init__(self, model: str) -> None:
            super().__init__()
            self.model = model

    def __init__(
        self,
        model: str | None = None,
        *,
        adapter: Literal["litellm", "acp"] | Any = "litellm",
        system: str | None = None,
        placeholder: str = "Message...",
        tools: list[Any] | dict[str, Callable] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        thinking: bool | int = False,
        show_thinking: ShowThinkingMode | None = None,
        show_token_usage: bool = False,
        show_model_selector: bool = True,
        introspect: bool = True,
        introspect_scope: Literal["app", "screen", "parent"] = "app",
        assistant_name: str | None = None,
        cwd: str | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        **llm_kwargs: Any,
    ) -> None:
        """Create a chat widget.

        Args:
            model: LLM model (auto-detected if not set), or agent command for ACP
            adapter: Backend adapter - "litellm" (default) or "acp", or a module
            system: System prompt
            placeholder: Input placeholder text
            tools: List of functions and/or MCP servers, or dict of name -> function
            api_key: API key (usually from environment)
            api_base: Custom API endpoint
            temperature: Model temperature (0-1)
            thinking: Enable extended thinking. True for default budget, or int for token budget.
            show_thinking: How to display thinking - INLINE (animated in assistant) or SEPARATE (own block).
            show_token_usage: Show token usage in status bar after each response.
            show_model_selector: Show Ctrl+M to open model selector (default True).
            introspect: Auto-discover app widgets and create tools for LLM (default True).
            introspect_scope: Scope for introspection - "app", "screen", or "parent" (default "app").
            assistant_name: Override the assistant's display name (defaults to agent name from ACP or "Assistant").
            cwd: Working directory for ACP adapter (optional).
            **llm_kwargs: Extra args passed to LiteLLM
        """
        super().__init__(name=name, id=id, classes=classes)

        # Select adapter
        self._adapter: types.ModuleType
        if adapter == "litellm":
            self._adapter = llm_adapter_litellm
        elif adapter == "acp":
            from . import llm_adapter_acp

            self._adapter = llm_adapter_acp
        else:
            # Assume it's a module
            self._adapter = adapter  # type: ignore[assignment]

        # Model configuration
        self._model_id = model  # Will use adapter's auto-detect if None
        self._base_system = system or "You are a helpful assistant."
        self.system = self._base_system  # May be augmented by introspection
        self.placeholder = placeholder
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.thinking = thinking
        self.show_thinking = show_thinking
        self.show_token_usage = show_token_usage
        self.show_model_selector = show_model_selector
        self.introspect = introspect
        self.introspect_scope = introspect_scope
        self.assistant_name = assistant_name  # Override for assistant display name
        self.cwd = cwd  # Working directory for ACP adapter
        self.llm_kwargs = llm_kwargs

        # Adapter state (initialized in on_mount)
        self._model: AsyncModel | None = None
        self._conversation: AsyncConversation | None = None
        self._config_error: str | None = None

        # Session management (for ACP adapter)
        self._session_storage: SessionStorage | None = None
        self._pending_session_prompt = False
        self._message_history: list[dict[str, str]] = []  # Track messages for session persistence

        # Tool state
        self._tools: dict[str, Callable] = {}  # Local tools
        self._mcp_servers: list[Any] = []
        self._mcp_clients: list[Any] = []
        self._mcp_tools: dict[str, tuple[Any, str]] = {}  # MCP tool_name -> (client, tool_name)

        # Response state
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

    def _get_assistant_title(self) -> str | None:
        """Get the effective assistant title for display.

        Returns the override assistant_name if set, otherwise falls back to
        the agent name from ACP initialization, or None.
        """
        if self.assistant_name:
            return self.assistant_name
        if self._conversation and hasattr(self._conversation, "agent_name"):
            agent_name = getattr(self._conversation, "agent_name", None)
            if agent_name:
                return str(agent_name)
        return None

    def _register_tool(self, func: Callable, name: str | None = None) -> None:
        """Register a tool function internally."""
        tool_name = name or func.__name__
        # Wrap with custom name if provided
        if name and func.__name__ != name:
            func.__name__ = name
        self._tools[tool_name] = func

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
        """Initialize model and show status on mount."""
        # For ACP adapter with no model specified, show agent selector
        if self._model_id is None and self._adapter.__name__ == "textual_chat.llm_adapter_acp":
            self._show_agent_selector()
            return

        # Initialize the adapter model
        try:
            self._model = self._adapter.get_async_model(
                self._model_id,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            if self._model:
                # Pass cwd to ACP adapter if set
                if self.cwd and self._adapter.__name__ == "textual_chat.llm_adapter_acp":
                    self._conversation = self._model.conversation(cwd=self.cwd)  # type: ignore[call-arg]
                else:
                    self._conversation = self._model.conversation()
                self._update_model_status()
        except ValueError as e:
            self._config_error = str(e)
            self._show_error(self._config_error)

        # Initialize session storage for ACP adapter
        if self._adapter.__name__ == "textual_chat.llm_adapter_acp":
            self._session_storage = SessionStorage()
            # Check for previous session
            if self._model and self._session_storage:
                prev_session_data = self._session_storage.get_session(self._model.model_id)
                if prev_session_data:
                    self._pending_session_prompt = True
                    self._show_session_prompt_in_input()

        # Introspect app and register discovered tools
        if self.introspect:
            self._perform_introspection()

        # Connect to MCP servers
        if self._mcp_servers:
            await self._connect_mcp_servers()

    def _update_model_status(self) -> None:
        """Update status to show current model."""
        if not self._model:
            return
        model_id = self._model.model_id

        # For ACP adapter, show agent selector hint
        if self._adapter.__name__ == "textual_chat.llm_adapter_acp":
            self._set_status(f"Using {model_id}. Ctrl+A for agents")
        elif self.show_model_selector:
            self._set_status(f"Using {model_id}. Ctrl+M for models")
        else:
            self._set_status(f"Using {model_id}")

    def _get_available_models(self) -> list[tuple[str, str, str]]:
        """Get list of available models as (display_name, model_id, provider) tuples."""
        models = []

        # Anthropic models
        if os.getenv("ANTHROPIC_API_KEY"):
            models.extend(
                [
                    ("Claude Sonnet 4", "claude-sonnet-4-20250514", "Anthropic"),
                    ("Claude Opus 4", "claude-opus-4-20250514", "Anthropic"),
                    ("Claude Haiku 3.5", "claude-3-5-haiku-latest", "Anthropic"),
                ]
            )

        # OpenAI models
        if os.getenv("OPENAI_API_KEY"):
            models.extend(
                [
                    ("GPT-4o", "gpt-4o", "OpenAI"),
                    ("GPT-4o Mini", "gpt-4o-mini", "OpenAI"),
                    ("GPT-4 Turbo", "gpt-4-turbo", "OpenAI"),
                    ("o1", "o1", "OpenAI"),
                    ("o1-mini", "o1-mini", "OpenAI"),
                ]
            )

        # GitHub Models
        if os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_KEY"):
            models.extend(
                [
                    ("GPT-4o (GitHub)", "github/gpt-4o", "GitHub Models"),
                    ("GPT-4o Mini (GitHub)", "github/gpt-4o-mini", "GitHub Models"),
                    ("Claude Sonnet 3.5 (GitHub)", "github/claude-3.5-sonnet", "GitHub Models"),
                ]
            )

        # Google models
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            models.extend(
                [
                    ("Gemini 1.5 Flash", "gemini/gemini-1.5-flash", "Google"),
                    ("Gemini 1.5 Pro", "gemini/gemini-1.5-pro", "Google"),
                    ("Gemini 2.0 Flash", "gemini/gemini-2.0-flash-exp", "Google"),
                ]
            )

        # Groq models
        if os.getenv("GROQ_API_KEY"):
            models.extend(
                [
                    ("Llama 3.1 8B (Groq)", "groq/llama-3.1-8b-instant", "Groq"),
                    ("Llama 3.1 70B (Groq)", "groq/llama-3.1-70b-versatile", "Groq"),
                    ("Mixtral 8x7B (Groq)", "groq/mixtral-8x7b-32768", "Groq"),
                ]
            )

        # DeepSeek models
        if os.getenv("DEEPSEEK_API_KEY"):
            models.extend(
                [
                    ("DeepSeek Chat", "deepseek/deepseek-chat", "DeepSeek"),
                    ("DeepSeek Coder", "deepseek/deepseek-coder", "DeepSeek"),
                ]
            )

        return models

    def _get_available_agents(self) -> list[tuple[str, str, str]]:
        """Get list of available ACP agents as (display_name, agent_command, description) tuples."""
        agents = [
            (
                "Claude Code",
                "claude-code-acp",
                "Anthropic's official CLI agent with web search, bash, and file tools",
            ),
            (
                "OpenCode",
                "opencode acp",
                "Open-source alternative agent with similar capabilities",
            ),
        ]
        return agents

    def _show_agent_selector(self) -> None:
        """Show agent selection modal."""
        agents = self._get_available_agents()
        current = self._model.model_id if self._model else None
        self.app.push_screen(
            AgentSelectModal(agents, current),
            self._on_agent_selected,
        )

    def _on_agent_selected(self, agent_command: str | None) -> None:
        """Handle agent selection from modal."""
        if not agent_command:
            return

        # Set the model ID to the selected agent
        self._model_id = agent_command

        # Initialize the agent
        try:
            self._model = self._adapter.get_async_model(
                self._model_id,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            if self._model:
                # Pass cwd to ACP adapter if set
                if self.cwd and self._adapter.__name__ == "textual_chat.llm_adapter_acp":
                    self._conversation = self._model.conversation(cwd=self.cwd)  # type: ignore[call-arg]
                else:
                    self._conversation = self._model.conversation()
                self._update_model_status()

                # Initialize session storage for ACP adapter
                if self._adapter.__name__ == "textual_chat.llm_adapter_acp":
                    self._session_storage = SessionStorage()
                    # Check for previous session
                    if self._model and self._session_storage:
                        prev_session_data = self._session_storage.get_session(self._model.model_id)
                        if prev_session_data:
                            self._pending_session_prompt = True
                            self._show_session_prompt_in_input()

                # Introspect app and register discovered tools
                if self.introspect:
                    self._perform_introspection()

        except Exception as e:
            self._set_status(f"Failed to initialize agent: {e}")
            log.exception(f"Failed to initialize agent: {e}")

    def action_select_agent(self) -> None:
        """Show agent selection modal (ACP adapter only)."""
        if self._adapter.__name__ != "textual_chat.llm_adapter_acp":
            return
        self._show_agent_selector()

    def action_select_model(self) -> None:
        """Show model selection modal."""
        if not self.show_model_selector:
            return
        models = self._get_available_models()
        if not models:
            self.notify("No models available. Set API keys.", severity="warning")
            return
        current = self._model.model_id if self._model else None
        self.app.push_screen(
            ModelSelectModal(models, current),
            self._on_model_selected,
        )

    def _on_model_selected(self, model_id: str | None) -> None:
        """Handle model selection from modal."""
        if not model_id:
            return
        if self._model and model_id == self._model.model_id:
            return
        try:
            self._model = self._adapter.get_async_model(
                model_id,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            if self._model:
                # Pass cwd to ACP adapter if set
                if self.cwd and self._adapter.__name__ == "textual_chat.llm_adapter_acp":
                    self._conversation = self._model.conversation(cwd=self.cwd)  # type: ignore[call-arg]
                else:
                    self._conversation = self._model.conversation()
                self._update_model_status()
                self.post_message(self.ModelChanged(model_id))
        except Exception as e:
            self._set_status(f"Failed: {e}")

    def _perform_introspection(self) -> None:
        """Introspect the app and register discovered tools."""
        try:
            # Exclude our own widget and children
            exclude = {self.id} if self.id else set()
            exclude.add("chat-input")
            exclude.add("chat-messages")
            exclude.add("chat-status")

            tools, context = introspect_app(
                self.app,
                scope=self.introspect_scope,
                exclude_widgets=exclude,
            )

            # Register discovered tools
            for tool_name, func in tools.items():
                self._register_tool(func, tool_name)

            # Augment system prompt with context
            if context:
                self.system = f"{self._base_system}\n\nApplication context:\n{context}"

            log.debug(f"Introspection found {len(tools)} tools")
            log.debug(f"Context: {context}")
        except Exception as e:
            log.debug(f"Introspection failed: {e}")

    def _make_mcp_wrapper(self, client: Any, name: str, description: str | None) -> Callable:
        """Create an MCP tool wrapper function with proper closure capture."""

        async def wrapper(**kwargs: Any) -> str:
            result = await client.call_tool(name, kwargs)
            if hasattr(result, "content"):
                if isinstance(result.content, list):
                    texts = [c.text for c in result.content if hasattr(c, "text")]
                    return "\n".join(texts) if texts else str(result.content)
                return str(result.content)
            return str(result)

        wrapper.__name__ = name
        wrapper.__doc__ = description or f"Call {name}"
        return wrapper

    async def _connect_mcp_servers(self) -> None:
        """Connect to MCP servers and register their tools as wrapper functions."""
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

                # Get tools from this server and create wrapper functions
                tools = await client.list_tools()
                for tool in tools:
                    # Create wrapper with proper closure capture
                    wrapper = self._make_mcp_wrapper(client, tool.name, tool.description)
                    self._tools[tool.name] = wrapper
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
        widget = MessageWidget("error", error)
        container.mount(widget)

    def _show_session_prompt_in_input(self) -> None:
        """Replace the input with session resumption prompt."""
        try:
            # Remove the text input
            text_input = self.query_one("#chat-input", _ChatInput)
            text_input.remove()

            # Add the session prompt in its place
            input_area = self.query_one("#chat-input-area")
            prompt = SessionPromptInput(id="session-prompt")
            input_area.mount(prompt)
        except Exception as e:
            log.exception(f"Failed to show session prompt: {e}")

    def _add_message(
        self,
        role: str,
        content: str = "",
        loading: bool = False,
        title: str | None = None,
        before: MessageWidget | None = None,
    ) -> MessageWidget:
        """Add a message to the UI."""
        container = self.query_one("#chat-messages", ScrollableContainer)
        widget = MessageWidget(role, content, loading=loading, title=title)
        if before:
            # Insert before the specified widget using mount with before parameter
            container.mount(widget, before=before)
        else:
            container.mount(widget)
        container.scroll_end(animate=False)
        return widget

    async def on_session_prompt_input_session_choice(
        self, event: SessionPromptInput.SessionChoice
    ) -> None:
        """Handle session resumption choice."""
        self._pending_session_prompt = False

        # Remove the prompt and restore the text input
        try:
            prompt = self.query_one("#session-prompt", SessionPromptInput)
            prompt.remove()
            input_area = self.query_one("#chat-input-area")
            input_area.mount(_ChatInput(placeholder=self.placeholder, id="chat-input"))
        except Exception as e:
            log.exception(f"Failed to restore input: {e}")
            return

        if not self._session_storage or not self._model:
            return

        if event.resume:
            # Resume the previous session
            prev_session_data = self._session_storage.get_session(self._model.model_id)
            log.warning("========== SESSION RESUME CLICKED ==========")
            log.warning(f"Previous session data: {prev_session_data}")
            if prev_session_data:
                # Restore session ID to try loading it
                session_id = prev_session_data.get("session_id")
                log.warning(f"Found session ID in storage: {session_id}")
                if session_id and self._conversation and hasattr(self._conversation, "_session_id"):
                    self._conversation._session_id = session_id
                    log.warning(f"✅ Set conversation._session_id to: {session_id}")

                    # IMMEDIATELY trigger session loading by ensuring connection
                    log.warning("🔄 Triggering immediate session load...")
                    try:
                        # Force the conversation to connect and load the session NOW
                        # Use ensure_connected() instead of creating a dummy chain
                        if hasattr(self._conversation, "ensure_connected"):
                            await self._conversation.ensure_connected()
                        log.warning("✅ Session load triggered successfully")
                    except Exception as e:
                        log.warning(f"❌ Failed to trigger session load: {e}")
                else:
                    log.warning(
                        f"❌ Could not set session ID. session_id={session_id}, has_attr={hasattr(self._conversation, '_session_id')}"
                    )

                # Restore message history to UI
                messages = prev_session_data.get("messages", [])
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    # Use assistant name for assistant messages
                    title = None
                    if role == "assistant":
                        title = self._get_assistant_title()
                    self._add_message(role, content, title=title)
                    # Add to history tracking
                    self._message_history.append({"role": role, "content": content})

                # Show confirmation
                if messages:
                    self._set_status(f"Resumed session with {len(messages)} messages")
        else:
            # Start fresh - clear the stored session
            self._session_storage.clear_session(self._model.model_id)
            self._set_status("Starting new session")

    async def on__chat_input_submitted(self, event: _ChatInput.Submitted) -> None:
        """Handle message submission."""
        content = event.content
        if not content or self._is_responding:
            return

        if self._config_error:
            self.notify("Please configure an LLM first", severity="error")
            return

        # Don't allow sending while session prompt is pending
        if self._pending_session_prompt:
            self.notify(
                "Please choose whether to resume the previous session first",
                severity="warning",
            )
            return

        await self._send(content)

    async def _send(self, content: str) -> None:
        """Send a message and get a response using the adapter."""
        self._is_responding = True
        self._cancel_requested = False

        # Add user message to UI and history
        self._add_message("user", content)
        self._message_history.append({"role": "user", "content": content})
        self.post_message(self.Sent(content))

        # Show responding indicator with animation
        # Use assistant name (from override or ACP agent)
        agent_title = self._get_assistant_title()
        assistant_widget = self._add_message("assistant", loading=True, title=agent_title)
        self._set_status("Responding...")

        try:
            # Guard against uninitialized conversation
            if not self._conversation:
                self._show_error("Conversation not initialized")
                assistant_widget.mark_complete()
                return

            # Get tools list for adapter
            tools = list(self._tools.values()) if self._tools else None

            # Build options for caching
            options = {"cache": self._model.is_claude} if self._model else {}

            # Callbacks for tool display
            def before_tool(tc: ToolCall) -> None:
                tu = ToolUse(tc.name, tc.arguments)
                assistant_widget.add_tooluse(tu, tc.id)
                self._set_status(f"Using {tc.name}...")

            def after_tool(tc: ToolCall, result: ToolResult) -> None:
                assistant_widget.complete_tooluse(tc.id)
                self.post_message(self.ToolCalled(tc.name, tc.arguments, result.output))
                self._set_status("Responding...")

            # Use adapter's chain() for automatic tool handling
            chain = self._conversation.chain(
                content,
                system=self.system,
                tools=tools,
                before_call=before_tool,
                after_call=after_tool,
                options=options,
            )

            full_text = ""

            async for chunk in chain:
                if self._cancel_requested:
                    break
                # Get stream (creates new one after tool use)
                stream = assistant_widget.ensure_stream()
                await stream.write(chunk)
                full_text += chunk

            # Close the stream and mark message complete
            if assistant_widget._stream:
                await assistant_widget._stream.stop()
            assistant_widget.mark_complete()

            # Scroll to bottom
            container = self.query_one("#chat-messages", ScrollableContainer)
            container.scroll_end(animate=False)

            # Get token usage from final response (per-message, not cumulative)
            last_usage = None
            async for resp in chain.responses():
                usage = await resp.usage()
                if usage:
                    last_usage = usage

            if last_usage:
                cached = last_usage.details.get(
                    "cache_read_input_tokens", 0
                ) or last_usage.details.get("cached_tokens", 0)
                llm_log.debug(
                    f"=== Token Usage ===\n"
                    f"Input: {last_usage.input}, Output: {last_usage.output}, Cached: {cached}"
                )
                if self.show_token_usage:
                    assistant_widget.set_token_usage(last_usage.input, last_usage.output, cached)

            llm_log.debug(f"=== LLM Response ===\n{full_text}")
            self._set_status("")
            self.post_message(self.Responded(full_text))

            # Track assistant response in history
            self._message_history.append({"role": "assistant", "content": full_text})

            # Save session ID and message history for ACP adapter
            if (
                self._session_storage
                and self._model
                and self._conversation
                and hasattr(self._conversation, "_session_id")
            ):
                session_id = self._conversation._session_id
                if session_id:
                    self._session_storage.save_session(
                        self._model.model_id, session_id, self._message_history
                    )

        except asyncio.CancelledError:
            assistant_widget.mark_complete()
            assistant_widget.update_content("*Cancelled*")
            self._set_status("Cancelled")
        except Exception as e:
            assistant_widget.mark_complete()
            error_msg = f"Error: {e}"
            assistant_widget.update_error(str(e))
            self._set_status(error_msg)
            log.exception(f"Error in _send: {e}")
        finally:
            self._is_responding = False

    def action_clear(self) -> None:
        """Clear the chat and start a new conversation."""
        # Clear UI
        container = self.query_one("#chat-messages", ScrollableContainer)
        container.remove_children()
        # Reset conversation in adapter
        if self._conversation:
            self._conversation.clear()
        # Clear message history
        self._message_history.clear()
        # Clear stored session for ACP adapter
        if self._session_storage and self._model:
            self._session_storage.clear_session(self._model.model_id)
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

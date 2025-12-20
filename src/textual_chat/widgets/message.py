"""Message widget for chat display."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Markdown, Static
from textual_golden import Golden

# JSON type for tool arguments
JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]


@dataclass
class ToolUse:
    """Represents a tool use/call."""

    name: str
    args: dict[str, JSON]
    cwd: str | None = None

    def __str__(self) -> str:
        if not self.args:
            # No arguments - just show tool name
            return f"{self.name}()"

        # Format arguments concisely
        # Use agent's cwd if provided, otherwise fall back to process cwd
        agent_cwd = self.cwd or os.getcwd()
        parts = []
        for k, v in sorted(self.args.items(), key=lambda x: x[0]):
            # Truncate long values
            v_str = repr(v)
            # Replace agent's cwd with ./ for display purposes
            if agent_cwd in v_str:
                v_str = v_str.replace(agent_cwd, ".")
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
            parts.append(f"{k}={v_str}")

        args_str = ", ".join(parts)
        return f"{self.name}({args_str})"


def _humanize_tokens(n: int) -> str:
    """Humanize token count: 3000 -> 3k, 1500 -> 1.5k."""
    if n < 1000:
        return str(n)
    k = n / 1000
    if k >= 10:
        return f"{int(k)}k"
    formatted = f"{k:.1f}".rstrip("0").rstrip(".")
    return f"{formatted}k"


class ToolUseWidget(Static):
    """Widget showing a tool call with unfilled circle (running) or filled circle (complete)."""

    DEFAULT_CSS = """
    ToolUseWidget {
        height: auto;
        color: $text-muted;
    }
    """

    completed: reactive[bool] = reactive(False)

    def __init__(self, tool_use: ToolUse, tool_call_id: str) -> None:
        # Render directly as Static content (no nested children)
        super().__init__(f"○ {tool_use}", markup=False)
        self.tool_use = tool_use
        self.tool_call_id = tool_call_id

    def watch_completed(self, completed: bool) -> None:
        """Update display when completed changes."""
        if completed:
            # Replace with filled circle
            self.update(f"● {self.tool_use}")
            self.styles.color = "green"

    def complete(self) -> None:
        """Mark this tool call as complete."""
        self.completed = True


class MessageWidget(Widget):
    """A chat message widget that builds up content incrementally.

    Content is appended in order:
    - Markdown for text content
    - ToolUseWidget for tool calls
    - More Markdown after tools complete
    """

    DEFAULT_CSS = """
    MessageWidget {
        height: auto;
    }
    MessageWidget #message-content {
        height: auto;
    }
    """

    def __init__(
        self,
        role: str,
        content: str = "",
        loading: bool = False,
        title: str | None = None,
    ) -> None:
        super().__init__(classes=f"message {role}")
        self.role = role
        self._initial_content = content
        self._loading = loading
        self.border_title = title or role.title()
        self._tool_widgets: dict[str, ToolUseWidget] = {}  # tool_call_id -> widget
        self._current_markdown: Markdown | None = None
        self._stream: object | None = None  # Markdown.Stream type not in public API
        self._after_tooluse: bool = False  # Track if next markdown should have margin-top

    def compose(self) -> ComposeResult:
        with Vertical(id="message-content"):
            if self._loading:
                yield Golden("Responding...", id="loading-indicator")
            elif self._initial_content:
                yield Markdown(self._initial_content, classes="content")

    def _get_content_container(self) -> Vertical:
        """Get the content container."""
        return self.query_one("#message-content", Vertical)

    def _scroll_parent(self) -> None:
        """Scroll parent container to show this message."""
        try:
            if isinstance(self.parent, ScrollableContainer):
                self.parent.scroll_end(animate=False)
        except Exception:
            pass

    def _get_loading_indicator(self) -> Golden | None:
        """Get the loading indicator if present."""
        try:
            return self.query_one("#loading-indicator", Golden)
        except Exception:
            return None

    def _remove_loading(self) -> None:
        """Remove loading indicator if present."""
        loading = self._get_loading_indicator()
        if loading:
            loading.remove()

    async def update_content(self, content: str) -> None:
        """Replace all content with new markdown."""
        # Stop any active stream first
        if self._stream is not None:
            await self._stream.stop()  # type: ignore[attr-defined]
            self._stream = None

        self._loading = False
        self._remove_loading()

        container = self._get_content_container()
        # Remove all existing content
        for child in list(container.children):
            child.remove()

        # Add new markdown
        md = Markdown(content, classes="content")
        container.mount(md)
        self._current_markdown = md
        self.call_after_refresh(self._scroll_parent)

    def get_markdown_stream(self) -> object:
        """Get a streaming interface for efficient markdown updates (returns Markdown.Stream)."""
        container = self._get_content_container()
        loading = self._get_loading_indicator()

        # Create new Markdown widget for streaming
        md = Markdown(classes="content")
        # Mount before loading indicator to keep it at bottom
        container.mount(md, before=loading)
        self._current_markdown = md
        self._stream = Markdown.get_stream(md)

        self.call_after_refresh(self._scroll_parent)
        return self._stream

    async def ensure_stream(self) -> object:
        """Get current stream, or create a new one if needed (returns Markdown.Stream)."""
        if self._stream is None:
            # Add blank line separator after tools, before new text
            if self._after_tooluse:
                container = self._get_content_container()
                loading = self._get_loading_indicator()
                # Mount a blank static widget as separator
                separator = Static("", classes="tool-separator")
                separator.styles.height = 1
                container.mount(separator, before=loading)
                self._after_tooluse = False

            return self.get_markdown_stream()
        return self._stream

    async def add_tooluse(self, tu: ToolUse, tool_call_id: str) -> None:
        """Add a tool use widget (shows unfilled circle)."""
        # If we were streaming, stop it properly before discarding
        if self._stream is not None:
            await self._stream.stop()  # type: ignore[attr-defined]
            self._stream = None

        # Text before tools is now "done"
        self._current_markdown = None
        # Reset flag - consecutive tools don't need separator
        self._after_tooluse = False

        container = self._get_content_container()
        loading = self._get_loading_indicator()
        widget = ToolUseWidget(tu, tool_call_id)
        # Mount before loading indicator to keep it at bottom
        container.mount(widget, before=loading)
        self._tool_widgets[tool_call_id] = widget
        self.call_after_refresh(self._scroll_parent)

    def complete_tooluse(self, tool_call_id: str) -> None:
        """Mark a tool as complete (shows filled circle)."""
        if tool_call_id in self._tool_widgets:
            self._tool_widgets[tool_call_id].complete()
        # Set flag so next text gets a separator after this tool
        self._after_tooluse = True

    def mark_complete(self) -> None:
        """Mark the message as complete (removes loading indicator)."""
        self._remove_loading()

    def set_token_usage(self, prompt: int, completion: int, cached: int = 0) -> None:
        """Set token usage in border subtitle."""
        subtitle = f"↑{_humanize_tokens(prompt)} ↓{_humanize_tokens(completion)}"
        if cached:
            subtitle += f" ⚡{_humanize_tokens(cached)}"
        self.border_subtitle = subtitle

    def show_thinking_animated(self, thinking_text: str) -> None:
        """Show animated thinking indicator."""
        self._loading = False
        self._remove_loading()

        container = self._get_content_container()
        text = Static(f"Thinking: {thinking_text}", classes="content")
        text.styles.text_style = "italic"
        container.mount(text)
        self.call_after_refresh(self._scroll_parent)

    async def update_error(self, error: str) -> None:
        """Show error message in red."""
        # Stop any active stream first
        if self._stream is not None:
            await self._stream.stop()  # type: ignore[attr-defined]
            self._stream = None

        self._loading = False
        self._remove_loading()

        container = self._get_content_container()
        # Remove all existing content
        for child in list(container.children):
            child.remove()

        container.mount(Static(f"[red]Error: {error}[/red]", classes="content"))
        self.call_after_refresh(self._scroll_parent)

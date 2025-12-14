"""Message widget for chat display."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Markdown, Static
from textual_golden import BLUE, Golden


@dataclass
class ToolUse:
    """Represents a tool use/call."""

    name: str
    args: dict[str, Any]

    def __str__(self) -> str:
        args_str = ", ".join(
            f"{k}={v!r}" for k, v in sorted(self.args.items(), key=lambda x: x[0])
        )
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
    """Widget showing a tool call with spinner (running) or tick (complete)."""

    DEFAULT_CSS = """
    ToolUseWidget {
        height: auto;
        color: $text-muted;
    }
    ToolUseWidget Horizontal {
        height: auto;
    }
    """

    completed: reactive[bool] = reactive(False)

    def __init__(self, tool_use: ToolUse, tool_call_id: str) -> None:
        super().__init__()
        self.tool_use = tool_use
        self.tool_call_id = tool_call_id

    def compose(self) -> ComposeResult:
        # Spinner for running state
        spinner = Golden("", colors=BLUE, id="spinner")
        spinner.styles.width = 2
        spinner.styles.height = "auto"
        # Tool text
        text = Static(f" {self.tool_use}", markup=False, id="tool-text")
        text.styles.width = "1fr"
        text.styles.height = "auto"
        yield Horizontal(spinner, text)

    def watch_completed(self, completed: bool) -> None:
        """Update display when completed changes."""
        if completed:
            # Replace spinner with tick
            try:
                spinner = self.query_one("#spinner", Golden)
                spinner.remove()
                text = self.query_one("#tool-text", Static)
                text.update(f"✓ {self.tool_use}")
                text.styles.color = "green"
            except Exception:
                pass

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
        self._stream: Markdown.Stream | None = None
        self._after_tooluse: bool = (
            False  # Track if next markdown should have margin-top
        )

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
            if self.parent:
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

    def update_content(self, content: str) -> None:
        """Replace all content with new markdown."""
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

    def get_markdown_stream(self) -> Markdown.Stream:
        """Get a streaming interface for efficient markdown updates."""
        container = self._get_content_container()
        loading = self._get_loading_indicator()

        # Create new Markdown widget for streaming
        md = Markdown(classes="content")
        # Special case: add margin-top when markdown follows tool use
        if self._after_tooluse:
            md.styles.margin = (1, 0, 0, 0)
            self._after_tooluse = False
        # Mount before loading indicator to keep it at bottom
        container.mount(md, before=loading)
        self._current_markdown = md
        self._stream = Markdown.get_stream(md)

        self.call_after_refresh(self._scroll_parent)
        return self._stream

    def ensure_stream(self) -> Markdown.Stream:
        """Get current stream, or create a new one if needed (e.g., after tool use)."""
        if self._stream is None:
            return self.get_markdown_stream()
        return self._stream

    def add_tooluse(self, tu: ToolUse, tool_call_id: str) -> None:
        """Add a tool use widget (shows spinner)."""
        # If we were streaming, that markdown is now "done"
        # (text before tools)
        self._current_markdown = None
        self._stream = None
        self._after_tooluse = True  # Next markdown gets margin-top

        container = self._get_content_container()
        loading = self._get_loading_indicator()
        widget = ToolUseWidget(tu, tool_call_id)
        # Mount before loading indicator to keep it at bottom
        container.mount(widget, before=loading)
        self._tool_widgets[tool_call_id] = widget
        self.call_after_refresh(self._scroll_parent)

    def complete_tooluse(self, tool_call_id: str) -> None:
        """Mark a tool as complete (shows tick)."""
        if tool_call_id in self._tool_widgets:
            self._tool_widgets[tool_call_id].complete()

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

    def update_error(self, error: str) -> None:
        """Show error message in red."""
        self._loading = False
        self._remove_loading()

        container = self._get_content_container()
        # Remove all existing content
        for child in list(container.children):
            child.remove()

        container.mount(Static(f"[red]Error: {error}[/red]", classes="content"))
        self.call_after_refresh(self._scroll_parent)

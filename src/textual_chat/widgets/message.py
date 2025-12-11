"""Message widget for chat display."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Markdown, Static

from textual_golden import Golden, BLUE


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

    def to_widget(self) -> Widget:
        text = Static(str(self), markup=False)
        text.styles.width = "1fr"
        text.styles.height = "auto"
        text.styles.margin = (0, 0, 0, 0)
        return text


def _humanize_tokens(n: int) -> str:
    """Humanize token count: 3000 -> 3k, 1500 -> 1.5k."""
    if n < 1000:
        return str(n)
    k = n / 1000
    if k >= 10:
        return f"{int(k)}k"
    # Show one decimal, strip trailing zero
    formatted = f"{k:.1f}".rstrip("0").rstrip(".")
    return f"{formatted}k"


class MessageWidget(Static):
    """A chat message widget with loading states and tool display."""

    def __init__(
        self, role: str, content: str = "", loading: bool = False, title: str | None = None
    ) -> None:
        super().__init__(classes=f"message {role}")
        self.role = role
        self._content = content
        self._loading = loading
        self.border_title = title or role.title()
        self._active_tools: dict[str, tuple[ToolUse, float]] = {}
        self._tool_lock = asyncio.Lock()

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
        """Update the message content."""
        if not content:
            return
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

    def get_markdown_stream(self) -> Markdown.Stream:
        """Get a streaming interface for efficient markdown updates."""
        self._loading = False
        # Remove loading indicator
        for widget in list(self.children):
            if isinstance(widget, Golden):
                widget.remove()
        # Mount Markdown and return its stream
        md = Markdown(classes="content")
        self.mount(md)
        self.call_after_refresh(self._scroll_parent)
        return Markdown.get_stream(md)

    def set_token_usage(self, prompt: int, completion: int, cached: int = 0) -> None:
        """Set token usage in border subtitle."""
        subtitle = f"↑{_humanize_tokens(prompt)} ↓{_humanize_tokens(completion)}"
        if cached:
            subtitle += f" ⚡{_humanize_tokens(cached)}"
        self.border_subtitle = subtitle

    async def add_tooluse(self, tu: ToolUse, tool_call_id: str) -> None:
        """Track tool as running and update display."""
        import time
        async with self._tool_lock:
            self._active_tools[tool_call_id] = (tu, time.time())
            self._update_tool_display()

    async def remove_tooluse(self, tool_call_id: str | None) -> None:
        """Mark tool as complete and update display after minimum display time."""
        import time
        if not tool_call_id:
            return
        async with self._tool_lock:
            if tool_call_id not in self._active_tools:
                return
            _, start_time = self._active_tools[tool_call_id]
            elapsed = time.time() - start_time
            min_display = 0.3  # Show for at least 300ms
            if elapsed < min_display:
                # Schedule delayed removal
                self.set_timer(min_display - elapsed, lambda: self._do_remove_tool(tool_call_id))
            else:
                del self._active_tools[tool_call_id]
                self._update_tool_display()

    def _do_remove_tool(self, tool_call_id: str) -> None:
        """Actually remove the tool from tracking (called from timer)."""
        if tool_call_id in self._active_tools:
            del self._active_tools[tool_call_id]
            self._update_tool_display()

    def _update_tool_display(self) -> None:
        """Update the tool indicator display based on active tools."""
        # Remove any existing tool indicator
        for widget in list(self.children):
            if "tool-indicator" in widget.classes:
                widget.remove()

        if not self._active_tools:
            # No active tools - restore loading if needed
            if self._loading:
                has_loading = any(isinstance(w, Golden) for w in self.children)
                if not has_loading:
                    self.mount(Golden("Responding...", classes="content"))
            return

        # Remove loading indicator if present
        for widget in list(self.children):
            if isinstance(widget, Golden):
                widget.remove()

        # Show first active tool
        tool_call_id, (tu, _) = next(iter(self._active_tools.items()))

        label = Golden("Using ", colors=BLUE)
        label.styles.width = "auto"
        label.styles.height = "auto"
        label.styles.margin = (0, 0, 0, 0)

        container = Horizontal(
            label, tu.to_widget(), classes="content tool-indicator"
        )
        container.styles.height = "auto"
        container.styles.width = "100%"
        container.styles.margin = (0, 0, 0, 0)
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

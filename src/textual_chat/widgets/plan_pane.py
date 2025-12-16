"""Plan pane widget for displaying agent planning/reasoning."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, VerticalScroll
from textual.widgets import Markdown, Static


class PlanPane(VerticalScroll):
    """Displays agent planning and reasoning updates in real-time."""

    DEFAULT_CSS = """
    PlanPane {
        width: 35%;
        height: 100%;
        border-left: solid $primary-darken-2;
        padding: 1;
        display: none;  /* Hidden by default */
    }
    
    PlanPane.visible {
        display: block;
    }
    
    PlanPane #plan-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    PlanPane #plan-content {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the plan pane.

        Args:
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self._content = ""

    def compose(self) -> ComposeResult:
        """Compose the plan pane."""
        yield Static("Agent Plan", id="plan-title")
        yield Markdown("", id="plan-content")

    def clear(self) -> None:
        """Clear the plan content."""
        self._content = ""
        try:
            content_widget = self.query_one("#plan-content", Markdown)
            content_widget.update("")
        except Exception:
            pass

    async def append_text(self, text: str) -> None:
        """Append text to the plan content.

        Args:
            text: Text to append
        """
        self._content += text
        try:
            content_widget = self.query_one("#plan-content", Markdown)
            await content_widget.update(self._content)
            # Auto-scroll to bottom
            self.scroll_end(animate=False)
        except Exception:
            pass

    def show(self) -> None:
        """Show the plan pane."""
        self.add_class("visible")

    def hide(self) -> None:
        """Hide the plan pane."""
        self.remove_class("visible")

    def toggle(self) -> None:
        """Toggle visibility of the plan pane."""
        self.toggle_class("visible")

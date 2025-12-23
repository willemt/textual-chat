"""Chat input widget with Enter to submit, Shift+Enter for newlines."""

from __future__ import annotations

from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import TextArea


class ChatInput(TextArea):
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
        title: str | None = None,
        subtitle: str | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.show_line_numbers = False
        self._placeholder = placeholder
        self._title = title
        self._subtitle = subtitle

    def on_mount(self) -> None:
        """Set placeholder and border title after mount."""
        self.placeholder = self._placeholder
        if self._title:
            self.border_title = self._title
        if self._subtitle:
            self.border_subtitle = self._subtitle

    async def _on_key(self, event: object) -> None:
        """Handle key presses."""
        # Shift+Enter (comes through as ctrl+j) - insert newline
        if event.key == "ctrl+j":  # type: ignore[attr-defined]
            self.insert("\n")
            event.prevent_default()  # type: ignore[attr-defined]
            event.stop()  # type: ignore[attr-defined]
            return

        # Enter - submit message
        if event.key in ("enter", "ctrl+m"):  # type: ignore[attr-defined]
            content = self.text.strip()
            if content:
                self.post_message(self.Submitted(content))
                self.clear()
            event.prevent_default()  # type: ignore[attr-defined]
            event.stop()  # type: ignore[attr-defined]
            return

        # Page Up/Down - scroll chat messages
        if event.key in ("pageup", "pagedown"):  # type: ignore[attr-defined]
            try:
                container = self.app.query_one("#chat-messages")
                if event.key == "pageup":  # type: ignore[attr-defined]
                    container.scroll_page_up()
                else:
                    container.scroll_page_down()
                event.prevent_default()  # type: ignore[attr-defined]
                event.stop()  # type: ignore[attr-defined]
                return
            except NoMatches:
                pass

        # Let TextArea handle everything else
        await super()._on_key(event)  # type: ignore[arg-type]

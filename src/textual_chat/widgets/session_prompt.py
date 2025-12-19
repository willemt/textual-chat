"""Session resumption prompt for input area."""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, Static


class SessionPromptInput(Widget):
    """Widget showing session prompt with Yes/No buttons in the input area."""

    # Make widget focusable so it can receive key events
    can_focus = True

    DEFAULT_CSS = """
    SessionPromptInput {
        height: auto;
        width: 100%;
        border: round $accent;
        background: $surface;
        padding: 1;
    }

    SessionPromptInput Vertical {
        height: auto;
        width: 100%;
    }

    SessionPromptInput .hotkey-hint {
        color: $text-muted;
        text-style: italic;
        padding: 0 0 1 0;
    }

    SessionPromptInput Horizontal {
        height: auto;
        width: 100%;
        align: left middle;
    }

    SessionPromptInput Label {
        width: auto;
        height: auto;
        padding: 0 1 0 0;
        content-align: left middle;
    }

    SessionPromptInput Button {
        margin-left: 1;
        min-width: 10;
        height: 3;
    }
    """

    class SessionChoice(Message):
        """Posted when user chooses yes or no."""

        def __init__(self, resume: bool) -> None:
            super().__init__()
            self.resume = resume

    def on_mount(self) -> None:
        """Focus the widget when mounted so it can receive key events."""
        self.focus()

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("Previous session found, use?", id="prompt-text")
            yield Button("[1] Yes", variant="success", id="btn-yes", flat=True, compact=True)
            yield Button("[2] No", variant="error", id="btn-no", flat=True, compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "btn-yes":
            self.post_message(self.SessionChoice(resume=True))
        elif event.button.id == "btn-no":
            self.post_message(self.SessionChoice(resume=False))

    def on_key(self, event: events.Key) -> None:
        """Handle key presses for hotkeys 1, 2."""
        if event.key == "1":
            self.post_message(self.SessionChoice(resume=True))
            event.prevent_default()
        elif event.key == "2":
            self.post_message(self.SessionChoice(resume=False))
            event.prevent_default()

"""Session resumption prompt for input area."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label


class SessionPromptInput(Widget):
    """Widget showing session prompt with Yes/No buttons in the input area."""

    DEFAULT_CSS = """
    SessionPromptInput {
        height: auto;
        min-height: 3;
        width: 100%;
        border: round $surface-lighten-1;
        background: transparent;
        padding: 1 0;
    }

    SessionPromptInput Horizontal {
        height: auto;
        width: 100%;
    }

    SessionPromptInput #prompt-text {

        content-align-vertical: middle;
    }

    SessionPromptInput Button {
        margin: 0 1;
        min-width: 8;
    }

    Label {
        width: 100%;
        height: 1fr;
        padding: 1;
        color: white;
    }
    """

    class SessionChoice(Message):
        """Posted when user chooses yes or no."""

        def __init__(self, resume: bool) -> None:
            super().__init__()
            self.resume = resume

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("Previous session found, use?", id="prompt-text")
            yield Button("Yes", variant="success", id="btn-yes", flat=True, compact=True)
            yield Button("No", variant="error", id="btn-no", flat=True, compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "btn-yes":
            self.post_message(self.SessionChoice(resume=True))
        elif event.button.id == "btn-no":
            self.post_message(self.SessionChoice(resume=False))

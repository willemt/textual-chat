"""Chat as a floating modal, like web chatbot widgets.

Press F1 to toggle the chat panel in the bottom-right corner.
The main content remains visible behind the chat.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Footer, Static

from textual_chat import Chat


class ChatModal(ModalScreen):
    """A modal screen containing the chat widget."""

    CSS = """
    ChatModal {
        align: right bottom;
        padding: 0 1 1 0;
    }

    #chat-modal-container {
        width: 80;
        height: 70%;
        background: $panel;
        border: round $primary;
    }

    #chat-modal-container Chat {
        width: 100%;
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("f1", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            Chat(system="You are a helpful assistant embedded in an app."),
            id="chat-modal-container",
        )


class ChatbotApp(App):
    """App with a toggleable chatbot modal in the bottom-right corner."""

    CSS = """
    #main-content {
        width: 100%;
        height: 100%;
        padding: 2;
    }
    """

    BINDINGS = [
        Binding("f1", "toggle_chat", "Toggle Chat"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            Static(
                "[bold]My Application[/bold]\n\n"
                "This demonstrates a chatbot modal pattern,\n"
                "similar to web chat widgets.\n\n"
                "[dim]Press F1 to toggle the AI assistant.[/dim]"
            ),
            id="main-content",
        )
        yield Footer()

    def action_toggle_chat(self) -> None:
        """Toggle the chat modal."""
        self.push_screen(ChatModal())


if __name__ == "__main__":
    ChatbotApp().run()

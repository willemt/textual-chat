"""Chat as a toggleable sidebar.

Press F1 to toggle the chat panel. Main content remains visible.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Static

from textual_chat import Chat


class ChatbotApp(App):
    """App with a toggleable chatbot overlay in the bottom-right corner."""

    CSS = """
    #main-content {
        width: 1fr;
        height: 100%;
        padding: 2;
    }

    #chat-sidebar {
        width: 100;
        height: 100%;
        dock: right;
        background: $panel;
        border-left: solid $primary;
    }

    #chat-sidebar Chat {
        width: 100%;
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("f1", "toggle_chat", "Toggle Chat"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        # Main application content
        yield Container(
            Static(
                "[bold]My Application[/bold]\n\n"
                "This demonstrates a chat sidebar pattern.\n"
                "The main content stays visible.\n\n"
                "[dim]Press F1 to toggle the AI assistant.[/dim]"
            ),
            id="main-content",
        )

        # Chat sidebar (docked right)
        yield Container(
            Chat(system="You are a helpful assistant embedded in an app."),
            id="chat-sidebar",
        )

        yield Footer()

    def on_mount(self) -> None:
        """Hide chat sidebar initially."""
        self.query_one("#chat-sidebar").display = False

    def action_toggle_chat(self) -> None:
        """Toggle the chat sidebar visibility."""
        sidebar = self.query_one("#chat-sidebar")
        sidebar.display = not sidebar.display

        # Focus the chat input when shown
        if sidebar.display:
            self.query_one(Chat).query_one("#chat-input").focus()


if __name__ == "__main__":
    ChatbotApp().run()

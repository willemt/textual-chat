"""Add chat to an existing app as a sidebar."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from textual_chat import Chat

# Chat with app-specific context
chat = Chat(system="You help users navigate this application.")


@chat.tool
def get_status() -> str:
    """Get the current application status."""
    return "All systems operational. 3 pending tasks."


@chat.tool
def list_tasks() -> str:
    """List the user's tasks."""
    return "1. Review PR #42\n2. Fix login bug\n3. Update documentation"


class MyApp(App):
    CSS = """
    #content { width: 1fr; padding: 2; }
    #sidebar { width: 50; border-left: solid $primary; }
    """

    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Toggle Chat"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="content"):
                yield Static(
                    "[bold]My App[/bold]\n\nYour content here.\n\nPress Ctrl+B to toggle the AI assistant."
                )
            yield chat
        yield Footer()

    def action_toggle_sidebar(self) -> None:
        self.query_one(Chat).display = not self.query_one(Chat).display


if __name__ == "__main__":
    MyApp().run()

"""Chat with extended thinking enabled (Claude models)."""

from textual.app import App, ComposeResult
from textual.widgets import Footer

from textual_chat import Chat

# Enable extended thinking - the model will show its reasoning
chat = Chat(
    model="claude-sonnet-4-20250514",  # Extended thinking works with Claude
    thinking=True,  # Enable extended thinking (default 10k token budget)
    # thinking=50000,  # Or specify a custom budget
)


class ThinkingApp(App):
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield chat
        yield Footer()


if __name__ == "__main__":
    ThinkingApp().run()

#!/usr/bin/env python3
"""Demo showing context window usage tracking in the chat input border subtitle.

The progress bar shows how much of the model's context window has been used.
"""

from textual.app import App, ComposeResult

from textual_chat import Chat


class ContextWindowDemo(App):
    """Demo app showing context window usage display."""

    CSS = """
    Screen {
        background: $surface;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        # show_token_usage shows per-message token counts
        # The context window usage is always shown in the border subtitle
        yield Chat(show_token_usage=True)


if __name__ == "__main__":
    ContextWindowDemo().run()

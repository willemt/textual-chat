"""LLM chat for humans. Add AI to your Textual app in 6 lines of code.

    from textual.app import App, ComposeResult
    from textual_chat import Chat

    class MyApp(App):
        def compose(self) -> ComposeResult:
            yield Chat()

    MyApp().run()

That's it. See https://github.com/your/textual-chat for more.
"""

# The only import most people need
from .chat import Chat

# Advanced components (most users won't need these)
from .db import ChatDatabase, Conversation, Message

# Re-export GoldenWave for convenience
from textual_golden import GoldenWave

__version__ = "0.1.0"
__all__ = ["Chat", "ChatDatabase", "Conversation", "Message", "GoldenWave"]

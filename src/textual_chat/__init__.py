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
# Re-export Golden for convenience
from textual_golden import Golden

from .agent_manager import get_agent_manager
from .chat import INLINE, SEPARATE, Chat, get_async_model
from .session_storage import get_session_storage
from .slash_command import SlashCommand, SlashCommandManager
from .widgets import MessageWidget, ToolUse

__version__ = "0.1.0"
__all__ = [
    "Chat",
    "INLINE",
    "SEPARATE",
    "MessageWidget",
    "SlashCommand",
    "SlashCommandManager",
    "ToolUse",
    "Golden",
    "get_async_model",
    "get_agent_manager",
    "get_session_storage",
]

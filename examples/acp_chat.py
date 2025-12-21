#!/usr/bin/env python
"""TUI chat with an ACP agent.

Run with:
    uv run python examples/acp_chat.py examples/tool_agent.py
"""

import sys
from pathlib import Path

from textual.app import App, ComposeResult

from textual_chat import Chat


class ACPChatApp(App):
    """Chat app that talks to an ACP agent."""

    CSS = """
    Screen {
        background: $surface;
    }
    """

    def __init__(self, agent_path: str):
        super().__init__()
        self.agent_path = agent_path

    def compose(self) -> ComposeResult:
        yield Chat(
            model=self.agent_path,
            adapter="acp",  # Use ACP adapter
            system="You are a helpful assistant.",
            show_token_usage=False,  # ACP doesn't provide token counts
            show_model_selector=False,  # Not applicable for ACP
        )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/acp_chat.py <agent_script.py>")
        print("\nExamples:")
        print("  uv run python examples/acp_chat.py examples/echo_agent.py")
        print("  uv run python examples/acp_chat.py examples/tool_agent.py")
        sys.exit(1)

    agent_path = sys.argv[1]
    if not Path(agent_path).exists():
        print(f"Error: Agent script not found: {agent_path}")
        sys.exit(1)

    app = ACPChatApp(agent_path)
    app.run()


if __name__ == "__main__":
    main()

"""Test script for message interruption functionality.

This tests that users can interrupt the agent with new messages while it's working,
and the agent will receive context about what it was doing when interrupted.
"""

from textual.app import App, ComposeResult
from textual_chat import Chat
import time


class TestInterruptApp(App):
    def compose(self) -> ComposeResult:
        # Use ACP adapter for testing interrupt functionality
        chat = Chat(
            adapter="acp",
            model="claude-code",  # Or any available ACP agent
        )

        # Add a slow tool to simulate long-running agent work
        @chat.tool
        def slow_task(duration: int = 5) -> str:
            """A slow task that takes some time to complete.

            Args:
                duration: How many seconds to wait (default 5)
            """
            time.sleep(duration)
            return f"Task completed after {duration} seconds"

        yield chat


if __name__ == "__main__":
    print("=" * 60)
    print("MESSAGE INTERRUPT TEST (ACP)")
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. Start the app")
    print("2. Send a message that triggers a slow operation")
    print("   Example: 'Run the slow_task tool for 10 seconds'")
    print("3. While the agent is responding, send a NEW message")
    print("4. The agent should be INTERRUPTED and receive:")
    print("   - Context about the original task")
    print("   - Your new message with priority")
    print()
    print("Expected behavior (ACP adapter):")
    print("✓ New message cancels current agent work")
    print("✓ Agent receives combined prompt with context:")
    print("   [Context: I was working on: 'original task']")
    print("   [INTERRUPTION] User's new message")
    print("✓ Agent responds to new message immediately")
    print("✓ Status shows 'Interrupting agent with new message...'")
    print()
    print("For non-ACP adapters:")
    print("- Falls back to queuing behavior")
    print("- Messages process sequentially after current response")
    print("=" * 60)
    print()

    TestInterruptApp().run()

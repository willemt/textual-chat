"""Tests for message interruption functionality.

This module contains both unit tests and an interactive test app.
The interactive test requires manual execution to verify behavior.
"""

import time

import pytest
from textual.app import App, ComposeResult

from textual_chat import Chat


class TestMessageInterruptConfig:
    """Unit tests for message interrupt configuration."""

    def test_chat_has_interrupt_method(self) -> None:
        """Chat widget has _interrupt_with_message method."""
        chat = Chat()
        assert hasattr(chat, "_interrupt_with_message")

    def test_chat_has_cancel_requested_flag(self) -> None:
        """Chat widget has _cancel_requested flag."""
        chat = Chat()
        assert hasattr(chat, "_cancel_requested")

    def test_chat_has_response_task(self) -> None:
        """Chat widget has _response_task attribute."""
        chat = Chat()
        assert hasattr(chat, "_response_task")


# Interactive test app for manual testing
class InterruptTestApp(App):
    """Interactive app for testing message interruption.

    Run with: python -m pytest tests/test_message_interrupt.py::test_interactive_interrupt -s
    """

    def compose(self) -> ComposeResult:
        chat = Chat(
            adapter="acp",
            model="claude-code",
        )

        @chat.tool
        def slow_task(duration: int = 5) -> str:
            """A slow task that takes some time to complete."""
            time.sleep(duration)
            return f"Task completed after {duration} seconds"

        yield chat


@pytest.mark.skip(
    reason="Interactive test - run manually with: python tests/test_message_interrupt.py"
)
def test_interactive_interrupt() -> None:
    """Interactive test for message interruption.

    To run manually:
        python tests/test_message_interrupt.py

    Instructions:
    1. Start the app
    2. Send a message that triggers a slow operation
       Example: 'Run the slow_task tool for 10 seconds'
    3. While the agent is responding, send a NEW message
    4. The agent should be INTERRUPTED and receive context about the original task
    """
    InterruptTestApp().run()


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
    print("- New message cancels current agent work")
    print("- Agent receives combined prompt with context")
    print("- Agent responds to new message immediately")
    print("- Status shows 'Interrupting agent with new message...'")
    print()
    print("For non-ACP adapters:")
    print("- Falls back to queuing behavior")
    print("- Messages process sequentially after current response")
    print("=" * 60)
    print()

    InterruptTestApp().run()

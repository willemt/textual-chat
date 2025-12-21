"""Tests for message queue functionality.

This module contains both unit tests and an interactive test app.
The interactive test requires manual execution to verify behavior.
"""

import time
from collections import deque

import pytest
from textual.app import App, ComposeResult

from textual_chat import Chat


class TestMessageQueueConfig:
    """Unit tests for message queue configuration."""

    def test_chat_has_pending_messages(self) -> None:
        """Chat widget has _pending_messages queue."""
        chat = Chat()
        assert hasattr(chat, "_pending_messages")

    def test_pending_messages_is_deque(self) -> None:
        """_pending_messages is a deque."""
        chat = Chat()
        assert isinstance(chat._pending_messages, deque)

    def test_chat_has_process_next_queued_message(self) -> None:
        """Chat widget has _process_next_queued_message method."""
        chat = Chat()
        assert hasattr(chat, "_process_next_queued_message")

    def test_chat_has_send_queued(self) -> None:
        """Chat widget has _send_queued method."""
        chat = Chat()
        assert hasattr(chat, "_send_queued")


# Interactive test app for manual testing
class QueueTestApp(App):
    """Interactive app for testing message queue.

    Run with: python tests/test_message_queue.py
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


@pytest.mark.skip(reason="Interactive test - run manually with: python tests/test_message_queue.py")
def test_interactive_queue() -> None:
    """Interactive test for message queue.

    To run manually:
        python tests/test_message_queue.py

    Instructions:
    1. Start the app
    2. Send a message that triggers a slow operation
       Example: 'Run the slow_task tool for 10 seconds'
    3. While the agent is responding, type and send more messages
    4. Messages should appear with yellow/warning border (pending)
    5. After agent completes, queued messages should process
    6. Press Escape to cancel (will also clear the queue)
    """
    QueueTestApp().run()


if __name__ == "__main__":
    print("=" * 60)
    print("MESSAGE QUEUE TEST")
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. Start the app")
    print("2. Send a message that triggers a slow operation")
    print("   Example: 'Run the slow_task tool for 10 seconds'")
    print("3. While the agent is responding, type and send more messages")
    print("4. Messages should appear with yellow/warning border (pending)")
    print("5. After agent completes, queued messages should process")
    print("6. Press Escape to cancel (will also clear the queue)")
    print()
    print("Expected behavior:")
    print("- Messages can be sent while agent is working")
    print("- Pending messages show with warning border (yellow)")
    print("- Status shows 'Message queued (N pending)...'")
    print("- Messages process sequentially after current response")
    print("- Escape cancels current response AND clears queue")
    print("=" * 60)
    print()

    QueueTestApp().run()

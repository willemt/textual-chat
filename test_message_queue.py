"""Test script for message queue functionality.

This tests that users can send multiple messages while the agent is responding.
"""

from textual.app import App, ComposeResult
from textual_chat import Chat
import time


class TestQueueApp(App):
    def compose(self) -> ComposeResult:
        # Use ACP adapter for testing
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

    TestQueueApp().run()

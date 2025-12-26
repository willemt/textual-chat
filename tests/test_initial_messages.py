"""Tests for initial_messages feature."""

import pytest

from textual_chat import Chat


class TestInitialMessagesConfig:
    """Unit tests for initial_messages configuration."""

    def test_initial_messages_accepts_string(self) -> None:
        """initial_messages accepts a single string."""
        chat = Chat(initial_messages="Hello, world!")
        assert chat._initial_messages == ["Hello, world!"]

    def test_initial_messages_accepts_list(self) -> None:
        """initial_messages accepts a list of strings."""
        chat = Chat(initial_messages=["First", "Second", "Third"])
        assert chat._initial_messages == ["First", "Second", "Third"]

    def test_initial_messages_defaults_to_empty(self) -> None:
        """initial_messages defaults to empty list."""
        chat = Chat()
        assert chat._initial_messages == []

    def test_initial_messages_none_becomes_empty(self) -> None:
        """initial_messages=None becomes empty list."""
        chat = Chat(initial_messages=None)
        assert chat._initial_messages == []

    def test_has_send_initial_messages_method(self) -> None:
        """Chat has _send_initial_messages method."""
        chat = Chat()
        assert hasattr(chat, "_send_initial_messages")
        assert callable(chat._send_initial_messages)

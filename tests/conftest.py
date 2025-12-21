"""Pytest configuration and shared fixtures for textual-chat tests."""

import pytest

from textual_chat import Chat


@pytest.fixture
def chat() -> Chat:
    """Create a Chat widget instance."""
    return Chat()

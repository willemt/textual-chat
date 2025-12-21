"""Tests for interrupt hotkey implementation."""

import pytest
from textual.binding import Binding

from textual_chat import Chat


class TestInterruptHotkeys:
    """Tests for Ctrl+C and Escape interrupt bindings."""

    @pytest.fixture
    def chat(self) -> Chat:
        """Create a Chat instance."""
        return Chat()

    @pytest.fixture
    def bindings(self) -> dict[str, Binding]:
        """Get Chat bindings as a dict keyed by key."""
        bindings_dict: dict[str, Binding] = {}
        for binding in Chat.BINDINGS:
            if isinstance(binding, Binding):
                bindings_dict[binding.key] = binding
        return bindings_dict

    def test_ctrl_c_binding_exists(self, bindings: dict[str, Binding]) -> None:
        """Ctrl+C binding is defined."""
        assert "ctrl+c" in bindings

    def test_ctrl_c_calls_cancel_action(self, bindings: dict[str, Binding]) -> None:
        """Ctrl+C binding calls the 'cancel' action."""
        binding = bindings["ctrl+c"]
        assert binding.action == "cancel"

    def test_ctrl_c_is_visible(self, bindings: dict[str, Binding]) -> None:
        """Ctrl+C binding is visible in the UI."""
        binding = bindings["ctrl+c"]
        assert binding.show is True

    def test_ctrl_c_description(self, bindings: dict[str, Binding]) -> None:
        """Ctrl+C binding has 'Interrupt' description."""
        binding = bindings["ctrl+c"]
        assert binding.description == "Interrupt"

    def test_action_cancel_method_exists(self, chat: Chat) -> None:
        """Chat has action_cancel method."""
        assert hasattr(chat, "action_cancel")
        assert callable(chat.action_cancel)

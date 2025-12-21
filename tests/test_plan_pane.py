"""Tests for plan pane implementation."""

from textual_chat.events import PlanChunk, StreamEvent
from textual_chat.widgets import PlanPane
from textual_chat.widgets.message import JSON


class TestPlanChunk:
    """Tests for PlanChunk event."""

    def test_import(self) -> None:
        """PlanChunk can be imported."""
        assert PlanChunk is not None

    def test_is_stream_event(self) -> None:
        """PlanChunk is part of StreamEvent union type."""
        assert PlanChunk in StreamEvent.__args__

    def test_create_plan_chunk(self) -> None:
        """Can create a PlanChunk instance."""
        chunk = PlanChunk(text="Test plan text")
        assert chunk.text == "Test plan text"

    def test_plan_chunk_with_entries(self) -> None:
        """Can create a PlanChunk with entries."""
        entries: list[dict[str, JSON]] = [
            {"content": "Step 1", "status": "completed"},
            {"content": "Step 2", "status": "in_progress"},
        ]
        chunk = PlanChunk(text="", entries=entries)
        assert chunk.entries == entries


class TestPlanPane:
    """Tests for PlanPane widget."""

    def test_import(self) -> None:
        """PlanPane can be imported from widgets."""
        assert PlanPane is not None

    def test_create_plan_pane(self) -> None:
        """Can create a PlanPane widget."""
        pane = PlanPane(id="test-pane")
        assert pane.id == "test-pane"

    def test_plan_pane_has_update_plan(self) -> None:
        """PlanPane has update_plan method."""
        pane = PlanPane()
        assert hasattr(pane, "update_plan")

    def test_plan_pane_has_clear(self) -> None:
        """PlanPane has clear method."""
        pane = PlanPane()
        assert hasattr(pane, "clear")


class TestAcpAdapterIntegration:
    """Tests for ACP adapter plan pane integration."""

    def test_acp_adapter_imports(self) -> None:
        """llm_adapter_acp module imports successfully."""
        from textual_chat import llm_adapter_acp

        assert llm_adapter_acp is not None

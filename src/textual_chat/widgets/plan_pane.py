"""Plan pane widget for displaying agent planning/reasoning."""

from __future__ import annotations

import logging
from typing import Union

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import ListView, ListItem, Label, Static

# JSON type for plan data
JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]

log = logging.getLogger(__name__)


class PlanPane(VerticalScroll):
    """Displays agent planning and reasoning updates in real-time."""

    DEFAULT_CSS = """
    PlanPane {
        width: 28%;
        height: 100%;
        border-left: solid $primary-darken-2;
        padding: 1;
        display: none;  /* Hidden by default */
    }

    PlanPane.visible {
        display: block;
    }

    PlanPane #plan-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    PlanPane ListView {
        height: 1fr;
        background: transparent;
        border: none;
    }

    PlanPane ListItem {
        background: transparent;
        padding: 0 1;
        height: auto;
    }

    PlanPane ListItem.completed {
        color: $success;
    }

    PlanPane ListItem.in_progress {
        color: $warning;
        text-style: bold;
    }

    PlanPane ListItem.pending {
        color: $text-muted;
    }

    PlanPane ListItem Label {
        width: 100%;
        height: auto;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the plan pane.

        Args:
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)

    def compose(self) -> ComposeResult:
        """Compose the plan pane."""
        yield Static("Agent Plan", id="plan-title")
        yield ListView(id="plan-list")

    def clear(self) -> None:
        """Clear the plan content."""
        try:
            list_view = self.query_one("#plan-list", ListView)
            list_view.clear()
        except Exception as e:
            log.error(f"Failed to clear plan pane: {e}", exc_info=True)

    async def update_plan(self, entries: list[dict[str, JSON]]) -> None:
        """Update plan with entries.

        Args:
            entries: List of plan entries with 'content', 'status', 'priority' fields
        """
        try:
            log.info(f"📋 PlanPane.update_plan called with {len(entries)} entries")
            log.info(f"📋 Entries received: {entries}")

            list_view = self.query_one("#plan-list", ListView)
            list_view.clear()
            log.info(f"📋 Cleared list_view")

            for i, entry in enumerate(entries):
                status_val = entry.get("status", "pending")
                content_val = entry.get("content", "")

                # Ensure status and content are strings
                status = str(status_val) if status_val is not None else "pending"
                content = str(content_val) if content_val is not None else ""

                log.info(f"📋 Processing entry {i}: status='{status}', content='{content}'")

                # Create status icon
                if status == "completed":
                    icon = "●"
                elif status == "in_progress":
                    icon = "▶"
                else:  # pending
                    icon = "○"

                # Create label with icon
                label_text = f"{icon} {content}"
                label = Label(label_text)

                # Create list item with status class
                item = ListItem(label)
                item.add_class(status)

                await list_view.append(item)
                log.info(f"📋 Appended item {i} to list_view")

            log.info(
                f"📋 PlanPane.update_plan completed, list_view has {len(list_view.children)} items"
            )

        except Exception as e:
            log.error(f"Failed to update plan pane: {e}", exc_info=True)

    def show(self) -> None:
        """Show the plan pane."""
        self.add_class("visible")

    def hide(self) -> None:
        """Hide the plan pane."""
        self.remove_class("visible")

    def toggle(self) -> None:
        """Toggle visibility of the plan pane."""
        self.toggle_class("visible")

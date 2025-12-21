"""Permission request prompt widget."""

from __future__ import annotations

from typing import Any, Literal, Union

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, Static

# JSON type for permission data
JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]


class PermissionPrompt(Widget):
    """Widget showing permission request with approval options."""

    # Make widget focusable so it can receive key events
    can_focus = True

    DEFAULT_CSS = """
    PermissionPrompt {
        height: auto;
        width: 100%;
        border: round $warning;
        background: $surface;
        padding: 1;
        margin: 1 0;
    }

    PermissionPrompt Vertical {
        height: auto;
        width: 100%;
    }

    PermissionPrompt Label {
        width: 100%;
        height: auto;
        padding: 0 0 1 0;
        color: $warning;
        text-style: bold;
    }

    PermissionPrompt Static {
        width: 100%;
        height: auto;
        padding: 0 0 1 0;
        color: $text;
    }

    PermissionPrompt .hotkey-hint {
        color: $text-muted;
        text-style: italic;
    }

    PermissionPrompt Horizontal {
        height: auto;
        width: 100%;
        align: left middle;
    }

    PermissionPrompt Button {
        margin-left: 1;
        min-width: 15;
        height: 3;
    }
    """

    class PermissionResponse(Message):
        """Posted when user selects an option."""

        def __init__(self, request_id: str, option_id: str) -> None:
            super().__init__()
            self.request_id = request_id
            self.option_id = option_id

    def __init__(
        self,
        request_id: str,
        tool_call: dict[str, JSON],
        options: list[dict[str, JSON]],
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.request_id = request_id
        self.tool_call = tool_call
        self.options = options
        self.border_title = "Permission Request"

    def on_mount(self) -> None:
        """Focus the widget when mounted so it can receive key events."""
        self.focus()

    def compose(self) -> ComposeResult:
        with Vertical():
            # Tool call details
            tool_title = self.tool_call.get("title", "Unknown tool")
            yield Static(f"Tool: {tool_title}")

            # Options as buttons with hotkeys
            with Horizontal():
                # Determine button variants and labels based on common permission patterns
                for idx, option in enumerate(self.options):
                    option_id = str(option.get("option_id", ""))
                    option_name = str(option.get("name", option_id))

                    # Map hotkeys: 1 = always accept, 2 = accept, 3 = reject
                    # Try to intelligently assign based on option names
                    hotkey = str(idx + 1)
                    variant: Literal["default", "primary", "success", "warning", "error"] = (
                        "default"
                    )

                    # Determine button variant based on option name
                    name_lower = option_name.lower()
                    if "always" in name_lower or idx == 0:
                        variant = "primary"
                        hotkey = "1"
                    elif "accept" in name_lower or "allow" in name_lower or "approve" in name_lower:
                        variant = "success"
                        hotkey = "2"
                    elif "reject" in name_lower or "deny" in name_lower or "decline" in name_lower:
                        variant = "error"
                        hotkey = "3"

                    button_label = f"[{hotkey}] {option_name}"

                    yield Button(
                        button_label,
                        variant=variant,
                        id=f"btn-{option_id}",
                        classes="permission-option",
                        flat=True,
                        compact=True,
                    )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.has_class("permission-option"):
            # Extract option_id from button id (btn-{option_id})
            button_id = event.button.id or ""
            if button_id.startswith("btn-"):
                option_id = button_id[4:]  # Remove "btn-" prefix
                self.post_message(self.PermissionResponse(self.request_id, option_id))

    def on_key(self, event: events.Key) -> None:
        """Handle key presses for hotkeys 1, 2, 3."""
        if event.key in ("1", "2", "3"):
            # Map key to option index
            key_index = int(event.key) - 1

            # Find option by matching hotkey logic from compose()
            for idx, option in enumerate(self.options):
                option_id = str(option.get("option_id", ""))
                option_name = str(option.get("name", option_id))
                name_lower = option_name.lower()

                # Determine which hotkey this option should have
                assigned_key = None
                if "always" in name_lower or idx == 0:
                    assigned_key = "1"
                elif "accept" in name_lower or "allow" in name_lower or "approve" in name_lower:
                    assigned_key = "2"
                elif "reject" in name_lower or "deny" in name_lower or "decline" in name_lower:
                    assigned_key = "3"

                # If this option matches the pressed key, select it
                if assigned_key == event.key:
                    self.post_message(self.PermissionResponse(self.request_id, option_id))
                    event.prevent_default()
                    return

            # Fallback: use index-based selection if no match found
            if key_index < len(self.options):
                option_id = str(self.options[key_index].get("option_id", ""))
                self.post_message(self.PermissionResponse(self.request_id, option_id))
                event.prevent_default()

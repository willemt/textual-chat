"""Modal for selecting an ACP agent."""

from __future__ import annotations

import os
import shutil

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets._option_list import Option


class AgentSelectModal(ModalScreen[str | None]):
    """Modal for selecting an ACP agent."""

    DEFAULT_CSS = """
    AgentSelectModal {
        align: center middle;
    }
    AgentSelectModal > Vertical {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    AgentSelectModal #title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }
    AgentSelectModal OptionList {
        height: auto;
        max-height: 20;
    }
    AgentSelectModal #agent-info {
        text-align: center;
        color: $text-muted;
        height: 1;
        margin-top: 1;
    }
    AgentSelectModal #custom-label {
        text-align: left;
        color: $text;
        margin-top: 1;
        margin-bottom: 0;
    }
    AgentSelectModal #custom-agent-input {
        width: 100%;
        height: 3;
        margin-top: 0;
        margin-bottom: 1;
    }
    AgentSelectModal #hint {
        text-align: center;
        color: $text-muted;
        padding-top: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, agents: list[tuple[str, str, str]], current: str | None = None) -> None:
        """Initialize modal.

        Args:
            agents: List of (display_name, agent_command, description) tuples
            current: Currently selected agent_command
        """
        super().__init__()
        self.agents = agents
        self.current = current
        self._agent_info: dict[str, str] = {cmd: desc for _, cmd, desc in agents}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Select Agent", id="title")
            options = [
                Option(f"{'● ' if cmd == self.current else '  '}{name}", id=cmd)
                for name, cmd, _ in self.agents
            ]
            yield OptionList(*options)
            yield Static("", id="agent-info")
            yield Static("Or enter a custom agent command:", id="custom-label")
            yield Input(
                placeholder="e.g., python /path/to/agent.py",
                id="custom-agent-input",
            )
            yield Static(
                "[i]Enter to select, Tab to switch focus, Escape to cancel[/i]",
                id="hint",
            )

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Update info when option is highlighted."""
        agent_cmd = event.option.id
        if agent_cmd and agent_cmd in self._agent_info:
            info = self._agent_info[agent_cmd]
            self.query_one("#agent-info", Static).update(f"[i]{info}[/i]")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle selection from the option list."""
        self.dismiss(event.option.id)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle custom agent input submission."""
        custom_input = self.query_one("#custom-agent-input", Input)
        custom_agent = custom_input.value.strip()

        if not custom_agent:
            self.notify("Please enter an agent command", severity="warning")
            return

        # Basic validation - check if command looks valid
        parts = custom_agent.split()
        if not parts:
            self.notify("Invalid agent command", severity="warning")
            return

        executable = parts[0]

        # Check if executable exists
        if not (os.path.isabs(executable) and os.path.exists(executable)) and not shutil.which(
            executable
        ):
            self.notify(
                f"Warning: '{executable}' not found. Proceeding anyway...",
                severity="warning",
                timeout=5,
            )

        self.dismiss(custom_agent)

    def action_cancel(self) -> None:
        self.dismiss(None)

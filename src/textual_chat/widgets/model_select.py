"""Modal for selecting an LLM model."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets._option_list import Option


class ModelSelectModal(ModalScreen[str | None]):
    """Modal for selecting an LLM model."""

    DEFAULT_CSS = """
    ModelSelectModal {
        align: center middle;
    }
    ModelSelectModal > Vertical {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    ModelSelectModal #title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }
    ModelSelectModal OptionList {
        height: auto;
        max-height: 20;
    }
    ModelSelectModal #model-info {
        text-align: center;
        color: $text-muted;
        height: 1;
        margin-top: 1;
    }
    ModelSelectModal #hint {
        text-align: center;
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, models: list[tuple[str, str, str]], current: str | None = None) -> None:
        """Initialize modal.

        Args:
            models: List of (display_name, model_id, provider) tuples
            current: Currently selected model_id
        """
        super().__init__()
        self.models = models
        self.current = current
        self._model_info: dict[str, str] = {mid: provider for _, mid, provider in models}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Select Model", id="title")
            options = [
                Option(f"{'● ' if mid == self.current else '  '}{name}", id=mid)
                for name, mid, _ in self.models
            ]
            yield OptionList(*options)
            yield Static("", id="model-info")
            yield Static("[i]Enter to select, Escape to cancel[/i]", id="hint")

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Update info when option is highlighted."""
        model_id = event.option.id
        if model_id and model_id in self._model_info:
            info = self._model_info[model_id]
            self.query_one("#model-info", Static).update(f"[i]{info}[/i]")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)

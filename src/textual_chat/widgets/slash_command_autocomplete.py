"""Slash command autocomplete for TextArea widgets."""

from __future__ import annotations

from collections.abc import Sequence

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.geometry import Offset, Region, Spacing
from textual.widget import Widget
from textual.widgets import OptionList, TextArea
from textual.widgets.option_list import Option
from textual_autocomplete.fuzzy_search import FuzzySearch

from ..slash_command import SlashCommand


class SlashCommandAutocomplete(Widget):
    """Autocomplete widget for slash commands in a TextArea."""

    BINDINGS = [
        Binding("escape", "hide", "Hide dropdown", show=False),
    ]

    DEFAULT_CSS = """\
    SlashCommandAutocomplete {
        height: auto;
        width: auto;
        max-height: 12;
        display: none;
        background: $surface;
        overlay: screen;
        border: round $primary;
        padding: 0;

        & > OptionList {
            width: auto;
            height: auto;
            border: none;
            padding: 0;
            margin: 0;
            scrollbar-size-vertical: 1;
            text-wrap: nowrap;
            color: $foreground;
            background: transparent;
        }
    }
    """

    def __init__(
        self,
        target: TextArea | str,
        commands: Sequence[SlashCommand] | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        """Initialize the slash command autocomplete widget.

        Args:
            target: A TextArea instance or a selector string.
            commands: The slash commands to autocomplete.
            name: The widget name.
            id: The widget ID.
            classes: The widget classes.
            disabled: Whether the widget is disabled.
        """
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._target = target
        self.commands = list(commands) if commands else []
        self._fuzzy_search = FuzzySearch()
        self._previous_terminal_cursor_position = (0, 0)

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        option_list = OptionList()
        option_list.can_focus = False
        yield option_list

    def on_mount(self) -> None:
        """Subscribe to target widget events."""
        self.target.message_signal.subscribe(self, self._listen_to_messages)  # type: ignore[arg-type]
        self.watch(self.target, "has_focus", self._handle_focus_change)

        def _realign(_: object = None) -> None:
            if (
                self.is_attached
                and self._previous_terminal_cursor_position != self.app.cursor_position
            ):
                self._align_to_target()
                self._previous_terminal_cursor_position = self.app.cursor_position

        self.screen.screen_layout_refresh_signal.subscribe(self, _realign)

    @property
    def target(self) -> TextArea:
        """Get the target TextArea widget."""
        if isinstance(self._target, TextArea):
            return self._target
        else:
            target = self.screen.query_one(self._target)
            assert isinstance(target, TextArea)
            return target

    @property
    def option_list(self) -> OptionList:
        """Get the option list widget."""
        return self.query_one(OptionList)

    def _listen_to_messages(self, event: events.Event) -> None:
        """Listen to events from the target widget."""
        try:
            option_list = self.option_list
        except NoMatches:
            return

        if isinstance(event, events.Key) and option_list.option_count:
            displayed = self.styles.display != "none"
            highlighted = option_list.highlighted or 0

            if event.key == "down":
                # Show dropdown or move selection
                event.prevent_default()
                event.stop()
                if displayed:
                    highlighted = (highlighted + 1) % option_list.option_count
                else:
                    self.action_show()
                    highlighted = 0
                option_list.highlighted = highlighted

            elif event.key == "up":
                # Move selection up
                if displayed:
                    event.prevent_default()
                    event.stop()
                    highlighted = (highlighted - 1) % option_list.option_count
                    option_list.highlighted = highlighted

            elif event.key == "tab":
                # Complete with tab
                if displayed:
                    event.prevent_default()
                    event.stop()
                    self._complete(option_index=highlighted)

            elif event.key == "escape":
                # Hide dropdown
                if displayed:
                    event.prevent_default()
                    event.stop()
                    self.action_hide()

        if isinstance(event, TextArea.Changed):
            self._handle_target_update()

    def _handle_focus_change(self, has_focus: bool) -> None:
        """Handle focus changes on the target widget."""
        if not has_focus:
            self.action_hide()

    def _handle_target_update(self) -> None:
        """Handle updates to the target widget text."""
        search_string = self._get_search_string()

        if self._should_show_dropdown(search_string):
            self._rebuild_options(search_string)
            self._align_to_target()
            self.action_show()
        else:
            self.action_hide()

    def _get_search_string(self) -> str:
        """Get the current search string from the target widget.

        Returns the text from the start of the line up to the cursor position.
        """
        target = self.target
        cursor_row, cursor_col = target.cursor_location

        # Get the current line text up to the cursor (get_line returns rich.Text)
        current_line = target.get_line(cursor_row).plain
        search_text = current_line[:cursor_col]

        # Only return text after the slash if it starts with /
        if search_text.startswith("/"):
            return search_text[1:]  # Remove the leading /
        return ""

    def _should_show_dropdown(self, search_string: str) -> bool:
        """Determine if the dropdown should be shown."""
        # Only show if we're at the start of a line with a /
        target = self.target
        cursor_row, cursor_col = target.cursor_location
        current_line = target.get_line(cursor_row).plain
        line_up_to_cursor = current_line[:cursor_col]

        # Must start with / and cursor must be on that same word
        if not line_up_to_cursor.startswith("/"):
            return False

        # Don't show if there's a space after the /
        if " " in line_up_to_cursor:
            return False

        return True

    def _rebuild_options(self, search_string: str) -> None:
        """Rebuild the dropdown options based on the search string."""
        option_list = self.option_list
        option_list.clear_options()

        if not self.target.has_focus:
            return

        matches = self._get_matches(search_string)
        if matches:
            option_list.add_options(matches)
            option_list.highlighted = 0

    def _get_matches(self, search_string: str) -> list[Option]:
        """Get matching commands based on the search string."""
        if not search_string:
            # Show all commands if no search string
            return [Option(f"/{cmd.name} - {cmd.description}") for cmd in self.commands]

        # Fuzzy search through commands
        matches_and_scores: list[tuple[SlashCommand, float]] = []
        for cmd in self.commands:
            score, offsets = self._fuzzy_search.match(search_string, cmd.name)
            if score > 0:
                matches_and_scores.append((cmd, score))

        # Sort by score
        matches_and_scores.sort(key=lambda x: x[1], reverse=True)

        # Create options
        return [Option(f"/{cmd.name} - {cmd.description}") for cmd, _ in matches_and_scores]

    def _complete(self, option_index: int) -> None:
        """Complete the command at the given option index."""
        if self.styles.display == "none" or self.option_list.option_count == 0:
            return

        option = self.option_list.get_option_at_index(option_index)
        # Extract command name from the option (format is "/name - description")
        if hasattr(option.prompt, "plain"):
            command_text = option.prompt.plain
        else:
            command_text = str(option.prompt)
        command_name = command_text.split(" - ")[0]  # Gets "/name"

        # Replace the current /... text with the completed command
        target = self.target
        cursor_row, cursor_col = target.cursor_location
        current_line = target.get_line(cursor_row).plain

        # Find where the / starts
        slash_start = current_line.find("/")
        if slash_start == -1:
            return

        # Delete from slash to cursor
        target.delete((cursor_row, slash_start), (cursor_row, cursor_col))

        # Insert the completed command
        target.insert(command_name)

        self.action_hide()

    def _align_to_target(self) -> None:
        """Align the dropdown to the cursor position in the target widget."""
        x, y = self.target.cursor_screen_offset
        dropdown = self.option_list
        width, height = dropdown.outer_size

        # Position dropdown above the cursor, offset by 2 to clear the message box border
        x, y, _width, _height = Region(x - 1, y - height - 2, width, height).constrain(
            "inside",
            "none",
            Spacing.all(0),
            self.screen.scrollable_content_region,
        )
        self.absolute_offset = Offset(x, y)

    def action_hide(self) -> None:
        """Hide the dropdown."""
        self.styles.display = "none"

    def action_show(self) -> None:
        """Show the dropdown."""
        self.styles.display = "block"

    @on(OptionList.OptionSelected)
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection (click)."""
        self._complete(event.option_index)

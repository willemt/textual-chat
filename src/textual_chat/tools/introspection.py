"""DOM introspection for automatic tool generation.

Walks the Textual DOM to discover widgets and create tools that allow
the LLM to interact with the application.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    Static,
    TabbedContent,
    TextArea,
)

from .datatable import create_datatable_tools

if TYPE_CHECKING:
    from textual.app import App


def _sanitize_tool_name(name: str) -> str:
    """Sanitize a name to be valid for tool names.

    Tool names must match: ^[a-zA-Z0-9_-]{1,128}$
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Ensure it doesn't start with a digit or hyphen
    if sanitized and (sanitized[0].isdigit() or sanitized[0] == "-"):
        sanitized = "_" + sanitized
    # Truncate to 128 chars
    return sanitized[:128]


def introspect_app(
    app: App,
    scope: str = "app",
    exclude_widgets: set[str] | None = None,
) -> tuple[dict[str, Callable], str]:
    """Introspect a Textual app and generate tools + context.

    Args:
        app: The Textual application
        scope: "app" (entire app), "screen" (current screen only), or "parent" (chat's parent)
        exclude_widgets: Widget IDs to exclude (e.g., the Chat widget itself)

    Returns:
        Tuple of (tools_dict, context_string)
    """
    tools: dict[str, Callable] = {}
    context_parts: list[str] = []
    exclude = exclude_widgets or set()

    # Get app-level context
    if app.__doc__:
        context_parts.append(f"Application: {app.__doc__.strip()}")

    # Get screen context
    screen = app.screen
    if screen.__doc__:
        context_parts.append(f"Current screen: {screen.__doc__.strip()}")

    # Determine which widgets to introspect
    # App and Screen are both Widget subclasses, cast for type checker
    if scope == "app":
        root = cast(Widget, app)
    elif scope == "screen":
        root = cast(Widget, screen)
    else:
        root = cast(Widget, app)

    # Walk the DOM and discover widgets
    discovered = _discover_widgets(root, exclude)

    # Generate tools for each discovered widget type
    for i, widget in enumerate(discovered["datatables"]):
        if widget.id:
            widget_id = _sanitize_tool_name(widget.id)
        else:
            widget_id = "table" if i == 0 else f"table_{i + 1}"
        dt_tools = create_datatable_tools(cast(DataTable, widget), widget_id)
        tools.update(dt_tools)
        context_parts.append(f"DataTable '{widget_id}': queryable data table")

    for i, widget in enumerate(discovered["buttons"]):
        if widget.id:
            widget_id = _sanitize_tool_name(widget.id)
        else:
            widget_id = "button" if i == 0 else f"button_{i + 1}"
        btn_tools = _create_button_tools(cast(Button, widget), widget_id)
        tools.update(btn_tools)
        label = str(widget.label) if hasattr(widget, "label") else widget_id
        context_parts.append(f"Button '{widget_id}': {label}")

    for i, widget in enumerate(discovered["inputs"]):
        if widget.id:
            widget_id = _sanitize_tool_name(widget.id)
        else:
            widget_id = "input" if i == 0 else f"input_{i + 1}"
        input_tools = _create_input_tools(cast(Input, widget), widget_id)
        tools.update(input_tools)
        placeholder = getattr(widget, "placeholder", "")
        context_parts.append(f"Input '{widget_id}': {placeholder or 'text input'}")

    for i, widget in enumerate(discovered["textareas"]):
        if widget.id:
            widget_id = _sanitize_tool_name(widget.id)
        else:
            widget_id = "textarea" if i == 0 else f"textarea_{i + 1}"
        ta_tools = _create_textarea_tools(cast(TextArea, widget), widget_id)
        tools.update(ta_tools)
        context_parts.append(f"TextArea '{widget_id}': multi-line text editor")

    for i, widget in enumerate(discovered["labels"]):
        if widget.id:
            widget_id = _sanitize_tool_name(widget.id)
        else:
            widget_id = "label" if i == 0 else f"label_{i + 1}"
        label_tools = _create_label_tools(widget, widget_id)
        tools.update(label_tools)

    for i, widget in enumerate(discovered["tabbed_contents"]):
        # Use widget id, or "tabs" for first one, "tabs_2" for second, etc.
        if widget.id:
            widget_id = _sanitize_tool_name(widget.id)
        else:
            widget_id = "tabs" if i == 0 else f"tabs_{i + 1}"
        tab_tools = _create_tabbed_content_tools(widget, widget_id)  # type: ignore[arg-type]
        tools.update(tab_tools)
        # Get tab names for context
        tab_names = [tab.id or str(j) for j, tab in enumerate(widget.query("TabPane"))]
        context_parts.append(f"TabbedContent '{widget_id}': tabs [{', '.join(tab_names)}]")

    # Add screen navigation tools
    screen_tools = _create_screen_tools(app)
    tools.update(screen_tools)
    if screen_tools:
        context_parts.append("Screen navigation: can push/pop screens")

    # Add app action tools
    action_tools = _create_action_tools(app)
    tools.update(action_tools)
    if action_tools:
        action_names = ", ".join(action_tools.keys())
        context_parts.append(f"App actions: {action_names}")

    context = "\n".join(context_parts)
    return tools, context


def _discover_widgets(
    root: Widget,
    exclude: set[str],
) -> dict[str, list[Widget]]:
    """Walk the DOM and categorize widgets by type."""
    discovered: dict[str, list[Widget]] = {
        "datatables": [],
        "buttons": [],
        "inputs": [],
        "textareas": [],
        "labels": [],
        "tabbed_contents": [],
        "other": [],
    }

    def walk(widget: Widget) -> None:
        # Skip excluded widgets
        if widget.id and widget.id in exclude:
            return

        # Categorize by type
        if isinstance(widget, DataTable):
            discovered["datatables"].append(widget)
        elif isinstance(widget, Button):
            discovered["buttons"].append(widget)
        elif isinstance(widget, Input):
            discovered["inputs"].append(widget)
        elif isinstance(widget, TextArea):
            # Skip chat input areas
            if widget.id != "chat-input":
                discovered["textareas"].append(widget)
        elif isinstance(widget, (Label, Static)):
            # Only include labeled ones
            if widget.id:
                discovered["labels"].append(widget)
        elif isinstance(widget, TabbedContent):
            discovered["tabbed_contents"].append(widget)

        # Recurse into children
        for child in widget.children:
            walk(child)

    walk(root)
    return discovered


def _create_button_tools(button: Button, name: str) -> dict[str, Callable]:
    """Create tools for a Button widget."""

    def click() -> str:
        """Click this button."""
        button.press()
        return f"Clicked button '{name}'"

    return {f"click_{name}": click}


def _create_input_tools(input_widget: Input, name: str) -> dict[str, Callable]:
    """Create tools for an Input widget."""

    def get_value() -> str:
        """Get the current value of this input field."""
        return input_widget.value

    def set_value(value: str) -> str:
        """Set the value of this input field.

        Args:
            value: The new value to set
        """
        input_widget.value = value
        return f"Set '{name}' to '{value}'"

    def focus() -> str:
        """Focus this input field."""
        input_widget.focus()
        return f"Focused input '{name}'"

    return {
        f"get_{name}": get_value,
        f"set_{name}": set_value,
        f"focus_{name}": focus,
    }


def _create_textarea_tools(textarea: TextArea, name: str) -> dict[str, Callable]:
    """Create tools for a TextArea widget."""

    def get_text() -> str:
        """Get the current text content."""
        return textarea.text

    def set_text(text: str) -> str:
        """Set the text content.

        Args:
            text: The new text content
        """
        textarea.text = text
        return f"Set '{name}' text"

    def append_text(text: str) -> str:
        """Append text to the end.

        Args:
            text: The text to append
        """
        textarea.text += text
        return f"Appended to '{name}'"

    return {
        f"get_{name}_text": get_text,
        f"set_{name}_text": set_text,
        f"append_{name}_text": append_text,
    }


def _create_label_tools(label: Widget, name: str) -> dict[str, Callable]:
    """Create tools for a Label/Static widget."""

    def read() -> str:
        """Read the text displayed in this label."""
        if hasattr(label, "renderable"):
            return str(label.renderable)
        return str(label.render())

    return {f"read_{name}": read}


def _create_tabbed_content_tools(tabbed: TabbedContent, name: str) -> dict[str, Callable]:
    """Create tools for a TabbedContent widget."""

    def get_active_tab() -> str:
        """Get the currently active tab ID."""
        return tabbed.active

    def list_tabs() -> list[str]:
        """List all available tab IDs."""
        return [tab.id for tab in tabbed.query("TabPane") if tab.id]

    def switch_tab(tab_id: str) -> str:
        """Switch to a different tab.

        Args:
            tab_id: The ID of the tab to switch to
        """
        tabbed.active = tab_id
        return f"Switched to tab '{tab_id}'"

    return {
        f"get_active_tab_{name}": get_active_tab,
        f"list_tabs_{name}": list_tabs,
        f"switch_tab_{name}": switch_tab,
    }


def _create_screen_tools(app: App) -> dict[str, Callable]:
    """Create tools for screen navigation."""
    tools: dict[str, Callable] = {}

    def get_screen_stack() -> list[str]:
        """Get the current screen stack."""
        return [s.__class__.__name__ for s in app.screen_stack]

    def pop_screen() -> str:
        """Go back to the previous screen."""
        if len(app.screen_stack) > 1:
            app.pop_screen()
            return f"Popped screen, now on {app.screen.__class__.__name__}"
        return "Cannot pop - already on root screen"

    tools["get_screen_stack"] = get_screen_stack
    tools["pop_screen"] = pop_screen

    # Add tools for installed screens
    for screen_name in app._installed_screens:

        def push_screen(name: str = screen_name) -> str:
            """Push a screen onto the stack.

            Args:
                name: Internal screen name parameter
            """
            app.push_screen(name)
            return f"Pushed screen '{name}'"

        safe_name = _sanitize_tool_name(screen_name)
        tools[f"push_screen_{safe_name}"] = push_screen

    return tools


def _create_action_tools(app: App) -> dict[str, Callable]:
    """Create tools for app actions from BINDINGS."""
    tools: dict[str, Callable] = {}

    # Get bindings from app and current screen
    all_bindings = []
    if hasattr(app, "BINDINGS"):
        all_bindings.extend(app.BINDINGS)
    if hasattr(app.screen, "BINDINGS"):
        all_bindings.extend(app.screen.BINDINGS)

    for binding in all_bindings:
        # Bindings can be Binding objects or tuples - only process Binding objects
        if not isinstance(binding, Binding):
            continue

        # Skip common/internal actions
        action = binding.action
        if action in ("focus_next", "focus_previous"):
            continue

        # Use just the action name for the tool name (not the full binding repr)
        action_name = _sanitize_tool_name(action)
        description = binding.description

        def run_action(act: str = action, _app: App = app) -> str:
            """Run this action."""
            result = _app.run_action(act)
            # run_action may return a coroutine, but we can't await it in sync context
            # The action will still execute, we just don't wait for completion
            if result is not None:
                pass  # Acknowledge the result exists
            return f"Executed action '{act}'"

        # Use description in docstring
        run_action.__doc__ = f"{description}. Triggers the '{action}' action."
        tools[f"action_{action_name}"] = run_action

    return tools

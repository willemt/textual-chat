"""DOM introspection for automatic tool generation.

Walks the Textual DOM to discover widgets and create tools that allow
the LLM to interact with the application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Static, TextArea

from .datatable import create_datatable_tools

if TYPE_CHECKING:
    from textual.app import App
    from textual.screen import Screen


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
    if scope == "app":
        root = app
    elif scope == "screen":
        root = screen
    else:
        root = app  # Default to app

    # Walk the DOM and discover widgets
    discovered = _discover_widgets(root, exclude)

    # Generate tools for each discovered widget type
    for widget in discovered["datatables"]:
        widget_id = widget.id or f"table_{id(widget)}"
        dt_tools = create_datatable_tools(widget, widget_id)
        tools.update(dt_tools)
        context_parts.append(f"DataTable '{widget_id}': queryable data table")

    for widget in discovered["buttons"]:
        widget_id = widget.id or f"button_{id(widget)}"
        btn_tools = _create_button_tools(widget, widget_id)
        tools.update(btn_tools)
        label = str(widget.label) if hasattr(widget, "label") else widget_id
        context_parts.append(f"Button '{widget_id}': {label}")

    for widget in discovered["inputs"]:
        widget_id = widget.id or f"input_{id(widget)}"
        input_tools = _create_input_tools(widget, widget_id)
        tools.update(input_tools)
        placeholder = getattr(widget, "placeholder", "")
        context_parts.append(f"Input '{widget_id}': {placeholder or 'text input'}")

    for widget in discovered["textareas"]:
        widget_id = widget.id or f"textarea_{id(widget)}"
        ta_tools = _create_textarea_tools(widget, widget_id)
        tools.update(ta_tools)
        context_parts.append(f"TextArea '{widget_id}': multi-line text editor")

    for widget in discovered["labels"]:
        widget_id = widget.id or f"label_{id(widget)}"
        label_tools = _create_label_tools(widget, widget_id)
        tools.update(label_tools)

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

        tools[f"push_screen_{screen_name}"] = push_screen

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
        # Skip common/internal actions
        action = binding.action if hasattr(binding, "action") else str(binding)
        if action in ("quit", "cancel", "focus_next", "focus_previous"):
            continue

        action_name = action.replace(".", "_")
        description = binding.description if hasattr(binding, "description") else action

        def run_action(act: str = action) -> str:
            """Run this action."""
            app.action(act)
            return f"Executed action '{act}'"

        # Use description in docstring
        run_action.__doc__ = f"{description}. Triggers the '{action}' action."
        tools[f"action_{action_name}"] = run_action

    return tools

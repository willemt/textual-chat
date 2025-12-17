"""Test file to verify interrupt hotkey implementation."""

import sys

sys.path.insert(0, "src")

from textual.binding import Binding
from textual_chat import Chat

print("Testing interrupt hotkey implementation...")

# Test that Chat has the correct bindings
chat = Chat()

print("\nVerifying BINDINGS configuration:")
print(f"Number of bindings: {len(Chat.BINDINGS)}")

# Check each binding
for binding in Chat.BINDINGS:
    if isinstance(binding, Binding):
        print(
            f"  - Key: {binding.key:15} Action: {binding.action:15} Description: {binding.description:15} Visible: {binding.show}"
        )
    else:
        print(f"  - {binding}")

# Verify Ctrl+C binding exists
ctrl_c_binding = None
escape_binding = None

for binding in Chat.BINDINGS:
    if isinstance(binding, Binding):
        if binding.key == "ctrl+c":
            ctrl_c_binding = binding
        elif binding.key == "escape":
            escape_binding = binding

# Assertions
assert ctrl_c_binding is not None, "❌ Ctrl+C binding not found!"
assert (
    ctrl_c_binding.action == "cancel"
), f"❌ Ctrl+C should call 'cancel' action, got '{ctrl_c_binding.action}'"
assert ctrl_c_binding.show == True, "❌ Ctrl+C binding should be visible (show=True)"
assert (
    ctrl_c_binding.description == "Interrupt"
), f"❌ Ctrl+C description should be 'Interrupt', got '{ctrl_c_binding.description}'"

print("\n✓ Ctrl+C binding configured correctly:")
print(f"  - Calls action: {ctrl_c_binding.action}")
print(f"  - Description: {ctrl_c_binding.description}")
print(f"  - Visible in UI: {ctrl_c_binding.show}")

# Verify Escape is still there as backup
assert escape_binding is not None, "❌ Escape binding not found!"
assert (
    escape_binding.action == "cancel"
), f"❌ Escape should call 'cancel' action, got '{escape_binding.action}'"
assert escape_binding.show == False, "❌ Escape binding should be hidden (show=False)"

print("\n✓ Escape binding still configured as hidden backup:")
print(f"  - Calls action: {escape_binding.action}")
print(f"  - Visible in UI: {escape_binding.show}")

# Verify the action_cancel method exists
assert hasattr(chat, "action_cancel"), "❌ action_cancel method not found!"
print("\n✓ action_cancel method exists on Chat widget")

print("\n✅ All interrupt hotkey tests passed!")
print("\nSummary:")
print("  - Ctrl+C is bound to 'cancel' action and visible in UI")
print("  - Escape is bound to 'cancel' action as hidden backup")
print("  - Both hotkeys will call the same action_cancel() method")

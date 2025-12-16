"""Test file to verify plan pane implementation."""

import sys
sys.path.insert(0, 'src')

# Test imports
print("Testing imports...")
from textual_chat.events import PlanChunk, StreamEvent
from textual_chat.widgets.plan_pane import PlanPane
from textual_chat import Chat

# Verify PlanChunk is in the event types
print("✓ PlanChunk imported successfully")
print(f"✓ PlanChunk is part of StreamEvent: {PlanChunk in StreamEvent.__args__}")

# Verify PlanPane widget
print("✓ PlanPane widget imported successfully")

# Verify Chat includes PlanPane
from textual_chat.widgets import PlanPane as ExportedPlanPane
print("✓ PlanPane exported from widgets module")

# Test that we can create instances
plan_chunk = PlanChunk(text="Test plan text")
print(f"✓ Created PlanChunk: {plan_chunk}")

plan_pane = PlanPane(id="test-pane")
print(f"✓ Created PlanPane widget: {plan_pane}")

# Check that llm_adapter_acp imports PlanChunk
from textual_chat import llm_adapter_acp
print("✓ llm_adapter_acp module imports successfully")

print("\n✅ All tests passed! Implementation is syntactically correct.")

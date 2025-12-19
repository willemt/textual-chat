# Smart Interrupt Implementation Summary

## Issue #5: Agent Support for Sending Messages While Working

### Problem
Users couldn't send messages while an agent was responding, and even when they could queue messages, the agent had no context about being interrupted or what the user's new priority was.

### Solution: Smart Interrupt System
Implemented an intelligent interrupt mechanism that:
1. **Cancels the current agent task** using ACP protocol
2. **Provides context** about what was being worked on
3. **Prioritizes the new message** while maintaining conversation coherence
4. **Falls back gracefully** for non-ACP adapters

## How It Works

### For ACP Adapters (Smart Interrupt)

When a user sends a message while the agent is working:

1. **Cancellation Phase**
   - Sets `_cancel_requested = True`
   - Cancels the Python asyncio task
   - Sends ACP `cancel()` notification to agent process
   - Waits 100ms for clean cancellation

2. **Context Building**
   - Retrieves original task from `_current_user_message`
   - Constructs combined prompt with context:
     ```
     [Context: I was working on: "original task"]
     
     [INTERRUPTION] The user has sent a new message that takes priority:
     
     {new_message}
     
     Please address this new message. If it's related to the previous task, 
     you may continue with that context. If it's unrelated, focus on the new request.
     ```

3. **Re-execution**
   - Adds user's actual message to UI (not the combined prompt)
   - Sends combined prompt to agent internally
   - Agent receives full context and can respond intelligently

### For Non-ACP Adapters (Queue Fallback)

- Falls back to the original queuing behavior
- Messages show with "pending" visual state
- Processed sequentially after current response completes

## Code Changes

### 1. Added Context Tracking (`chat.py:424`)
```python
self._current_user_message: str | None = None  # Track current prompt for interruption context
```

### 2. Refactored Message Sending (`chat.py:972-979`)
```python
async def _send(self, content: str) -> None:
    """Send a message and get a response using the adapter."""
    # Add user message to UI and history
    self._add_message("user", content)
    self._message_history.append({"role": "user", "content": content})
    self.post_message(self.Sent(content))
    
    # Send to agent
    await self._send_internal(content)
```

Separated UI updates from agent communication, enabling interrupt to show user's message while sending context internally.

### 3. New Internal Send Method (`chat.py:981-1143`)
```python
async def _send_internal(self, content: str) -> None:
    """Internal method to send a prompt to the agent (without modifying UI)."""
```

Core logic for agent communication without UI updates. Used for sending context-enriched prompts during interrupts.

### 4. Smart Interrupt Method (`chat.py:1145-1207`)
```python
async def _interrupt_with_message(self, new_message: str) -> None:
    """Interrupt the current agent task with a new message.
    
    For ACP adapter: Cancels the current task and sends a combined prompt with context.
    For other adapters: Falls back to queuing behavior.
    """
```

Key features:
- Detects ACP adapter and capabilities
- Cancels at both asyncio and ACP protocol levels
- Builds intelligent context-aware prompt
- Falls back gracefully for unsupported adapters

### 5. Modified Input Handler (`chat.py:964-967`)
```python
# If agent is responding, interrupt with the new message
if self._is_responding:
    await self._interrupt_with_message(content)
    return
```

Changed from blocking/queuing to smart interruption.

## User Experience

### ACP Adapter Flow
1. User sends: "Write a long blog post"
2. Agent starts working (using tools, thinking, writing)
3. User interrupts: "Actually, make it short instead"
4. Status shows: "⚡ Interrupting agent with new message..."
5. Agent receives context and adjusts immediately
6. Response: "I see you want it short instead. Here's a concise version..."

### Visual Feedback
- **Interrupting**: Status shows "⚡ Interrupting agent with new message..."
- **New message**: Appears normally in chat (not marked as pending)
- **Agent context**: Hidden from user, sent internally

### Non-ACP Adapter Flow
- Same as before: messages queue and process sequentially
- Visual "pending" indicator with yellow border
- Status shows queue count

## Technical Details

### ACP Protocol Integration
Uses the standard ACP `cancel()` method:
```python
await self._conversation._conn.cancel(self._conversation._session_id)
```

This sends a `session/cancel` notification to the agent, allowing it to clean up gracefully.

### Context Preservation
The combined prompt format ensures agents understand:
- What they were working on (context)
- Why they're receiving a new message (interruption)
- What to prioritize (new message)
- How to handle it (related vs unrelated)

### Graceful Degradation
- Checks for ACP adapter: `self._adapter.__name__ == "textual_chat.llm_adapter_acp"`
- Verifies connection exists: `hasattr(self._conversation, "_conn")`
- Confirms session ID: `hasattr(self._conversation, "_session_id")`
- Falls back to queuing if any check fails

## Files Modified

1. **`src/textual_chat/chat.py`**
   - Added `_current_user_message` tracker
   - Refactored `_send()` to separate UI from agent communication
   - Added `_send_internal()` for context-enriched prompts
   - Implemented `_interrupt_with_message()` with ACP cancel support
   - Modified input handler to use interruption

2. **`test_message_interrupt.py`** (new)
   - Test script demonstrating interrupt functionality
   - Instructions for manual testing
   - Expected behavior documentation

## Benefits Over Simple Queuing

### Queue-Only Approach (Previous)
- ❌ Agent unaware of new messages until completion
- ❌ Wastes compute on potentially obsolete work
- ❌ User waits for irrelevant response
- ❌ No way to change direction mid-task

### Smart Interrupt Approach (New)
- ✅ Agent immediately aware of new priorities
- ✅ Can adjust response to new context
- ✅ Respects user's change of mind
- ✅ Maintains conversation coherence
- ✅ Provides smooth UX without data loss

## Example Scenarios

### Scenario 1: Direction Change
```
User: "Analyze this large dataset and create visualizations"
Agent: *starts complex analysis*
User: "Wait, just give me a summary instead"
Agent: *cancels analysis, provides summary with context*
```

### Scenario 2: Additional Context
```
User: "Debug this error"
Agent: *investigating*
User: "I just noticed it only happens on Windows"
Agent: *receives context, focuses on Windows-specific issues*
```

### Scenario 3: Related Follow-up
```
User: "Explain quantum computing"
Agent: *writing detailed explanation*
User: "Focus on practical applications"
Agent: *pivots to applications while using prior context*
```

## Testing

### Manual Testing
```bash
python test_message_interrupt.py
```

### Test Cases
1. ✅ Interrupt with unrelated message
2. ✅ Interrupt with clarification
3. ✅ Interrupt with priority change
4. ✅ Multiple rapid interrupts
5. ✅ Interrupt during tool execution
6. ✅ Fallback for non-ACP adapters

## Future Enhancements

Possible improvements:
1. **User choice**: Option to queue vs interrupt
2. **Smart detection**: Auto-detect if message is related
3. **Undo support**: Ability to resume cancelled work
4. **Partial results**: Save agent's work-in-progress
5. **Multi-threading**: True parallel agent sessions

## Compatibility

- **ACP adapters**: Full smart interrupt support
- **LiteLLM adapters**: Falls back to queuing
- **Custom adapters**: Falls back to queuing
- **No breaking changes**: Existing code continues to work

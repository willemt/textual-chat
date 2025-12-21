# Slash Command Autocomplete Feature

## Overview

This implementation adds an autocomplete popup for slash commands in the textual-chat application. When users type `/` in the chat input, a dropdown menu appears showing available commands with fuzzy search filtering.

## Implementation Details

### Files Modified/Created

1. **`pyproject.toml`**
   - Added `textual-autocomplete>=3.0.0` dependency

2. **`src/textual_chat/widgets/slash_command_autocomplete.py`** (NEW)
   - Custom autocomplete widget adapted for TextArea (since textual-autocomplete only supports Input)
   - `SlashCommand` dataclass for command definitions
   - `SlashCommandAutocomplete` widget with fuzzy search

3. **`src/textual_chat/widgets/__init__.py`**
   - Exported `SlashCommand` and `SlashCommandAutocomplete`

4. **`src/textual_chat/chat.py`**
   - Added imports for `SlashCommand` and `SlashCommandAutocomplete`
   - Added `_get_slash_commands()` method to generate available commands
   - Integrated autocomplete widget in `compose()` method

5. **`test_autocomplete.py`** (NEW)
   - Test suite to verify the implementation

## Features

### User Experience
- **Trigger**: Type `/` at the start of a line in the chat input
- **Navigation**: Use arrow keys (↑/↓) to navigate options
- **Selection**: Press Tab or Enter to complete the command
- **Cancel**: Press Escape to hide the dropdown
- **Fuzzy Search**: Commands are filtered as you type (e.g., typing `/mo` shows `/model`)

### Adaptive Commands
The autocomplete shows different commands based on the adapter:
- **LiteLLM adapter**: `/help`, `/model`
- **ACP adapter**: `/help`, `/agent`

### Technical Features
- Fuzzy search powered by `textual-autocomplete`'s `FuzzySearch`
- Proper keyboard event handling (prevents default behavior when dropdown is active)
- Screen-aligned dropdown positioning
- Auto-hide on focus loss
- Reactive to target widget changes

## Architecture

### SlashCommand Dataclass
```python
@dataclass
class SlashCommand:
    name: str           # Command name (without /)
    description: str    # Description shown in dropdown
    handler: Callable[[], None] | None = None  # Optional handler
```

### SlashCommandAutocomplete Widget
- Extends Textual's `Widget`
- Composes an `OptionList` for the dropdown
- Subscribes to TextArea events for text changes
- Implements fuzzy matching and filtering
- Handles completion and insertion

### Integration Points
1. **Chat.compose()**: Yields the autocomplete widget with `#chat-input` as target
2. **Chat._get_slash_commands()**: Provides available commands based on adapter
3. **Chat._handle_slash_command()**: Existing handler processes completed commands

## Testing

Run the test suite:
```bash
python test_autocomplete.py
```

Test manually with any example:
```bash
python examples/basic.py
```

Then type `/` in the input to see the autocomplete popup.

## Keyboard Controls

| Key | Action |
|-----|--------|
| `/` | Show autocomplete popup |
| ↓ | Navigate down / show popup |
| ↑ | Navigate up |
| Tab | Complete selection |
| Enter | Complete selection (when dropdown is shown) |
| Escape | Hide dropdown |

## Design Decisions

1. **TextArea vs Input**: Created custom autocomplete for TextArea instead of switching to Input to preserve multiline support
2. **Fuzzy Search**: Used textual-autocomplete's fuzzy search for better UX
3. **Command Registry**: Created `SlashCommand` dataclass for structured command definitions
4. **Adaptive Commands**: Commands change based on adapter to avoid showing invalid options
5. **Line-start only**: Autocomplete only triggers at the start of a line to avoid conflicts with regular text

## Future Enhancements

Potential improvements:
- Add command arguments autocomplete
- Support custom user-defined commands
- Add command history/recently used
- Show command usage examples in dropdown
- Add keyboard shortcuts display

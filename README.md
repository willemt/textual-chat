# textual-chat

**LLM chat for humans.** Add AI to your terminal app in a few lines of code.

```python
from textual.app import App, ComposeResult
from textual_chat import Chat

class MyApp(App):
    def compose(self) -> ComposeResult:
        yield Chat()

MyApp().run()
```

That's it. No configuration, no boilerplate, no PhD required.

## Features

- 🚀 **Zero-config** - Auto-detects your LLM setup and just works
- 🛠️ **Function calling** - Decorate Python functions as tools
- 🤖 **ACP agents** - Works with Claude Code, OpenCode, and custom agents
- 💭 **Extended thinking** - See the model's reasoning process (Claude)
- 💾 **Session persistence** - Resume conversations across restarts
- 🎨 **Fully customizable** - It's a Textual widget, style it however you want
- ⚡ **Real-time streaming** - See responses and tool calls as they happen

## Install

```bash
uv add textual-chat
```

Or with pip: `pip install textual-chat`

For ACP agent support:
```bash
uv add textual-chat[acp]
```

## Quick Start

### 1. Set an API key (or run Ollama locally)

```bash
# Pick one:
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Or run locally (free!):
brew install ollama && ollama run llama3.2
```

textual-chat auto-detects your setup. It just works.

### 2. Create your chat app

```python
from textual.app import App, ComposeResult
from textual_chat import Chat

chat = Chat(
    model="claude-sonnet-4-20250514",  # Optional: pick your model
    system="You are a helpful assistant."  # Optional: set personality
)

class MyApp(App):
    def compose(self) -> ComposeResult:
        yield chat

MyApp().run()
```

## Tools (Function Calling)

Give the LLM superpowers with a decorator:

```python
chat = Chat()

@chat.tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"72°F and sunny in {city}"

@chat.tool
def search_docs(query: str) -> str:
    """Search the documentation."""
    return search(query)
```

The LLM will call your functions when needed. Type hints become the schema. Docstrings become descriptions. No JSON wrestling required.

## Agent Client Protocol (ACP)

Work with powerful agents like Claude Code or build your own:

```python
Chat(
    model="claude-code-acp",  # or "opencode acp" or path to your agent
    adapter="acp",
    cwd="/path/to/project",  # Working directory for the agent
)
```

ACP agents have their own tools (bash, file operations, web search, etc.) and can:
- Execute commands in your environment
- Read and write files
- Search the web
- Maintain long-running sessions

See `examples/acp_chat.py` for a complete example.

## Extended Thinking

Enable Claude's extended thinking to see its reasoning:

```python
Chat(
    model="claude-sonnet-4-20250514",
    thinking=True,  # Show reasoning (10k token budget)
    # thinking=50000,  # Or specify custom budget
)
```

The model will show its internal thought process before responding.

## Session Management

Sessions are automatically saved and restored:

```python
chat = Chat(
    model="claude-sonnet-4-20250514",
    cwd="/path/to/project",  # Sessions are per-directory
)
```

When you restart your app, you'll be prompted to resume or start fresh. Sessions include:
- Full conversation history
- Agent state (for ACP agents)
- Tool definitions

## Configuration

### Basic Options

```python
Chat(
    model="claude-sonnet-4-20250514",  # Model ID or agent command
    adapter="litellm",                  # "litellm" or "acp"
    system="You are a pirate.",         # System prompt
    temperature=0.9,                    # Response randomness (0-2)
    thinking=True,                      # Extended thinking (Claude)
    show_token_usage=True,              # Show token counts
    show_model_selector=True,           # Allow model switching
)
```

### Advanced Options

```python
Chat(
    api_key="sk-...",                   # Override API key
    api_base="https://...",             # Custom endpoint
    cwd="/path/to/project",             # Working directory
    max_tokens=4096,                    # Max response length
    introspect=True,                    # Auto-discover tools from parent app
)
```

## Embed Anywhere

textual-chat is just a Textual widget. Use it however you want:

### Sidebar

```python
from textual.containers import Horizontal

class MyApp(App):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield MyMainContent()
            yield Chat(system="You help users with this app.")
```

### Modal

```python
class MyApp(App):
    def action_show_assistant(self) -> None:
        self.push_screen(ChatModal())

class ChatModal(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Chat()
```

See `examples/chatbot_sidebar.py` and `examples/chatbot_modal.py` for complete examples.

## Keyboard Shortcuts

- `Ctrl+L` - Clear chat
- `Ctrl+X` - Interrupt agent response
- `Escape` - Cancel response
- `Enter` - Send message
- `/model` - Switch model (if enabled)
- `/agent` - Switch agent (ACP mode)

## Events

React to chat events in your app:

```python
def on_chat_sent(self, event: Chat.Sent):
    """User sent a message."""
    log(f"User: {event.content}")

def on_chat_responded(self, event: Chat.Responded):
    """Assistant responded."""
    log(f"Assistant: {event.content}")

def on_chat_tool_called(self, event: Chat.ToolCalled):
    """A tool was executed."""
    log(f"Tool {event.name}({event.arguments}) returned: {event.result}")
```

## Model Auto-detection

No model specified? We check (in order):

1. `ANTHROPIC_API_KEY` → Claude Sonnet 4
2. `OPENAI_API_KEY` → GPT-4o-mini
3. `GITHUB_TOKEN` → GitHub Models
4. `GEMINI_API_KEY` → Gemini Flash
5. `GROQ_API_KEY` → Llama 3.1
6. `DEEPSEEK_API_KEY` → DeepSeek
7. Ollama running? → Local Llama

No keys found? You'll see a friendly setup guide.

## Architecture

textual-chat uses an **event-based streaming architecture**:

### Event Types

Responses stream as typed events in chronological order:

- **MessageChunk** - Text from the assistant
- **ThoughtChunk** - Reasoning text (extended thinking)
- **ToolCallStart** - Tool beginning execution
- **ToolCallComplete** - Tool finished with output
- **TokenUsage** - Token consumption stats

This allows real-time UI updates and natural ordering of interleaved tool calls and responses.

### Adapters

Two adapters provide a unified interface:

- **LiteLLM adapter** - 100+ LLM providers (OpenAI, Anthropic, Google, etc.)
- **ACP adapter** - Agent Client Protocol (Claude Code, custom agents)

Both implement the same event-based streaming API, so switching between them is seamless.

## Examples

```bash
# Basic examples
uv run examples/basic.py              # Minimal chat
uv run examples/with_tools.py         # Weather, time, calculator
uv run examples/custom_model.py       # Pirate personality
uv run examples/with_thinking.py      # Extended thinking

# Integration examples
uv run examples/in_larger_app.py      # Sidebar integration
uv run examples/chatbot_modal.py      # Modal dialog
uv run examples/chatbot_sidebar.py    # Sidebar assistant

# ACP agent examples
uv run examples/acp_chat.py examples/echo_agent.py
uv run examples/acp_chat.py examples/tool_agent.py
uv run examples/opencode_chat.py      # OpenCode integration
```

## Development

### Setup

```bash
git clone https://github.com/your-org/textual-chat.git
cd textual-chat
uv sync --all-extras
```

### Type Checking

```bash
uv run mypy src/textual_chat
```

### Code Formatting

```bash
uv run ruff check src/textual_chat
uv run black src/textual_chat
```

### Pre-commit Hooks

```bash
uv run pre-commit install
```

## Philosophy

- **Sensible defaults** - Works out of the box
- **Progressive disclosure** - Simple things simple, complex things possible
- **Tools are just functions** - No schema files, no JSON, just Python
- **Plays nice** - It's a Textual widget, compose it however you want
- **Event-driven** - Real-time streaming with natural event ordering

## Contributing

Contributions welcome! This is an open-source project.

Areas we'd love help with:
- Additional examples
- Documentation improvements
- Bug fixes and testing
- New adapter types (MCP, custom protocols)

## License

MIT

---

Built with [Textual](https://textual.textualize.io/) • Powered by [LiteLLM](https://litellm.ai/) and [ACP](https://agentclientprotocol.github.io/)

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

- **Zero-config** - Auto-detects your LLM setup and just works
- **Function calling** - Decorate Python functions as tools
- **ACP agents** - Works with Claude Code, OpenCode, and custom agents
- **Extended thinking** - See the model's reasoning process (Claude)
- **Session persistence** - Resume conversations across restarts
- **Fully customizable** - It's a Textual widget, style it however you want

## Install

```bash
uv add textual-chat
```

Or with pip: `pip install textual-chat`

For ACP agent support: `uv add textual-chat[acp]`

## Quick Start

Set an API key (or run Ollama locally):

```bash
export ANTHROPIC_API_KEY=sk-ant-...  # or OPENAI_API_KEY
# Or: brew install ollama && ollama run llama3.2
```

Then run:

```python
from textual.app import App, ComposeResult
from textual_chat import Chat

class MyApp(App):
    def compose(self) -> ComposeResult:
        yield Chat(
            model="claude-sonnet-4-20250514",  # Optional
            system="You are a helpful assistant.",  # Optional
        )

MyApp().run()
```

## Tools

Give the LLM superpowers with a decorator:

```python
chat = Chat()

@chat.tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"72°F and sunny in {city}"
```

Type hints become the schema. Docstrings become descriptions.

## Examples

See the `examples/` folder for complete examples:

| Example | Description |
|---------|-------------|
| `basic.py` | Minimal chat app |
| `with_tools.py` | Function calling |
| `with_thinking.py` | Extended thinking (Claude) |
| `with_mcp.py` | MCP server tools |
| `custom_model.py` | Custom model and system prompt |
| `in_larger_app.py` | Sidebar integration with tools |
| `chatbot_modal.py` | Modal dialog pattern |
| `chatbot_sidebar.py` | Toggleable sidebar |
| `with_tabs.py` | Tabbed interface |
| `acp_chat.py` | ACP agent integration |

Run any example:

```bash
uv run examples/basic.py
uv run examples/acp_chat.py examples/echo_agent.py
```

## Configuration

```python
Chat(
    model="claude-sonnet-4-20250514",  # Model ID or agent command
    adapter="litellm",                  # "litellm" or "acp"
    system="You are a pirate.",         # System prompt
    temperature=0.9,                    # Response randomness
    thinking=True,                      # Extended thinking (Claude)
    tools=[fn1, fn2],                   # Tool functions
    cwd="/path/to/project",             # Working directory
    show_token_usage=True,              # Show token counts
    show_model_selector=True,           # Allow /model switching
)
```

## Events

React to chat events in your app:

```python
def on_chat_sent(self, event: Chat.Sent):
    log(f"User: {event.content}")

def on_chat_responded(self, event: Chat.Responded):
    log(f"Assistant: {event.content}")

def on_chat_tool_called(self, event: Chat.ToolCalled):
    log(f"Tool {event.name}: {event.result}")
```

## Model Auto-detection

No model specified? We check (in order):

1. `ANTHROPIC_API_KEY` → Claude Sonnet 4
2. `OPENAI_API_KEY` → GPT-4o-mini
3. `GITHUB_TOKEN` → GitHub Models
4. `GEMINI_API_KEY` → Gemini Flash
5. `GROQ_API_KEY` → Llama 3.1
6. Ollama running → Local Llama

## Development

```bash
git clone https://github.com/anthropics/textual-chat.git
cd textual-chat
uv sync --all-extras
uv run pytest tests/
uv run mypy src/textual_chat
```

## License

MIT

---

Built with [Textual](https://textual.textualize.io/) and [LiteLLM](https://litellm.ai/)

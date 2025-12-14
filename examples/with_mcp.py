"""Chat with MCP server tools.

This example shows how to use FastMCP servers with the Chat widget.
"""

from mcp.server import FastMCP
from textual.app import App, ComposeResult
from textual.widgets import Footer

from textual_chat import Chat

# Create an MCP server with tools
mcp = FastMCP("Demo Server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Nice to meet you."


# You can also pass regular functions alongside MCP
def get_secret() -> str:
    """Get a secret message."""
    return "The cake is a lie."


# Create chat with MCP server and regular tools in one list
chat = Chat(
    tools=[get_secret, mcp],
    system="You are a helpful assistant with access to math and greeting tools.",
)


class MCPApp(App):
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield chat
        yield Footer()


if __name__ == "__main__":
    MCPApp().run()

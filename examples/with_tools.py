"""Chat with tools. The LLM can call your Python functions."""

import random
from datetime import datetime

from textual.app import App, ComposeResult

from textual_chat import Chat


# Define tool functions
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    conditions = ["Sunny", "Cloudy", "Rainy", "Windy"]
    return f"{random.choice(conditions)}, {random.randint(60, 85)}°F in {city}"


def get_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%I:%M %p on %B %d, %Y")


def calculate(expression: str) -> str:
    """Calculate a math expression like '2 + 2' or '100 / 4'."""
    try:
        # Only allow safe characters
        if all(c in "0123456789+-*/.() " for c in expression):
            return str(eval(expression))
        return "Invalid expression"
    except Exception as e:
        return f"Error: {e}"


# Pass tools directly to Chat
chat = Chat(tools=[get_weather, get_time, calculate])


class ToolApp(App):
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield chat


if __name__ == "__main__":
    ToolApp().run()

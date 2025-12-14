"""Customize the model, system prompt, and behavior."""

from textual.app import App, ComposeResult

from textual_chat import Chat

chat = Chat(
    model="claude-sonnet-4-20250514",  # Or "gpt-4o", "ollama/llama3", etc.
    system="You are a friendly pirate. Respond in pirate speak, matey!",
    temperature=0.9,
)


class PirateApp(App):
    def compose(self) -> ComposeResult:
        yield chat


if __name__ == "__main__":
    PirateApp().run()

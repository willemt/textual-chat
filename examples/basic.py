"""The simplest possible chat app. 6 lines of code."""

from textual.app import App, ComposeResult

from textual_chat import Chat


class ChatApp(App):
    def compose(self) -> ComposeResult:
        yield Chat()


if __name__ == "__main__":
    ChatApp().run()

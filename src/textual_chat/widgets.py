"""Chat widgets for Textual applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Markdown, Static

from textual_golden import Golden, BLUE

if TYPE_CHECKING:
    from .db import ChatDatabase, Conversation

Role = Literal["user", "assistant", "system"]


@dataclass
class ToolUse:
    """Represents a tool use/call."""

    name: str
    args: dict[str, Any]

    def __str__(self) -> str:
        args_str = ", ".join(
            f"{k}={v!r}" for k, v in sorted(self.args.items(), key=lambda x: x[0])
        )
        return f"{self.name}({args_str})"

    def to_widget(self) -> Widget:
        text = Static(str(self), markup=False)
        text.styles.width = "1fr"
        text.styles.height = "auto"
        text.styles.margin = (0, 0, 0, 0)
        return text


def _humanize_tokens(n: int) -> str:
    """Humanize token count: 3000 -> 3k, 1500 -> 1.5k."""
    if n < 1000:
        return str(n)
    k = n / 1000
    if k >= 10:
        return f"{int(k)}k"
    # Show one decimal, strip trailing zero
    formatted = f"{k:.1f}".rstrip("0").rstrip(".")
    return f"{formatted}k"


class MessageWidget(Static):
    """A chat message widget with loading states and tool display."""

    def __init__(
        self, role: str, content: str = "", loading: bool = False, title: str | None = None
    ) -> None:
        super().__init__(classes=f"message {role}")
        self.role = role
        self._content = content
        self._loading = loading
        self.border_title = title or role.title()

    def compose(self) -> ComposeResult:
        if self._loading:
            yield Golden("Responding...", classes="content")
        else:
            yield Markdown(self._content, classes="content")

    def _scroll_parent(self) -> None:
        """Scroll parent container to show this message."""
        try:
            if self.parent:
                self.parent.scroll_end(animate=False)
        except Exception:
            pass

    def update_content(self, content: str) -> None:
        """Update the message content."""
        self._content = content
        self._loading = False
        try:
            # Remove any existing content widget
            old_content = self.query_one(".content")
            old_content.remove()
        except NoMatches:
            pass
        # Mount new Markdown content
        self.mount(Markdown(content, classes="content"))
        # Scroll after refresh to ensure content is rendered
        self.call_after_refresh(self._scroll_parent)

    def set_token_usage(self, prompt: int, completion: int, cached: int = 0) -> None:
        """Set token usage in border subtitle."""
        subtitle = f"↑{_humanize_tokens(prompt)} ↓{_humanize_tokens(completion)}"
        if cached:
            subtitle += f" ⚡{_humanize_tokens(cached)}"
        self.border_subtitle = subtitle

    def add_tooluse(self, tu: ToolUse) -> None:
        """Show animated indicator while tool is running."""
        try:
            # Remove current content
            content = self.query_one(".content")
            content.remove()
        except NoMatches:
            pass
        # Mount "Using" in blue wave, rest in regular text
        label = Golden("Using ", colors=BLUE)
        label.styles.width = "auto"
        label.styles.height = "auto"
        label.styles.margin = (0, 0, 0, 0)

        container = Horizontal(label, tu.to_widget(), classes="content")
        container.styles.height = "auto"
        container.styles.width = "100%"
        container.styles.margin = (0, 0, 0, 0)
        self.mount(container)
        self.call_after_refresh(self._scroll_parent)

    def show_thinking_animated(self, thinking_text: str) -> None:
        """Show animated purple 'Thinking:' label with regular text."""
        try:
            content = self.query_one(".content")
            content.remove()
        except NoMatches:
            pass
        text = Static(f"Thinking: {thinking_text}")
        text.styles.width = "1fr"
        text.styles.text_style = "italic"
        container = Horizontal(text, classes="content")
        container.styles.height = "auto"
        container.styles.width = "100%"
        self.mount(container)
        self.call_after_refresh(self._scroll_parent)

    def update_error(self, error: str) -> None:
        """Show error message in red."""
        self._content = f"Error: {error}"
        self._loading = False
        try:
            content = self.query_one(".content")
            content.remove()
        except NoMatches:
            pass
        # Use Static with Rich markup for colored error
        self.mount(Static(f"[red]Error: {error}[/red]", classes="content"))
        self.call_after_refresh(self._scroll_parent)


@dataclass
class ChatMessageData:
    """Data for a chat message."""

    role: Role
    content: str
    id: str | None = None  # Database message ID for persistence


class ChatMessage(Static):
    """A single chat message widget."""

    DEFAULT_CSS = """
    ChatMessage {
        width: 100%;
        padding: 1 2;
        margin: 0 0 1 0;
    }

    ChatMessage.user {
        background: $primary-darken-2;
    }

    ChatMessage.assistant {
        background: $surface;
    }

    ChatMessage.system {
        background: $warning-darken-3;
        color: $warning;
    }

    ChatMessage .message-role {
        text-style: bold;
        margin-bottom: 1;
    }

    ChatMessage .message-content {
        width: 100%;
    }
    """

    def __init__(
        self,
        role: Role,
        content: str,
        *,
        message_id: str | None = None,
        name: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=role)
        self.role = role
        self.content = content
        self.message_id = message_id  # Database message ID

    def compose(self) -> ComposeResult:
        role_display = self.role.capitalize()
        yield Static(f"[bold]{role_display}[/bold]", classes="message-role")
        yield Markdown(self.content, classes="message-content")

    def update_content(self, content: str) -> None:
        """Update the message content (for streaming)."""
        self.content = content
        try:
            markdown = self.query_one(".message-content", Markdown)
            markdown.update(content)
        except Exception:
            pass


class ChatView(Widget):
    """A chat view widget with message history and input."""

    DEFAULT_CSS = """
    ChatView {
        width: 100%;
        height: 100%;
        layout: vertical;
    }

    ChatView #chat-messages {
        height: 1fr;
        width: 100%;
        padding: 1;
    }

    ChatView #chat-input-container {
        height: auto;
        width: 100%;
        padding: 1;
        dock: bottom;
    }

    ChatView #chat-input {
        width: 100%;
    }
    """

    messages: reactive[list[ChatMessageData]] = reactive(list, init=False)

    class MessageSubmitted(Message):
        """Posted when a user submits a message."""

        def __init__(self, content: str) -> None:
            super().__init__()
            self.content = content

    def __init__(
        self,
        *,
        placeholder: str = "Type a message...",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.placeholder = placeholder
        self._messages: list[ChatMessageData] = []

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="chat-messages")
        with Vertical(id="chat-input-container"):
            yield Input(placeholder=self.placeholder, id="chat-input")

    @on(Input.Submitted, "#chat-input")
    def handle_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        content = event.value.strip()
        if content:
            event.input.clear()
            self.post_message(self.MessageSubmitted(content))

    def add_message(self, role: Role, content: str) -> ChatMessage:
        """Add a message to the chat view."""
        message_data = ChatMessageData(role=role, content=content)
        self._messages.append(message_data)

        container = self.query_one("#chat-messages", ScrollableContainer)
        message_widget = ChatMessage(role, content)
        container.mount(message_widget)
        message_widget.scroll_visible()

        return message_widget

    def get_messages(self) -> list[ChatMessageData]:
        """Get all messages as a list of ChatMessageData."""
        return list(self._messages)

    def clear_messages(self) -> None:
        """Clear all messages from the chat."""
        self._messages.clear()
        container = self.query_one("#chat-messages", ScrollableContainer)
        container.remove_children()

    def get_last_message_widget(self) -> ChatMessage | None:
        """Get the last message widget if it exists."""
        container = self.query_one("#chat-messages", ScrollableContainer)
        messages = container.query(ChatMessage)
        if messages:
            return messages.last()
        return None


class PersistentChatView(ChatView):
    """A ChatView with SQLite persistence for conversations."""

    class ConversationLoaded(Message):
        """Posted when a conversation is loaded."""

        def __init__(self, conversation_id: str) -> None:
            super().__init__()
            self.conversation_id = conversation_id

    class ConversationCreated(Message):
        """Posted when a new conversation is created."""

        def __init__(self, conversation_id: str) -> None:
            super().__init__()
            self.conversation_id = conversation_id

    def __init__(
        self,
        db: "ChatDatabase",
        *,
        conversation_id: str | None = None,
        placeholder: str = "Type a message...",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize a persistent chat view.

        Args:
            db: The ChatDatabase instance for persistence
            conversation_id: Optional existing conversation to load
            placeholder: Input placeholder text
            name: Widget name
            id: Widget ID
            classes: Widget classes
        """
        super().__init__(placeholder=placeholder, name=name, id=id, classes=classes)
        self.db = db
        self._conversation_id: str | None = conversation_id
        self._pending_load = conversation_id is not None

    @property
    def conversation_id(self) -> str | None:
        """Get the current conversation ID."""
        return self._conversation_id

    def on_mount(self) -> None:
        """Load conversation on mount if ID was provided."""
        if self._pending_load and self._conversation_id:
            self.load_conversation(self._conversation_id)
            self._pending_load = False

    def new_conversation(
        self, name: str | None = None, model: str | None = None
    ) -> str:
        """Start a new conversation.

        Args:
            name: Optional conversation name
            model: Optional model identifier

        Returns:
            The new conversation ID
        """
        self.clear_messages()
        conversation = self.db.create_conversation(name=name, model=model)
        self._conversation_id = conversation.id
        self.post_message(self.ConversationCreated(conversation.id))
        return conversation.id

    def load_conversation(self, conversation_id: str) -> bool:
        """Load an existing conversation.

        Args:
            conversation_id: The conversation ID to load

        Returns:
            True if loaded successfully, False if not found
        """
        conversation = self.db.get_conversation(conversation_id)
        if conversation is None:
            return False

        self.clear_messages()
        self._conversation_id = conversation_id

        # Load all messages into the view
        container = self.query_one("#chat-messages", ScrollableContainer)
        for msg in conversation.messages:
            message_data = ChatMessageData(
                role=msg.role, content=msg.content, id=msg.id
            )
            self._messages.append(message_data)
            message_widget = ChatMessage(msg.role, msg.content, message_id=msg.id)
            container.mount(message_widget)

        # Scroll to bottom
        if conversation.messages:
            container.scroll_end(animate=False)

        self.post_message(self.ConversationLoaded(conversation_id))
        return True

    def add_message(self, role: Role, content: str) -> ChatMessage:
        """Add a message and persist to database.

        Args:
            role: Message role
            content: Message content

        Returns:
            The ChatMessage widget
        """
        # Ensure we have a conversation
        if self._conversation_id is None:
            self.new_conversation()

        # Persist to database
        db_message = self.db.add_message(self._conversation_id, role, content)

        # Add to internal list
        message_data = ChatMessageData(role=role, content=content, id=db_message.id)
        self._messages.append(message_data)

        # Create and mount widget
        container = self.query_one("#chat-messages", ScrollableContainer)
        message_widget = ChatMessage(role, content, message_id=db_message.id)
        container.mount(message_widget)
        message_widget.scroll_visible()

        return message_widget

    def update_message(self, message_widget: ChatMessage, content: str) -> None:
        """Update a message's content and persist the change.

        Use this for streaming - call after streaming completes to save final content.

        Args:
            message_widget: The message widget to update
            content: The new content
        """
        message_widget.update_content(content)

        # Update in database if we have a message ID
        if message_widget.message_id:
            self.db.update_message(message_widget.message_id, content)

        # Update internal data
        for msg_data in self._messages:
            if msg_data.id == message_widget.message_id:
                msg_data.content = content
                break

    def delete_last_message(self) -> None:
        """Delete the last message from view and database."""
        if not self._messages:
            return

        msg_data = self._messages.pop()
        if msg_data.id:
            self.db.delete_message(msg_data.id)

        container = self.query_one("#chat-messages", ScrollableContainer)
        messages = container.query(ChatMessage)
        if messages:
            messages.last().remove()

    def get_conversation(self) -> "Conversation | None":
        """Get the current conversation with all messages from database."""
        if self._conversation_id is None:
            return None
        return self.db.get_conversation(self._conversation_id)

"""SQLite persistence for chat conversations."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    """A single chat message."""

    role: Role
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tokens: int | None = None


@dataclass
class Conversation:
    """A chat conversation with messages."""

    id: str
    name: str
    model: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    messages: list[Message] = field(default_factory=list)


class ChatDatabase:
    """SQLite database for persisting chat conversations."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        model TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        tokens INTEGER,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id);

    CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations(updated_at DESC);
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the database.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.textual-chat/chat.db
        """
        if db_path is None:
            db_dir = Path.home() / ".textual-chat"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "chat.db"

        self.db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        conn.executescript(self.SCHEMA)
        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys = ON")
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    # Conversation operations

    def create_conversation(
        self,
        name: str | None = None,
        model: str | None = None,
    ) -> Conversation:
        """Create a new conversation.

        Args:
            name: Optional name for the conversation
            model: Optional model identifier

        Returns:
            The created Conversation
        """
        conv_id = str(uuid.uuid4())
        now = datetime.utcnow()

        if name is None:
            name = f"Chat {now.strftime('%Y-%m-%d %H:%M')}"

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO conversations (id, name, model, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conv_id, name, model, now.isoformat(), now.isoformat()),
        )
        conn.commit()

        return Conversation(
            id=conv_id,
            name=name,
            model=model,
            created_at=now,
            updated_at=now,
        )

    def get_conversation(self, conv_id: str) -> Conversation | None:
        """Get a conversation by ID with all its messages.

        Args:
            conv_id: The conversation ID

        Returns:
            The Conversation or None if not found
        """
        conn = self._get_connection()

        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()

        if row is None:
            return None

        conversation = Conversation(
            id=row["id"],
            name=row["name"],
            model=row["model"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

        # Load messages
        message_rows = conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            """,
            (conv_id,),
        ).fetchall()

        for msg_row in message_rows:
            conversation.messages.append(
                Message(
                    id=msg_row["id"],
                    role=msg_row["role"],
                    content=msg_row["content"],
                    timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                    tokens=msg_row["tokens"],
                )
            )

        return conversation

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        """List conversations ordered by most recently updated.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip

        Returns:
            List of Conversations (without messages loaded)
        """
        conn = self._get_connection()

        rows = conn.execute(
            """
            SELECT * FROM conversations
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        return [
            Conversation(
                id=row["id"],
                name=row["name"],
                model=row["model"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def update_conversation(
        self,
        conv_id: str,
        name: str | None = None,
        model: str | None = None,
    ) -> None:
        """Update conversation metadata.

        Args:
            conv_id: The conversation ID
            name: New name (if provided)
            model: New model (if provided)
        """
        conn = self._get_connection()
        now = datetime.utcnow()

        updates = ["updated_at = ?"]
        params: list = [now.isoformat()]

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if model is not None:
            updates.append("model = ?")
            params.append(model)

        params.append(conv_id)

        conn.execute(
            f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()

    def delete_conversation(self, conv_id: str) -> None:
        """Delete a conversation and all its messages.

        Args:
            conv_id: The conversation ID
        """
        conn = self._get_connection()
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()

    # Message operations

    def add_message(
        self,
        conv_id: str,
        role: Role,
        content: str,
        tokens: int | None = None,
    ) -> Message:
        """Add a message to a conversation.

        Args:
            conv_id: The conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            tokens: Optional token count

        Returns:
            The created Message
        """
        msg_id = str(uuid.uuid4())
        now = datetime.utcnow()

        conn = self._get_connection()

        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, timestamp, tokens)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (msg_id, conv_id, role, content, now.isoformat(), tokens),
        )

        # Update conversation's updated_at
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now.isoformat(), conv_id),
        )

        conn.commit()

        return Message(
            id=msg_id,
            role=role,
            content=content,
            timestamp=now,
            tokens=tokens,
        )

    def update_message(
        self,
        msg_id: str,
        content: str,
        tokens: int | None = None,
    ) -> None:
        """Update a message's content (e.g., after streaming completes).

        Args:
            msg_id: The message ID
            content: New content
            tokens: Optional token count
        """
        conn = self._get_connection()

        if tokens is not None:
            conn.execute(
                "UPDATE messages SET content = ?, tokens = ? WHERE id = ?",
                (content, tokens, msg_id),
            )
        else:
            conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                (content, msg_id),
            )

        conn.commit()

    def delete_message(self, msg_id: str) -> None:
        """Delete a message.

        Args:
            msg_id: The message ID
        """
        conn = self._get_connection()
        conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
        conn.commit()

    def search_conversations(self, query: str, limit: int = 20) -> list[Conversation]:
        """Search conversations by message content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching Conversations
        """
        conn = self._get_connection()

        rows = conn.execute(
            """
            SELECT DISTINCT c.* FROM conversations c
            JOIN messages m ON c.id = m.conversation_id
            WHERE m.content LIKE ?
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (f"%{query}%", limit),
        ).fetchall()

        return [
            Conversation(
                id=row["id"],
                name=row["name"],
                model=row["model"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

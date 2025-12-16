"""Session storage for ACP conversations.

Maps working directories to ACP session IDs, enabling session forking
so agents can maintain context across multiple chat windows.

Also supports legacy agent_command-based storage for backwards compatibility.

Uses SQLite for persistence across app restarts.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Union

# JSON type for session data
JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]

log = logging.getLogger(__name__)


class SessionStorage:
    """Stores session IDs per working directory for reuse/forking.

    Uses SQLite database for persistence across app restarts.
    Supports both working directory-based and agent_command-based storage.
    """

    def __init__(self, db_path: Path | None = None, clear_on_init: bool = False):
        # Use XDG cache dir or fallback to home
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "textual-chat"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "acp_sessions.db"

        self.db_path = db_path
        self._init_db()

        # Clear stale sessions on initialization (app startup)
        # Sessions don't survive agent process restarts
        if clear_on_init:
            count = self.clear_all_sessions()
            if count > 0:
                log.info(f"Cleared {count} stale session(s) from previous app run")

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if old schema exists and drop it (schema migration)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            if cursor.fetchone():
                # Check if table has old schema (missing agent_command)
                cursor = conn.execute("PRAGMA table_info(sessions)")
                columns = [col[1] for col in cursor.fetchall()]
                if "agent_command" not in columns:
                    log.info("Migrating sessions table to new schema (adding agent_command)")
                    conn.execute("DROP TABLE sessions")

            # Working directory-based sessions (new API)
            # Each session is scoped to both working_dir AND agent_command
            # so OpenCode sessions don't get mixed with Claude Code sessions
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    working_dir TEXT NOT NULL,
                    agent_command TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (working_dir, agent_command)
                )
            """
            )
            # Agent command-based sessions (legacy API for chat.py)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    agent_command TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    messages TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()
        log.info(f"Session database initialized at: {self.db_path}")

    # ========== New API (working directory + agent command based) ==========

    def get_session_id(self, cwd: str | None, agent_command: str | None = None) -> str | None:
        """Get existing session ID for a working directory and agent.

        Args:
            cwd: Working directory path
            agent_command: Agent command (e.g., "claude-code-acp", "opencode acp")

        Returns:
            Session ID if one exists, None otherwise
        """
        if not cwd:
            log.warning("❌ get_session_id: cwd is None")
            return None

        if not agent_command:
            log.warning("❌ get_session_id: agent_command is None")
            return None

        # Normalize path
        normalized = str(Path(cwd).resolve())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT session_id FROM sessions WHERE working_dir = ? AND agent_command = ?",
                (normalized, agent_command),
            )
            row = cursor.fetchone()

        if row:
            session_id: str = row[0]
            log.warning(
                f"✅ Found existing session {session_id} for {normalized} + {agent_command}"
            )
            return session_id
        else:
            log.warning(f"❌ No existing session for {normalized} + {agent_command}")
            return None

    def delete_session_id(self, cwd: str | None, agent_command: str | None) -> None:
        """Delete stored session ID for a working directory and agent.

        Args:
            cwd: Working directory path
            agent_command: Agent command (e.g., "claude-code-acp", "opencode acp")
        """
        if not cwd or not agent_command:
            log.warning(f"❌ delete_session_id: missing cwd={cwd} or agent_command={agent_command}")
            return

        # Normalize path
        normalized = str(Path(cwd).resolve())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE working_dir = ? AND agent_command = ? RETURNING session_id",
                (normalized, agent_command),
            )
            row = cursor.fetchone()
            conn.commit()

        if row:
            log.warning(f"🗑️  Deleted session {row[0]} for {normalized} + {agent_command}")
        else:
            log.warning(f"⚠️  No session found to delete for {normalized} + {agent_command}")

    def store_session_id(self, cwd: str | None, agent_command: str | None, session_id: str) -> None:
        """Store a session ID for a working directory and agent.

        Args:
            cwd: Working directory path
            agent_command: Agent command (e.g., "claude-code-acp", "opencode acp")
            session_id: ACP session ID to store
        """
        if not cwd:
            log.warning(f"❌ store_session_id: cwd is None, cannot store {session_id}")
            return

        if not agent_command:
            log.warning(f"❌ store_session_id: agent_command is None, cannot store {session_id}")
            return

        # Normalize path
        normalized = str(Path(cwd).resolve())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (working_dir, agent_command, session_id, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(working_dir, agent_command) DO UPDATE SET
                    session_id = excluded.session_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (normalized, agent_command, session_id),
            )
            conn.commit()

        log.warning(f"💾 Stored session {session_id} for {normalized} + {agent_command}")

    # ========== Legacy API (agent_command-based, for chat.py) ==========

    def save_session(
        self,
        agent_command: str,
        session_id: str,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """Save a session ID and messages for a specific agent (legacy API).

        Args:
            agent_command: The agent command (used as key)
            session_id: The ACP session ID to persist
            messages: Optional list of messages to store
        """
        import json

        with sqlite3.connect(self.db_path) as conn:
            messages_json = json.dumps(messages) if messages else None
            conn.execute(
                """
                INSERT INTO agent_sessions (agent_command, session_id, messages, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(agent_command) DO UPDATE SET
                    session_id = excluded.session_id,
                    messages = excluded.messages,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (agent_command, session_id, messages_json),
            )
            conn.commit()

        log.debug(f"Saved session {session_id} for {agent_command}")

    def get_session(self, agent_command: str) -> dict[str, JSON] | None:
        """Get the last session data for a specific agent (legacy API).

        Args:
            agent_command: The agent command (used as key)

        Returns:
            Dict with "session_id" and optionally "messages" if found, None otherwise
        """
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT session_id, messages FROM agent_sessions WHERE agent_command = ?",
                (agent_command,),
            )
            row = cursor.fetchone()

        if row:
            session_id, messages_json = row
            result: dict[str, JSON] = {"session_id": session_id}
            if messages_json:
                try:
                    result["messages"] = json.loads(messages_json)
                except json.JSONDecodeError:
                    pass
            return result
        return None

    # ========== Common API ==========

    def clear_session(self, cwd_or_agent: str | None) -> None:
        """Clear stored session for a working directory or agent command.

        Args:
            cwd_or_agent: Working directory path or agent command
        """
        if not cwd_or_agent:
            return

        with sqlite3.connect(self.db_path) as conn:
            # Try both tables
            # Try as working directory
            try:
                normalized = str(Path(cwd_or_agent).resolve())
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE working_dir = ? RETURNING session_id",
                    (normalized,),
                )
                row = cursor.fetchone()
                if row:
                    log.info(f"Cleared session {row[0]} for working dir {normalized}")
            except:
                pass

            # Try as agent command
            cursor = conn.execute(
                "DELETE FROM agent_sessions WHERE agent_command = ? RETURNING session_id",
                (cwd_or_agent,),
            )
            row = cursor.fetchone()
            if row:
                log.info(f"Cleared session {row[0]} for agent {cwd_or_agent}")

            conn.commit()

    def clear_all_sessions(self) -> int:
        """Clear all stored sessions (useful on app startup).

        Returns:
            Number of sessions cleared
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            result = cursor.fetchone()
            count1: int = result[0] if result else 0

            cursor = conn.execute("SELECT COUNT(*) FROM agent_sessions")
            result = cursor.fetchone()
            count2: int = result[0] if result else 0

            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM agent_sessions")
            conn.commit()

        total = count1 + count2
        if total > 0:
            log.info(
                f"Cleared {total} sessions from database ({count1} cwd-based, {count2} agent-based)"
            )
        return total


# Global session storage instance
_storage = SessionStorage()


def get_session_storage() -> SessionStorage:
    """Get the global session storage instance."""
    return _storage

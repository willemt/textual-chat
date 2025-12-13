"""Session storage for persisting ACP session IDs and conversation history."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class SessionStorage:
    """Manages persistent storage of ACP session IDs and conversation history."""

    def __init__(self, storage_dir: Path | None = None):
        """Initialize session storage.

        Args:
            storage_dir: Directory to store session data.
                        Defaults to ~/.config/textual-chat/
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".config" / "textual-chat"

        self.storage_dir = storage_dir
        self.sessions_file = storage_dir / "sessions.json"

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_session(
        self, agent_command: str, session_id: str, messages: list[dict[str, str]] | None = None
    ) -> None:
        """Save a session ID and messages for a specific agent.

        Args:
            agent_command: The agent command (used as key)
            session_id: The ACP session ID to persist
            messages: Optional list of messages to store (format: [{"role": "user", "content": "..."}])
        """
        try:
            # Load existing sessions
            sessions = self._load_sessions()

            # Update with new session
            session_data = {
                "session_id": session_id,
                "agent_command": agent_command,
            }
            if messages is not None:
                session_data["messages"] = messages

            sessions[agent_command] = session_data

            # Write back to disk
            self.sessions_file.write_text(json.dumps(sessions, indent=2))
            log.debug(f"Saved session {session_id} for {agent_command}")

        except Exception as e:
            log.warning(f"Failed to save session: {e}")

    def get_session(self, agent_command: str) -> dict[str, Any] | None:
        """Get the last session data for a specific agent.

        Args:
            agent_command: The agent command (used as key)

        Returns:
            Dict with "session_id" and optionally "messages" if found, None otherwise
        """
        try:
            sessions = self._load_sessions()
            return sessions.get(agent_command)
        except Exception as e:
            log.warning(f"Failed to load session: {e}")
            return None

    def clear_session(self, agent_command: str) -> None:
        """Clear the saved session for a specific agent.

        Args:
            agent_command: The agent command (used as key)
        """
        try:
            sessions = self._load_sessions()
            if agent_command in sessions:
                del sessions[agent_command]
                self.sessions_file.write_text(json.dumps(sessions, indent=2))
                log.debug(f"Cleared session for {agent_command}")
        except Exception as e:
            log.warning(f"Failed to clear session: {e}")

    def _load_sessions(self) -> dict[str, Any]:
        """Load all sessions from disk.

        Returns:
            Dictionary mapping agent commands to session data
        """
        if not self.sessions_file.exists():
            return {}

        try:
            return json.loads(self.sessions_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Failed to load sessions file: {e}")
            return {}

"""Singleton agent manager for sharing ACP connections across chat windows.

This allows sessions to be forked within the same agent process,
enabling context sharing across multiple chat windows.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from acp import PROTOCOL_VERSION, connect_to_agent
from acp.client.connection import ClientSideConnection
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation, InitializeResponse

if TYPE_CHECKING:
    from .llm_adapter_acp import ACPClientHandler

log = logging.getLogger(__name__)


class SharedAgentConnection:
    """Shared connection to an ACP agent process."""

    def __init__(self, agent_command: str):
        self.agent_command = agent_command
        self._conn: ClientSideConnection | None = None
        self._proc: aio_subprocess.Process | None = None
        self._client: ACPClientHandler | None = None
        self.init_response: InitializeResponse | None = None
        self.agent_name: str | None = None
        self._lock = asyncio.Lock()

    async def ensure_connected(self) -> ClientSideConnection:
        """Ensure connection is established and return the connection object."""
        async with self._lock:
            if self._conn is not None:
                return self._conn

            log.info(f"Spawning new agent process: {self.agent_command}")

            # Parse agent command
            parts = self.agent_command.split()
            program = parts[0]
            args = parts[1:] if len(parts) > 1 else []

            program_path = Path(program)
            spawn_program = program
            spawn_args = args

            # If it's a Python file, run with interpreter
            if program_path.exists() and not os.access(program_path, os.X_OK):
                spawn_program = sys.executable
                spawn_args = [str(program_path), *args]

            # Spawn agent process
            self._proc = await asyncio.create_subprocess_exec(
                spawn_program,
                *spawn_args,
                stdin=aio_subprocess.PIPE,
                stdout=aio_subprocess.PIPE,
                stderr=aio_subprocess.PIPE,  # Capture stderr to see errors
            )

            if self._proc.stdin is None or self._proc.stdout is None:
                raise RuntimeError("Agent process does not expose stdio pipes")

            # Start background task to log stderr
            if self._proc.stderr:
                asyncio.create_task(self._log_stderr())

            # Import here to avoid circular dependency
            from .llm_adapter_acp import ACPClientHandler

            # Create client handler
            self._client = ACPClientHandler()

            # Increase buffer limit to handle large messages (e.g., session history)
            # Default is 64KB, we increase to 10MB to handle large session restores
            if self._proc.stdout:
                setattr(self._proc.stdout, "_limit", 10 * 1024 * 1024)

            # Connect to agent
            self._conn = connect_to_agent(self._client, self._proc.stdin, self._proc.stdout)

            # Initialize
            self.init_response = await self._conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(
                    fs=FileSystemCapability(
                        read_text_file=True,
                        write_text_file=True,
                    ),
                    terminal=True,
                ),
                client_info=Implementation(
                    name="textual-chat", title="Textual Chat", version="0.1.0"
                ),
            )

            # Log initialize response to check capabilities
            log.info(f"📋 ACP Initialize response: {self.init_response}")

            # Check if loadSession is supported
            if hasattr(self.init_response, "agent_capabilities"):
                agent_caps = self.init_response.agent_capabilities
                if agent_caps and hasattr(agent_caps, "load_session"):
                    log.info(f"   ✅ loadSession capability: {agent_caps.load_session}")
                else:
                    log.info("   ❌ loadSession capability: not present")

            # Extract agent name
            if hasattr(self.init_response, "agent_info"):
                agent_info = self.init_response.agent_info
                if agent_info:
                    if hasattr(agent_info, "title") and agent_info.title:
                        self.agent_name = agent_info.title
                    elif hasattr(agent_info, "name") and agent_info.name:
                        self.agent_name = agent_info.name

            log.info(f"Agent connected: {self.agent_name}")
            assert self._conn is not None, "Connection must be established"
            return self._conn

    async def _log_stderr(self) -> None:
        """Read and log stderr from the agent process."""
        if not self._proc or not self._proc.stderr:
            return

        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                decoded = line.decode().strip()
                if decoded:
                    log.warning(f"[{self.agent_command} stderr] {decoded}")
        except Exception as e:
            log.debug(f"Error reading stderr: {e}")

    async def close(self) -> None:
        """Close the connection and terminate the agent process."""
        async with self._lock:
            if self._proc and self._proc.returncode is None:
                self._proc.terminate()
                try:
                    await self._proc.wait()
                except ProcessLookupError:
                    pass
            self._conn = None
            self._proc = None


class AgentManager:
    """Manages shared agent connections per command."""

    def __init__(self) -> None:
        # Map agent_command -> SharedAgentConnection
        self._connections: dict[str, SharedAgentConnection] = {}
        self._instance_id = id(self)
        log.info(f"🆕 AgentManager instance created: {hex(self._instance_id)}")

    def get_connection(self, agent_command: str) -> SharedAgentConnection:
        """Get or create a shared agent connection.

        Args:
            agent_command: Command to spawn the agent

        Returns:
            Shared connection instance
        """
        log.info(f"📞 get_connection called on AgentManager instance: {hex(self._instance_id)}")
        if agent_command not in self._connections:
            log.info(f"Creating new shared connection for: {agent_command}")
            log.info(f"   Existing connections: {list(self._connections.keys())}")
            self._connections[agent_command] = SharedAgentConnection(agent_command)
        else:
            log.info(f"♻️  Reusing existing connection for: {agent_command}")
        return self._connections[agent_command]

    async def close_all(self) -> None:
        """Close all agent connections."""
        for agent_command, connection in self._connections.items():
            log.info(f"Closing agent: {agent_command}")
            await connection.close()
        self._connections.clear()


# Global agent manager instance
_manager: AgentManager | None = None


def get_agent_manager() -> AgentManager:
    """Get the global agent manager instance."""
    global _manager
    if _manager is None:
        log.info("🌍 Creating global AgentManager singleton")
        _manager = AgentManager()
    else:
        log.info(f"📌 Returning existing AgentManager singleton: {hex(id(_manager))}")
    return _manager

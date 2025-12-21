"""Slash command management system with decorator support."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat import Chat

# Type alias for command handlers
CommandHandler = Callable[["Chat"], Awaitable[None]]


@dataclass
class SlashCommand:
    """A slash command definition."""

    name: str
    """The command name (without the leading /)."""

    description: str
    """A description of what the command does."""

    handler: CommandHandler | None = None
    """Async handler function for the command. Receives the Chat instance."""

    hidden: bool = False
    """If True, command won't appear in /help but still works."""


@dataclass
class SlashCommandManager:
    """Manages slash commands - registration, lookup, and execution."""

    _commands: dict[str, SlashCommand] = field(default_factory=dict)

    def add(self, command: SlashCommand) -> None:
        """Add a slash command."""
        self._commands[command.name.lower()] = command

    def remove(self, name: str) -> bool:
        """Remove a slash command by name. Returns True if removed."""
        return self._commands.pop(name.lower(), None) is not None

    def get(self, name: str) -> SlashCommand | None:
        """Get a slash command by name."""
        return self._commands.get(name.lower())

    def all(self, include_hidden: bool = False) -> list[SlashCommand]:
        """Get all registered commands, optionally including hidden ones."""
        commands = [c for c in self._commands.values()]
        if not include_hidden:
            commands = [cmd for cmd in commands if not cmd.hidden]
        return sorted(commands, key=lambda c: c.name)

    def filter(self, predicate: Callable[[SlashCommand], bool]) -> list[SlashCommand]:
        """Filter commands by a predicate function."""
        return [cmd for cmd in self._commands.values() if predicate(cmd)]

    # Alias for backwards compatibility
    def list(self, include_hidden: bool = False) -> list[SlashCommand]:
        """Alias for all()."""
        return self.all(include_hidden=include_hidden)

    async def execute(self, name: str, chat: Chat) -> bool:
        """Execute a slash command. Returns True if command was found and executed."""
        command = self.get(name)
        if command and command.handler:
            await command.handler(chat)
            return True
        return False

    def help_text(self) -> str:
        """Generate help text for all visible commands."""
        lines = ["**Available slash commands:**", ""]
        for cmd in self.all(include_hidden=False):
            lines.append(f"- `/{cmd.name}` - {cmd.description}")
        return "\n".join(lines)

    def slash_command(
        self,
        name_or_func: CommandHandler | str | None = None,
        *,
        description: str | None = None,
        hidden: bool = False,
    ) -> CommandHandler | Callable[[CommandHandler], CommandHandler]:
        """Decorator to register a slash command.

        Usage:
            @manager.slash_command
            async def help(chat: Chat) -> None:
                '''Show available commands.'''
                ...

            @manager.slash_command("quit", description="Exit the app")
            async def exit_handler(chat: Chat) -> None:
                ...
        """
        name: str | None = None

        def decorator(fn: CommandHandler) -> CommandHandler:
            cmd_name = name if name else fn.__name__
            cmd_description = description if description else (fn.__doc__ or "").strip()
            self.add(
                SlashCommand(
                    name=cmd_name,
                    description=cmd_description,
                    handler=fn,
                    hidden=hidden,
                )
            )
            return fn

        # Called as @slash_command (no parens) - func is the decorated function
        if callable(name_or_func):
            return decorator(name_or_func)

        # Called as @slash_command("name") or @slash_command(name="name")
        if isinstance(name_or_func, str):
            name = name_or_func

        return decorator


# =============================================================================
# Default manager with built-in commands
# =============================================================================

_default_manager = SlashCommandManager()


@_default_manager.slash_command
async def help(chat: Chat) -> None:
    """Show available slash commands."""
    help_text = chat.slash_commands.help_text()
    chat._add_message("system", help_text)


@_default_manager.slash_command
async def model(chat: Chat) -> None:
    """Select a different LLM model."""
    if "llm_adapter_acp" in chat._adapter.__name__:
        chat.notify(
            "Model selection not available for ACP adapter. Use /agent instead.",
            severity="warning",
        )
        return
    if not chat.show_model_selector:
        chat.notify("Model selection is disabled.", severity="warning")
        return
    chat.action_select_model()


@_default_manager.slash_command
async def agent(chat: Chat) -> None:
    """Select a different ACP agent."""
    if "llm_adapter_acp" not in chat._adapter.__name__:
        chat.notify(
            "Agent selection only available for ACP adapter. Use /model instead.",
            severity="warning",
        )
        return
    chat.action_select_agent()


@_default_manager.slash_command
async def quit(chat: Chat) -> None:
    """Exit the application."""
    chat.app.exit()


def create_default_manager() -> SlashCommandManager:
    """Create a new SlashCommandManager with all built-in commands registered."""
    manager = SlashCommandManager()
    # Copy commands from the default manager
    for cmd in _default_manager.all(include_hidden=True):
        manager.add(cmd)
    return manager

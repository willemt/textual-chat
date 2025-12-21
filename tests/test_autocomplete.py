"""Tests for slash command autocomplete implementation."""

from textual.widgets import TextArea

from textual_chat import Chat
from textual_chat.widgets import SlashCommand, SlashCommandAutocomplete


class TestSlashCommand:
    """Tests for SlashCommand dataclass."""

    def test_create_slash_command(self) -> None:
        """Can create a SlashCommand instance."""
        cmd = SlashCommand(
            name="help",
            description="Show available commands",
        )
        assert cmd.name == "help"
        assert cmd.description == "Show available commands"

    def test_slash_command_str(self) -> None:
        """SlashCommand has sensible string representation."""
        cmd = SlashCommand(name="model", description="Select model")
        assert "model" in str(cmd) or "SlashCommand" in str(cmd)


class TestSlashCommandAutocomplete:
    """Tests for SlashCommandAutocomplete widget."""

    def test_import(self) -> None:
        """SlashCommandAutocomplete can be imported."""
        assert SlashCommandAutocomplete is not None

    def test_signature(self) -> None:
        """SlashCommandAutocomplete accepts target and commands parameters."""
        text_area = TextArea(id="test-input")
        commands = [
            SlashCommand(name="help", description="Show help"),
            SlashCommand(name="model", description="Select model"),
        ]

        # This may fail without app context, but validates the signature
        try:
            autocomplete = SlashCommandAutocomplete(
                target=text_area,
                commands=commands,
            )
            assert autocomplete is not None
        except Exception as e:
            # Expected to fail without app context
            # Just verify it's not a signature error
            assert "unexpected keyword argument" not in str(e).lower()


class TestChatSlashCommands:
    """Tests for Chat widget slash command integration."""

    def test_chat_has_slash_commands_manager(self) -> None:
        """Chat widget has slash_commands manager."""
        chat = Chat()
        assert hasattr(chat, "slash_commands")

    def test_slash_commands_returns_list(self) -> None:
        """slash_commands.list() returns a list of SlashCommand objects."""
        chat = Chat()
        commands = chat.slash_commands.list()
        assert isinstance(commands, list)
        assert len(commands) > 0
        assert all(isinstance(cmd, SlashCommand) for cmd in commands)

    def test_slash_commands_have_required_fields(self) -> None:
        """Each slash command has name and description."""
        chat = Chat()
        commands = chat.slash_commands.list()
        for cmd in commands:
            assert cmd.name, f"Command missing name: {cmd}"
            assert cmd.description, f"Command missing description: {cmd}"

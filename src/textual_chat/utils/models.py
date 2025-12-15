"""Model and agent discovery utilities."""

import os


def get_available_models() -> list[tuple[str, str, str]]:
    """Get list of available models as (display_name, model_id, provider) tuples."""
    models = []

    # Anthropic models
    if os.getenv("ANTHROPIC_API_KEY"):
        models.extend(
            [
                ("Claude Sonnet 4", "claude-sonnet-4-20250514", "Anthropic"),
                ("Claude Opus 4", "claude-opus-4-20250514", "Anthropic"),
                ("Claude Haiku 3.5", "claude-3-5-haiku-latest", "Anthropic"),
            ]
        )

    # OpenAI models
    if os.getenv("OPENAI_API_KEY"):
        models.extend(
            [
                ("GPT-4o", "gpt-4o", "OpenAI"),
                ("GPT-4o Mini", "gpt-4o-mini", "OpenAI"),
                ("GPT-4 Turbo", "gpt-4-turbo", "OpenAI"),
                ("o1", "o1", "OpenAI"),
                ("o1-mini", "o1-mini", "OpenAI"),
            ]
        )

    # GitHub Models
    if os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_KEY"):
        models.extend(
            [
                ("GPT-4o (GitHub)", "github/gpt-4o", "GitHub Models"),
                ("GPT-4o Mini (GitHub)", "github/gpt-4o-mini", "GitHub Models"),
                ("Claude Sonnet 3.5 (GitHub)", "github/claude-3.5-sonnet", "GitHub Models"),
            ]
        )

    # Google models
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        models.extend(
            [
                ("Gemini 1.5 Flash", "gemini/gemini-1.5-flash", "Google"),
                ("Gemini 1.5 Pro", "gemini/gemini-1.5-pro", "Google"),
                ("Gemini 2.0 Flash", "gemini/gemini-2.0-flash-exp", "Google"),
            ]
        )

    # Groq models
    if os.getenv("GROQ_API_KEY"):
        models.extend(
            [
                ("Llama 3.1 8B (Groq)", "groq/llama-3.1-8b-instant", "Groq"),
                ("Llama 3.1 70B (Groq)", "groq/llama-3.1-70b-versatile", "Groq"),
                ("Mixtral 8x7B (Groq)", "groq/mixtral-8x7b-32768", "Groq"),
            ]
        )

    # DeepSeek models
    if os.getenv("DEEPSEEK_API_KEY"):
        models.extend(
            [
                ("DeepSeek Chat", "deepseek/deepseek-chat", "DeepSeek"),
                ("DeepSeek Coder", "deepseek/deepseek-coder", "DeepSeek"),
            ]
        )

    return models


def get_available_agents() -> list[tuple[str, str, str]]:
    """Get list of available ACP agents as (display_name, agent_command, description) tuples."""
    agents = [
        (
            "Claude Code",
            "claude-code-acp",
            "Anthropic's official CLI agent with web search, bash, and file tools",
        ),
        (
            "OpenCode",
            "opencode acp",
            "Open-source alternative agent with similar capabilities",
        ),
    ]
    return agents

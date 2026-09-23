"""The session-scoped policy URL an agent function hands to its agent."""


def openai_session_url(base_url: str) -> str:
    """The OpenAI-compatible API root of a session: its base URL plus the ``/v1`` suffix."""
    return f"{base_url}/v1"

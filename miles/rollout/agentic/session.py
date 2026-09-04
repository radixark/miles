"""The session-scoped policy URL an agent function hands to its agent."""


def openai_session_url(base_url: str) -> str:
    """The OpenAI-compatible API root of a session: its base URL plus the ``/v1``
    suffix OpenAI-style clients expect.

    ``base_url`` already names the session server as the agent reaches it (see
    ``OpenAIEndpointTracer.agent_base_url``), so nothing is rewritten here. Shared
    by the agent-function legs so the suffix cannot drift between them.
    """
    return f"{base_url}/v1"

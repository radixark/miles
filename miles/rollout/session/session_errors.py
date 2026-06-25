"""Error types for the session module.

Hierarchy
---------
SessionError (base)
├── SessionNotFoundError       → 404  session does not exist
├── MessageValidationError     → 400  messages structure/content invalid
├── TokenizationError          → 500  TITO tokenizer / prefix mismatch
├── UpstreamResponseError      → 502  SGLang response invalid or unexpected
├── SessionBusyError           → 409  session already has an in-flight chat
└── SessionInvariantError      → 500  unreachable session-state invariant violated
"""


class SessionError(Exception):
    """Base class for all session-related errors."""

    status_code: int = 500


class SessionNotFoundError(SessionError):
    """Raised when the requested session ID does not exist."""

    status_code: int = 404


class MessageValidationError(SessionError):
    """Raised when request messages fail structural validation.

    Examples: user message after assistant, messages not append-only,
    rollback failed (no assistant checkpoint in matched prefix).
    """

    status_code: int = 400


class TokenizationError(SessionError):
    """Raised when TITO tokenization invariants are violated.

    Examples: pretokenized prefix mismatch between stored and new token IDs.
    """

    status_code: int = 500


class UpstreamResponseError(SessionError):
    """Raised when the upstream SGLang response is invalid or unexpected.

    Examples: missing meta_info, assistant content is None,
    output_token_logprobs length mismatch.
    """

    status_code: int = 502


class SessionBusyError(SessionError):
    """Raised when the session already has an in-flight chat completion.

    One linear trajectory admits one in-flight chat at a time.
    """

    status_code: int = 409


class SessionInvariantError(SessionError):
    """Raised when a session-state invariant that should be unreachable under
    the in-flight gate is violated (defensive; indicates a real bug).
    """

    status_code: int = 500

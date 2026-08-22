"""Error types for the session module.

Hierarchy
---------
SessionError (base)
├── SessionNotFoundError       → 404  session does not exist
├── SessionReclaimedError      → 410  session existed and was released by the trainer
├── MessageValidationError     → 400  messages structure/content invalid
├── TruncatedGenerationError   → 409  extending a length-truncated generation (v2)
├── TokenizationError          → 500  TITO tokenizer / prefix mismatch
└── UpstreamResponseError      → 502  SGLang response invalid or unexpected
"""


class SessionError(Exception):
    """Base class for all session-related errors."""

    status_code: int = 500


class SessionNotFoundError(SessionError):
    """Raised when the requested session ID does not exist."""

    status_code: int = 404


class SessionReclaimedError(SessionError):
    """Raised when the session existed but the trainer already released it.

    This is the "your run was abandoned" signal, distinct from the 404 a
    never-existent session ID gets. It happens when the trainer stops waiting
    for an agent -- for example the agent-server call exceeded
    ``AGENT_TRIAL_TIMEOUT`` -- collects whatever samples exist and deletes the
    session, while the agent is still running and still issuing requests.

    410 rather than 404 so the caller can tell "the trainer moved on without
    me" apart from a genuine bad-ID bug or a model-side fault; both otherwise
    surface through OpenAI-compatible clients as an indistinguishable
    NotFoundError.
    """

    status_code: int = 410


def reclaimed_error(session_id: str) -> SessionReclaimedError:
    """Build the canonical "the trainer moved on without you" error."""
    return SessionReclaimedError(
        f"session reclaimed: session_id={session_id} "
        "(the trainer released this session and is no longer waiting for the agent)"
    )


class MessageValidationError(SessionError):
    """Raised when request messages fail structural validation.

    Examples: user message after assistant, messages not append-only,
    rollback failed (no assistant checkpoint in matched prefix).
    """

    status_code: int = 400


class TruncatedGenerationError(SessionError):
    """Raised when a request extends a generation that ended with
    finish_reason='length'; only the v2 tree server raises it (v1 never
    branches)."""

    status_code: int = 409


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

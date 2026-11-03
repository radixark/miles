"""Error types for the session module.

Hierarchy
---------
SessionError (base)
├── SessionNotFoundError       → 404  session does not exist
├── MessageValidationError     → 400  messages structure/content invalid
├── TruncatedGenerationError   → 409  extending a length-truncated generation (v2)
├── TokenizationError          → 500  TITO tokenizer / prefix mismatch
├── UpstreamResponseError      → 502  SGLang response invalid or unexpected
├── UpstreamGenerationAbortedError → 503  SGLang aborted the generation
└── OutputStoreContractError   → 424  output store contract broken; the rollout stops
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


class UpstreamGenerationAbortedError(SessionError):
    """Raised when upstream aborts a generation without an HTTP error."""

    status_code: int = 503


class OutputStoreContractError(SessionError):
    """Raised for a ``StorePlanError``: the output store contract is broken, so retrying cannot fix it.

    The session client raises ``StorePlanError`` again on this status, and the rollout stops.
    """

    status_code: int = 424

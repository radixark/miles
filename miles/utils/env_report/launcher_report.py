import base64
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def decode_env_report(raw: str) -> dict[str, Any] | None:
    """Decode an env report string (base64-encoded JSON or raw JSON)."""
    if not raw:
        return None
    try:
        decoded = base64.b64decode(raw).decode()
        return json.loads(decoded)
    except Exception:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse env report", exc_info=True)
            return None

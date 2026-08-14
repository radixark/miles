import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
LAUNCHER_REPORT_ENV_VAR = "MILES_SCRIPT_ENV_REPORT"


def read_launcher_report(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        return _as_report_object(json.loads(Path(path).read_text()))
    except Exception:
        logger.warning("Failed to read the launcher report at %s", path, exc_info=True)
        return None


def _as_report_object(decoded: Any) -> dict[str, Any] | None:
    if isinstance(decoded, dict):
        return decoded
    logger.warning("The launcher report is valid json but not an object, so it names nothing: %r", decoded)
    return None

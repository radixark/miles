from __future__ import annotations

import base64
import pickle
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer


PICKLED_HATCH_MARKER = "MILES_PICKLED_HATCH"

PICKLED_TAG = "__miles_pickled__"


def _from_pickled(value: Any) -> Any:
    if not isinstance(value, dict) or (encoded := value.get(PICKLED_TAG)) is None:
        return value
    return pickle.loads(base64.b64decode(encoded))


def _to_pickled(value: Any) -> dict[str, str]:
    return {PICKLED_TAG: base64.b64encode(pickle.dumps(value)).decode()}


# TODO(MILES_PICKLED_HATCH): temporary, whitelisted escape hatch for the argparse Namespace a
# trainer is built from, which no wire type reproduces losslessly. Reclaim it once the
# arguments subsystem is split into wire-typed pieces; every other parameter must stay
# strictly wire-typed. Until then this unpickles whatever reaches the worker's rpc port, which
# assumes the run's network is trusted; it is the reason to reclaim it, not a property to keep.
Pickled = Annotated[
    Any,
    BeforeValidator(_from_pickled),
    PlainSerializer(_to_pickled, return_type=dict[str, str]),
]

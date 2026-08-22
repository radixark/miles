from __future__ import annotations

import argparse
import functools
import sys
from typing import Any
from unittest.mock import patch

from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.run_uuid import RUN_UUID_LENGTH
from miles.utils.workers.argv_utils import with_relax_parser_required_args

# megatron's own parser adds these and miles' code reads them, but a unit test builds only the miles
# extras, so nothing else would put them on the namespace
_TRAIN_BACKEND_DEFAULTS: dict[str, Any] = dict(
    disable_param_buffers_cpu_backup=False,
    lr_warmup_iters=None,
    load=None,
)

# declared with no default and resolved after parsing, so the raw parser value is one no production
# code has ever seen; these are what the resolution settles on for a plain single-deployment run
_RESOLVED_AFTER_PARSING: dict[str, Any] = dict(
    offload_train=False,
    offload_rollout=False,
    run_uuid="0" * RUN_UUID_LENGTH,
)


@functools.cache
def parser_defaults() -> dict[str, Any]:
    # a hand-written defaults dict falls behind the moment production reads a new argument, and the
    # test then dies on AttributeError somewhere deep instead of on the thing it was written for
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    with with_relax_parser_required_args(parser), patch.object(sys, "argv", ["test"]):
        parsed, _ = parser.parse_known_args([])
    return {**_TRAIN_BACKEND_DEFAULTS, **vars(parsed), **_RESOLVED_AFTER_PARSING}

import random
from argparse import Namespace
from copy import deepcopy
from typing import Any

from miles.utils.types import Sample

from miles.rollout.sglang_rollout import generate as _generate_base


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    # TODO: make it configurable after we have better arg parser
    min_osl = 40 * 1024
    max_osl = 64 * 1024

    modified_sampling_params = deepcopy(sampling_params)
    modified_sampling_params["ignore_eos"] = True
    modified_sampling_params["max_new_tokens"] = random.randrange(min_osl, max_osl)

    return await _generate_base(args, sample, modified_sampling_params)

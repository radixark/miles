from argparse import Namespace
from copy import deepcopy
from typing import Any

from miles.utils.types import Sample

from miles.rollout.sglang_rollout import generate as _generate_base


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    modified_sampling_params = deepcopy(sampling_params)
    TODO
    return await _generate_base(args, sample, modified_sampling_params)

import asyncio
import random

import aiohttp

from miles.utils.misc import load_function
from miles.utils.types import Sample

from .ocr import ocr_rm


def _resolve_rm_type(args, sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return (metadata.get("rm_type") or args.rm_type or "").strip()


async def remote_rm(args, generated_output, prompt: str):
    payload = {
        "prompt": prompt,
        "generated_output": generated_output,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    rm_type = _resolve_rm_type(args, sample)

    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type == "ocr":
        return await ocr_rm(args, sample)
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)

    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards

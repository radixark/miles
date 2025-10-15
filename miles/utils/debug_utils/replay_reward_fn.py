import asyncio
from typing import Annotated

import ray
import torch
import typer

from miles.utils.misc import load_function


def main(
    rollout_data_path: Annotated[str, typer.Option()],
    custom_rm_path: Annotated[str, typer.Option()],
):
    if not ray.is_initialized():
        ray.init()

    pack = torch.load(rollout_data_path)
    asyncio.run(_main_async(samples=pack["samples"], custom_rm_path=custom_rm_path))


async def _main_async(samples, custom_rm_path):
    rm_function = load_function(custom_rm_path)
    rewards = await asyncio.gather(*[rm_function(None, sample) for sample in samples])

    # TODO improve output
    for sample, reward in zip(samples, rewards, strict=True):
        print(f"recomputed_reward={reward} sample={sample}")


if __name__ == "__main__":
    typer.run(main)

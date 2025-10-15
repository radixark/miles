from typing import Annotated

import torch
import typer


def main(
    rollout_data_path: Annotated[str, typer.Option()],
):
    pack = torch.load(rollout_data_path)
    samples = pack["samples"]
    TODO


if __name__ == '__main__':
    typer.run(main)

"""Rollout data for the Step5 expansion tests: three samples, two with images, one text-only."""

import torch

from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.types import STEP5_EXPANDED_PROMPT_KEY

IMAGE = 7
BOS = 0
# What an image sample hands the Step5 VL model once expanded; a text sample carries none of them.
MODEL_INPUT_KEYS = ("step5_pixel_values", "step5_patch_pixel_values", "step5_num_patches", "step5_image_positions")


def set_trivial_parallel_state(cp_size: int = 1) -> None:
    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=GroupInfo(rank=0, size=cp_size, group=None),
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )


def image_inputs(expanded_prompt: list[int], num_patches: list[int], device: str) -> dict[str, torch.Tensor]:
    expanded = torch.tensor(expanded_prompt, device=device)
    return {
        "step5_pixel_values": torch.randn(len(num_patches), 3, 4, 4, device=device),
        "step5_patch_pixel_values": torch.randn(sum(num_patches), 3, 2, 2, device=device),
        "step5_num_patches": torch.tensor(num_patches, device=device),
        "step5_image_positions": (expanded == IMAGE).nonzero().flatten(),
        STEP5_EXPANDED_PROMPT_KEY: expanded,
    }


def rollout_data(device: str = "cpu") -> dict:
    # Request prompts carry one image token per image; the expanded prompts are what SGLang tokenized.
    prompts = [[5, IMAGE, 8], [5, 6, 8], [9, IMAGE, IMAGE]]
    responses = [[20, 21, 22], [23, 24], [25]]
    multimodal_train_inputs = [
        image_inputs([BOS, 5, 30, IMAGE, IMAGE, IMAGE, 31, 8], [0], device),
        None,
        image_inputs([BOS, 9, 32, IMAGE, IMAGE, 33, 30, IMAGE, 31, 30, IMAGE, IMAGE, 31], [1, 0], device),
    ]
    tokens = [torch.tensor(p + r, device=device) for p, r in zip(prompts, responses, strict=True)]
    return {
        "tokens": tokens,
        "loss_masks": [torch.ones(len(r), dtype=torch.int, device=device) for r in responses],
        "total_lengths": [len(t) for t in tokens],
        "response_lengths": [len(r) for r in responses],
        "multimodal_train_inputs": multimodal_train_inputs,
    }

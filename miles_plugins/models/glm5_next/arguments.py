"""Launch arguments for GLM-5.3 image SFT.

Deliberately free of heavy imports. `miles.utils.arguments` reaches this module while
building the parser for *every* Megatron run, so pulling the visual tower in from here
would make every model pay for -- and depend on -- the Transformers GLM-OCR modeling code.
`vision.py` imports the scope names from here rather than the other way round.
"""

from argparse import ArgumentParser

# Trainable scopes, ordered from smallest to largest.
VISION_TRAINABLE_CHOICES = ("none", "projector", "full")


def vision_scope_may_register_tower(args) -> bool:
    """Whether the requested scope puts the visual tower in the module tree.

    Lets callers on shared code paths decide whether to import the tower at all.
    """
    return getattr(args, "glm5_vision_trainable", "none") != "none"


def add_glm5_next_vision_arguments(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group(title="glm5-next-vision")
    group.add_argument(
        "--glm5-vision-trainable",
        choices=list(VISION_TRAINABLE_CHOICES),
        default="none",
        help=(
            "Which GLM-5.3 visual parameters image SFT trains. 'none' keeps the tower frozen and "
            "outside the optimizer and checkpoint; 'projector' trains the image-feature projection "
            "(post_layernorm, downsample, merger); 'full' also trains patch_embed and the ViT blocks. "
            "Any scope other than 'none' writes visual.* into the training checkpoint."
        ),
    )
    group.add_argument(
        "--glm5-vision-recompute",
        action="store_true",
        default=False,
        help=(
            "Recompute the ViT blocks' activations in the backward pass instead of storing them. "
            "Only affects --glm5-vision-trainable full, which is the scope that trains those "
            "blocks; trades compute for the activation memory the full tower would otherwise hold."
        ),
    )
    group.add_argument(
        "--glm5-freeze-language",
        action="store_true",
        default=False,
        help=(
            "Freeze all GLM-5.3 language parameters before DDP and optimizer construction. "
            "Requires a trainable visual scope. Autograd still traverses the language model "
            "to compute gradients for the image embeddings."
        ),
    )
    return parser

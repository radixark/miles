"""Per-arch adaptation specs — the one place a new architecture plugs into the FSDP backend.

Importing this package registers every arch's hooks; the mechanism modules never change. To add an
arch, create ``specs/<arch>.py`` registering only the hooks it needs and add it to the import below:

  * register_param_transform    [weight_bridge]    — train->rollout param rename/reshape at weight sync
  * register_model_patch        [class_patches]    — config-time patch of transformers classes
  * register_packing_patch      [packing.registry] — per-document state reset under THD packing
  * register_post_load_fixup    [post_load_fixups] — correct weights from_pretrained clobbered

An arch that needs none of these registers nothing. See the existing specs for examples.
"""

from . import glm4_moe_lite, nemotron_h, qwen3_5_moe, qwen3_moe  # noqa: F401  (imports trigger registration)

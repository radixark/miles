"""A naming stand-in for unit tests that attach specs to bare blocks instead of built layers."""

import dataclasses

from miles.backends.megatron_utils.lora.native.spec.base import AttachContext


class FixedPrefixNaming:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def block_prefix(self, layer, host_weight, leaves) -> str:
        del layer, host_weight, leaves
        return self.prefix


def named(context: AttachContext, prefix: str) -> AttachContext:
    """``context`` naming every block it attaches ``prefix``."""
    return dataclasses.replace(context, naming=FixedPrefixNaming(prefix))

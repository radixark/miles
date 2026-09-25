import logging

logger = logging.getLogger(__name__)


def strip_param_name_prefix(name: str | None) -> str | None:
    if name is None:
        return None
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name


def maybe_hide_te_flash_attn_4(args) -> None:
    """Make Transformer Engine pick FlashAttention 2 over 4 (see --te-disable-flash-attn-4).

    Must run in the training process before the first attention forward; TE reads this flag on every
    backend selection.
    """
    if not getattr(args, "te_disable_flash_attn_4", False):
        return
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils

    FlashAttentionUtils.v4_is_installed = False
    logger.info("Hid FlashAttention 4 from Transformer Engine; attention will use FlashAttention 2")

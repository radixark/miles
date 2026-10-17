"""Kimi K3 support for the Miles TorchTitan backend."""

from miles.backends.torchtitan_utils.models.kimi_k3.config import model_config_kwargs

__all__ = ["model_registry"]


def model_registry(flavor: str, attn_backend: str = "flex", *, hf_config):
    # Keep checkpoint configuration inspection independent of the optional GPU stack.
    from torchtitan.protocols.model_spec import ModelSpec

    from miles.backends.torchtitan_utils.models.kimi_k3.model import KimiK3Model
    from miles.backends.torchtitan_utils.models.kimi_k3.parallelize import parallelize_kimi_k3
    from miles.backends.torchtitan_utils.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

    return ModelSpec(
        name="kimi_k3",
        flavor=flavor,
        model=KimiK3Model.Config(**model_config_kwargs(hf_config)),
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiK3StateDictAdapter,
    )

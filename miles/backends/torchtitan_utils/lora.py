"""LoRA for the torchtitan backend. torchtitan 0.4.0 fuses q/k/v and gate/up into one projection each, so the
adapters need a fused-projection export before this is wired back up."""


def convert_model_config(model_config, *, hf_targets: list[str], rank: int, alpha: float):
    raise NotImplementedError("torchtitan LoRA is not ported to torchtitan 0.4.0 yet")

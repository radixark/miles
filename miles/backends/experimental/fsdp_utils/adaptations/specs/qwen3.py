"""Qwen3 dense adaptations."""

from ..class_patches import ModelPatchHook, register_model_patch


_TRUE_ON_POLICY_CONTRACT = "qwen3_dense_true_on_policy_v1"


def _is_qwen3(hf_config) -> bool:
    return str(getattr(hf_config, "model_type", "") or "") == "qwen3"


def _apply_true_on_policy_patch(hf_config, args) -> None:
    if not getattr(args, "true_on_policy_mode", False):
        return
    if getattr(args, "sglang_true_on_policy_contract", None) != _TRUE_ON_POLICY_CONTRACT:
        return
    if getattr(args, "fp16", False):
        raise ValueError(f"{_TRUE_ON_POLICY_CONTRACT} requires FSDP bf16 compute; --fp16 is unsupported")
    from ...models.qwen3 import apply_true_on_policy_patch_for_qwen3

    apply_true_on_policy_patch_for_qwen3()


register_model_patch(ModelPatchHook("qwen3_true_on_policy_precision", _is_qwen3, _apply_true_on_policy_patch))

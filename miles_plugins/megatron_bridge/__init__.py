"""Local Megatron-Bridge compatibility patches for Miles."""


def _patch_kimi_k25_vl_bridge_export():
    try:
        from megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge import KimiK25VLBridge
    except ImportError:
        return

    if getattr(KimiK25VLBridge, "_miles_plain_weight_export_patched", False):
        return

    original = KimiK25VLBridge.maybe_modify_converted_hf_weight

    def _patched(self, task, converted_weights_dict, hf_state_dict):
        result = {}
        for fqn, tensor in converted_weights_dict.items():
            if not self._is_quantized_expert_key(fqn):
                result[fqn] = tensor
                continue

            base = fqn[:-7] if fqn.endswith(".weight") else fqn
            packed_key = f"{base}.weight_packed"

            # BF16 checkpoints produced by convert_kimi_int4_to_bf16.py only
            # contain plain `.weight` tensors. Re-exporting them as INT4 triplets
            # breaks colocated weight updates because SGLang registers BF16 experts.
            if packed_key not in hf_state_dict:
                result[fqn] = tensor
                continue

            result.update(original(self, task, {fqn: tensor}, hf_state_dict))

        return result

    KimiK25VLBridge.maybe_modify_converted_hf_weight = _patched
    KimiK25VLBridge._miles_plain_weight_export_patched = True


_patch_kimi_k25_vl_bridge_export()

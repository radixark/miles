"""Unit tests for native-LoRA HF naming resolution — no GPU."""

from miles_plugins.lora.hf_adapter import resolve_hf_naming


def _write_index(tmp_path, keys):
    import json

    weight_map = {key: "a.safetensors" for key in keys}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return str(tmp_path)


class TestHfNaming:
    def test_deepseek_style_plural_shared_expert(self, tmp_path):
        path = _write_index(
            tmp_path,
            [
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
            ],
        )
        assert resolve_hf_naming(path) == ("model.layers.", "mlp.shared_experts.")

    def test_qwen3_5_nests_the_decoder_and_uses_singular(self, tmp_path):
        """The mtp block also has `layers.N.`; it must not win the prefix vote."""
        path = _write_index(
            tmp_path,
            [
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.1.mlp.shared_expert.up_proj.weight",
                "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                "vision_tower.encoder.blocks.0.wo.weight",
            ],
        )
        assert resolve_hf_naming(path) == ("model.language_model.layers.", "mlp.shared_expert.")

    def test_a_nested_mtp_block_does_not_win_the_prefix_vote(self, tmp_path):
        """Inkling-Small-4layer has more MTP tensors than decoder tensors."""
        path = _write_index(
            tmp_path,
            [
                "model.llm.layers.0.attn.wq_du.weight",
                *(f"model.mtp.layers.{i}.attn.wq_du.weight" for i in range(3)),
                *(f"model.mtp.layers.{i}.attn.wk_dv.weight" for i in range(3)),
            ],
        )
        assert resolve_hf_naming(path)[0] == "model.llm.layers."

    def test_kimi_k3_shared_experts_live_under_block_sparse_moe(self, tmp_path):
        path = _write_index(
            tmp_path,
            [
                "language_model.model.layers.0.mlp.gate_proj.weight",
                "language_model.model.layers.1.block_sparse_moe.shared_experts.up_proj.weight",
                "language_model.model.layers.1.block_sparse_moe.experts.7.w1.weight",
            ],
        )
        assert resolve_hf_naming(path) == ("language_model.model.layers.", "block_sparse_moe.shared_experts.")

    def test_missing_index_falls_back_to_the_plain_layout(self, tmp_path):
        assert resolve_hf_naming(str(tmp_path)) == ("model.layers.", "mlp.shared_expert.")
        assert resolve_hf_naming(None) == ("model.layers.", "mlp.shared_expert.")

"""Unit tests for the HF-namespace atomic group registry."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups


class TestHfAtomicUpdateGroups:
    def test_deepseekv4_registers_the_three_fused_pairs(self):
        groups = {g.key: g.suffixes for g in get_hf_atomic_update_groups("deepseekv4")}
        assert groups == {
            "wqkv_a": (".self_attn.wq_a.weight", ".self_attn.wkv.weight"),
            "compressor_wkv_gate": (".self_attn.compressor.wkv.weight", ".self_attn.compressor.wgate.weight"),
            "indexer_compressor_wkv_gate": (
                ".self_attn.indexer.compressor.wkv.weight",
                ".self_attn.indexer.compressor.wgate.weight",
            ),
        }

    def test_deepseekv4_suffixes_do_not_shadow_each_other(self):
        """endswith matching must resolve uniquely across the three v4 pairs."""
        suffixes = [s for g in get_hf_atomic_update_groups("deepseekv4") for s in g.suffixes]
        sample = {s: f"model.layers.0{s}" for s in suffixes}
        for suffix, name in sample.items():
            matches = [s for s in suffixes if name.endswith(s)]
            assert matches == [suffix], f"{name} matched {matches}"

    def test_model_groups_take_precedence_over_q_lora(self):
        groups = get_hf_atomic_update_groups("deepseekv4", q_lora_rank=1024)
        assert [g.key for g in groups] == ["wqkv_a", "compressor_wkv_gate", "indexer_compressor_wkv_gate"]

    def test_q_lora_fallback(self):
        groups = get_hf_atomic_update_groups("deepseekv3", q_lora_rank=1024)
        assert [g.suffixes for g in groups] == [
            ((".self_attn.q_a_proj.weight", ".self_attn.kv_a_proj_with_mqa.weight"))
        ]
        # Suffix matching covers both deepseek and kimi-vl prefixes.
        for prefix in ("model.layers.3", "language_model.model.layers.3"):
            assert f"{prefix}.self_attn.q_a_proj.weight".endswith(groups[0].suffixes[0])

    def test_no_q_lora_no_groups(self):
        assert get_hf_atomic_update_groups("qwen3") == []

    def test_inkling_registers_none(self):
        assert get_hf_atomic_update_groups("inkling") == []

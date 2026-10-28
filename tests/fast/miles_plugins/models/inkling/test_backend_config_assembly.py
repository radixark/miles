import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_config


class TestInklingBackendAssembly:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_provider_builds_the_real_transformer_config_from_backend_fields(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, enabled: bool, inkling_model: ModuleType
    ) -> None:
        """The real provider and builder preserve backend settings in the allocated model config."""
        model = inkling_model
        text = dict(
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
            swa_head_dim=4,
            sliding_window_size=16,
            d_rel=4,
            local_layer_ids=[],
            sconv_kernel_size=3,
            use_sconv=False,
            use_embed_norm=False,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            route_scale=1.0,
            intermediate_size=32,
            hidden_size=16,
            num_hidden_layers=6,
            vocab_size=64,
            rms_norm_eps=1e-5,
        )
        (tmp_path / "config.json").write_text(json.dumps({"text_config": text}))
        args = make_trainer_config(
            hf_checkpoint=str(tmp_path),
            allgather_cp=True,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            pipeline_model_parallel_size=3,
            bf16=enabled,
            sequence_parallel=enabled,
            expert_tensor_parallel_size=2 if enabled else None,
            context_parallel_size=2 if enabled else 1,
            variable_seq_lengths=enabled,
            moe_permute_fusion=enabled,
            fp32_residual_connection=enabled,
            decoder_first_pipeline_num_layers=1,
            decoder_last_pipeline_num_layers=1,
            max_position_embeddings=8192,
        )
        original = args.model_dump()
        block = object()
        spec_calls = []

        def spec(config: Any, vp_stage: int) -> object:
            spec_calls.append((config, vp_stage))
            return block

        monkeypatch.setattr(model, "get_inkling_block_spec", spec)
        monkeypatch.setattr(model, "InklingGPTModel", lambda **kwargs: SimpleNamespace(**kwargs))

        result = model.inkling_model_provider(pre_process=False, post_process=True, vp_stage=2, args=args)

        config = result.config
        assert isinstance(config, model.TransformerConfig)
        assert (
            config.tensor_model_parallel_size,
            config.expert_model_parallel_size,
            config.pipeline_model_parallel_size,
            config.bf16,
            config.sequence_parallel,
            config.expert_tensor_parallel_size,
            config.context_parallel_size,
            config.variable_seq_lengths,
            config.moe_permute_fusion,
            config.fp32_residual_connection,
            config.num_layers_in_first_pipeline_stage,
            config.num_layers_in_last_pipeline_stage,
        ) == (2, 2, 3, enabled, enabled, 2 if enabled else 1, 2 if enabled else 1, enabled, enabled, enabled, 1, 1)
        assert config.params_dtype == (torch.bfloat16 if enabled else torch.float32)
        assert config.pipeline_dtype == torch.float32
        assert config.inkling.hidden_size == 16
        assert config.moe_layer_freq == [1] * 6
        assert config.moe_activation_in_fp32 and config.moe_combine_in_fp32
        assert spec_calls == [(config, 2)]
        assert result.transformer_layer_spec is block
        assert result.max_sequence_length == 8192
        assert result.vocab_size == 64
        assert result.pre_process is False and result.post_process is True
        assert result.vp_stage == 2
        assert args.model_dump() == original

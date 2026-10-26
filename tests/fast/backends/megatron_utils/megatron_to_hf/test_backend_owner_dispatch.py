import pytest
import torch

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf, postprocess_hf_param
from miles.utils.args.runtime import TrainerConfig


class TestBackendOwnerDispatch:
    @pytest.mark.parametrize("model_name", ["qwen2", "llama", "glm4", "glm4moe", "qwen3moe"])
    def test_shared_dispatch_splits_qkv_using_backend_dimensions(self, model_name: str) -> None:
        """Shared export dispatch passes the backend owner to model-specific QKV conversion."""
        args = TrainerConfig.model_construct(
            backend=MegatronArgsNamespace(
                vocab_size=7, hidden_size=2, num_attention_heads=2, num_query_groups=1, kv_channels=1
            )
        )
        parameter = torch.arange(8).reshape(4, 2)

        result = convert_to_hf(
            args=args,
            model_name=model_name,
            name="module.module.decoder.layers.3.self_attention.linear_qkv.weight",
            param=parameter,
        )

        assert [name for name, _ in result] == [
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
        ]
        torch.testing.assert_close(result[0][1], torch.tensor([[0, 1], [2, 3]]))
        torch.testing.assert_close(result[1][1], torch.tensor([[4, 5]]))
        torch.testing.assert_close(result[2][1], torch.tensor([[6, 7]]))

    def test_export_trims_embedding_padding_using_backend_vocabulary(self) -> None:
        """Export trims padded embedding rows using the backend vocabulary size."""
        args = TrainerConfig.model_construct(backend=MegatronArgsNamespace(vocab_size=2))
        parameter = torch.arange(8).reshape(4, 2)

        [(name, value)] = convert_to_hf(
            args=args,
            model_name="qwen2",
            name="module.module.embedding.word_embeddings.weight",
            param=parameter,
        )

        assert name == "model.embed_tokens.weight"
        torch.testing.assert_close(value, torch.tensor([[0, 1], [2, 3]]))

    def test_postprocessing_trims_output_padding_using_backend_vocabulary(self) -> None:
        """Postprocessing trims padded output rows without reading flat configuration fields."""
        args = TrainerConfig.model_construct(backend=MegatronArgsNamespace(vocab_size=2))

        result = postprocess_hf_param(
            args=args,
            megatron_param_name="module.module.output_layer.weight",
            hf_param_name="lm_head.weight",
            param=torch.arange(8).reshape(4, 2),
        )

        torch.testing.assert_close(result, torch.tensor([[0, 1], [2, 3]]))

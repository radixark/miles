import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def test_provider_preserves_varlen_default_when_backend_omits_the_field(
    tmp_path: Path, inkling_model: ModuleType
) -> None:
    """An older backend without a varlen field preserves the upstream default."""
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 8}}))
    backend = SimpleNamespace(
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        bf16=True,
        sequence_parallel=False,
        expert_tensor_parallel_size=None,
        moe_permute_fusion=False,
        fp32_residual_connection=False,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        max_position_embeddings=128,
    )
    args = SimpleNamespace(backend=backend, hf_checkpoint=str(tmp_path))
    config = object()
    model = object()

    with (
        patch.object(inkling_model, "build_inkling_config", return_value=config) as build_config,
        patch.object(inkling_model, "get_inkling_block_spec", return_value=object()),
        patch.object(inkling_model, "InklingGPTModel", return_value=model),
    ):
        actual = inkling_model.inkling_model_provider(args=args)

    assert build_config.call_args.kwargs["varlen"] is True
    assert actual is model

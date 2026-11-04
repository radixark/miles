import argparse
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp


@pytest.mark.parametrize("model_name", ["inkling", "qwen3"])
@pytest.mark.parametrize("provide_origin", [False, True])
def test_ray_conversion_uses_saved_hf_config_and_copies_explicit_assets(
    local_ray_converter: ModuleType, tmp_path: Path, model_name: str, provide_origin: bool
) -> None:
    """Workers retain the saved HF configuration independently of the explicitly copied assets."""
    input_dir = tmp_path / "checkpoint"
    saved_hf = tmp_path / "saved-hf"
    assets_hf = tmp_path / "assets-hf"
    for directory, intermediate_size in [(saved_hf, 2), (assets_hf, 99)]:
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"model_type": "llama", "intermediate_size": intermediate_size, "vocab_size": 8})
        )
    tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    source_name = (
        "decoder.layers.0.mlp.experts.linear_fc1.weight0"
        if model_name == "inkling"
        else "embedding.word_embeddings.weight"
    )
    dist_cp.save({source_name: tensor}, checkpoint_id=str(input_dir))
    torch.save(
        {"args": argparse.Namespace(num_layers=1, num_experts=1, hf_checkpoint=str(saved_hf))},
        input_dir / "common.pt",
    )
    output_dir = tmp_path / "output"
    args = local_ray_converter.Args(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        origin_hf_dir=str(assets_hf) if provide_origin else None,
        model_name=model_name,
        force=False,
        max_file_bytes=1024,
        concurrency=1,
        task_group_bytes=1024,
        source_key_regex=None,
        dry_run_plan=False,
        progress=False,
        progress_interval_seconds=1,
    )

    assert local_ray_converter.convert_torch_dist_to_hf_ray(args) == str(output_dir)

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors.update(safetensors.torch.load_file(output_dir / shard))
    if model_name == "inkling":
        assert set(tensors) == {
            "model.llm.layers.0.mlp.experts.0.gate_proj.weight",
            "model.llm.layers.0.mlp.experts.0.up_proj.weight",
        }
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.gate_proj.weight"], tensor[:2])
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.up_proj.weight"], tensor[2:])
    else:
        assert set(tensors) == {"model.embed_tokens.weight"}
        torch.testing.assert_close(tensors["model.embed_tokens.weight"], tensor)
    if provide_origin:
        assert (output_dir / "config.json").read_bytes() == (assets_hf / "config.json").read_bytes()


class TestExportExpertScaleSelection:
    @pytest.mark.parametrize(
        ("legacy", "flags", "expected"),
        [
            ({"sglang_moe_runner_backend": "deep_gemm"}, [], "ue8m0"),
            ({"sglang_moe_runner_backend": "auto", "sglang_moe_a2a_backend": "deepep"}, [], "ue8m0"),
            ({"sglang_moe_runner_backend": "triton"}, [], None),
            ({"sglang_moe_runner_backend": "deep_gemm"}, ["--moe-runner-backend", "triton"], None),
            ({}, ["--moe-runner-backend", "deep_gemm"], "ue8m0"),
            (
                {"sglang_moe_runner_backend": "auto", "sglang_moe_a2a_backend": "none"},
                ["--moe-a2a-backend", "mooncake"],
                "ue8m0",
            ),
        ],
    )
    def test_explicit_export_target_overrides_legacy_checkpoint_values(
        self,
        local_ray_converter: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        legacy: dict[str, str],
        flags: list[str],
        expected: str | None,
    ) -> None:
        """Expert scale selection preserves old targets and honors explicit export targets."""
        from miles.backends.megatron_utils.megatron_to_hf.processors import quantizer_fp8

        monkeypatch.setattr(sys, "argv", ["convert", "--input-dir", "/input", "--output-dir", "/output", *flags])
        args = local_ray_converter.parse_args()
        conversion = local_ray_converter._make_conversion_args(args=args, megatron_args=argparse.Namespace(**legacy))
        monkeypatch.setattr(quantizer_fp8, "should_deepgemm_weight_requant_ue8m0", lambda **kwargs: True)

        assert (
            quantizer_fp8._get_scale_format(conversion, "model.layers.0.mlp.experts.0.gate_proj.weight", [128, 128])
            == expected
        )

    @pytest.mark.parametrize(
        ("deepgemm", "name", "flags", "missing", "expected"),
        [
            (False, "model.layers.0.mlp.experts.0.gate_proj.weight", [], None, None),
            (True, "model.layers.0.self_attn.q_proj.weight", [], None, "ue8m0"),
            (True, "model.layers.0.mlp.experts.0.gate_proj.weight", [], "moe_runner_backend", None),
            (
                True,
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                ["--moe-runner-backend", "auto"],
                "moe_a2a_backend",
                None,
            ),
        ],
    )
    def test_missing_target_is_required_only_when_scale_selection_reads_it(
        self,
        local_ray_converter: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        deepgemm: bool,
        name: str,
        flags: list[str],
        missing: str | None,
        expected: str | None,
    ) -> None:
        """Modern checkpoints require explicit expert targets only on the DeepGEMM expert path."""
        from miles.backends.megatron_utils.megatron_to_hf.processors import quantizer_fp8

        monkeypatch.setattr(sys, "argv", ["convert", "--input-dir", "/input", "--output-dir", "/output", *flags])
        conversion = local_ray_converter._make_conversion_args(
            args=local_ray_converter.parse_args(), megatron_args=argparse.Namespace()
        )
        monkeypatch.setattr(quantizer_fp8, "should_deepgemm_weight_requant_ue8m0", lambda **kwargs: deepgemm)

        if missing is not None:
            with pytest.raises(AttributeError, match=f"No common field {missing!r}"):
                quantizer_fp8._get_scale_format(conversion, name, [128, 128])
        else:
            assert quantizer_fp8._get_scale_format(conversion, name, [128, 128]) == expected


class TestExportHighPrecisionLayers:
    @pytest.mark.parametrize("method", ["mxfp8", "nvfp4"])
    @pytest.mark.parametrize(
        ("saved", "flags", "expected"),
        [
            ({"extra_high_precision_layers_megatron": ["linear_fc1"]}, [], ["linear_fc1"]),
            ({"extra_high_precision_layers_megatron": None}, [], None),
            ({}, [], None),
            ({}, ["--extra-high-precision-layers-megatron", "linear_fc1"], ["linear_fc1"]),
            (
                {"extra_high_precision_layers_megatron": ["linear_fc2"]},
                ["--extra-high-precision-layers-megatron", "linear_fc1"],
                ["linear_fc1"],
            ),
            (
                {"extra_high_precision_layers_megatron": ["linear_fc1"]},
                ["--extra-high-precision-layers-megatron"],
                [],
            ),
            ({}, ["--extra-high-precision-layers-megatron"], []),
        ],
    )
    def test_export_preserves_explicit_and_legacy_precision_choices(
        self,
        local_ray_converter: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        saved: dict[str, list[str] | None],
        flags: list[str],
        expected: list[str] | None,
    ) -> None:
        """CLI precision choices override legacy metadata and matched layers bypass real quantizers."""
        from miles.backends.megatron_utils.megatron_to_hf.processors import quantize_params

        monkeypatch.setattr(sys, "argv", ["convert", "--input-dir", "/input", "--output-dir", "/output", *flags])
        quantization_config = {"quant_method": method}
        conversion = local_ray_converter._make_conversion_args(
            args=local_ray_converter.parse_args(),
            megatron_args=argparse.Namespace(**saved),
        )
        assert conversion.extra_high_precision_layers_megatron == expected
        if expected:
            tensor = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
            tensors = [("model.layers.0.mlp.gate_proj.weight", tensor)]
            result = quantize_params(
                conversion, "decoder.layers.0.mlp.linear_fc1.weight", tensors, quantization_config
            )
            assert result is tensors
            assert result[0][1] is tensor

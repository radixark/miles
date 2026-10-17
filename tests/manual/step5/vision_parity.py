"""Compare real-weight vision features with SGLang and StepFun's reference.

Check FP32 RoPE angles against SGLang and BF16-rounded angles against StepFun's
reference. Reports include a same-tower, different-attention-kernel noise floor.
See the README for inputs and invocation.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import torch  # before anything that imports transformer_engine
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import miles_plugins.models.step5.vision as step5_vision
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.step5.vision import (
    Step5VisionConfig,
    Step5VisionEncoder,
    load_step5_vision_encoder,
    rope_angles,
)

_CASES = (("overview_728", 728, 1), ("crop_504", 504, 2))
_SGLANG_RENAMES = (
    (".attn.in_proj_weight", ".attn.qkv_proj.weight"),
    (".attn.in_proj_bias", ".attn.qkv_proj.bias"),
    (".attn.out_proj.", ".attn.proj."),
    (".mlp.c_fc", ".mlp.fc1"),
    (".mlp.c_proj", ".mlp.fc2"),
)


def _tower_weights(checkpoint: Path) -> dict[str, Tensor]:
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in sorted({shard for name, shard in weight_map.items() if name.startswith(("vision_model.", "vit_"))}):
        with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as f:
            weights |= {name: f.get_tensor(name) for name in f.keys() if name.startswith(("vision_model.", "vit_"))}
    return weights


class _ReferenceTower(nn.Module):
    """StepFun's ``Step4VisionEncoder`` plus the downsampling and projection of its ``_encode_images``."""

    def __init__(self, ref_code: str, vision_config: dict, weights: dict[str, Tensor], device: torch.device) -> None:
        super().__init__()
        sys.path.insert(0, ref_code)
        from vision_encoder import Step4VisionConfig, Step4VisionEncoder

        self.encoder = Step4VisionEncoder(Step4VisionConfig(**vision_config)).to(device=device, dtype=torch.bfloat16)
        self.encoder.load_state_dict(
            {k.removeprefix("vision_model."): v for k, v in weights.items() if "vit_large" not in k}
        )
        projector = weights["vit_large_projector.weight"]
        self.projector = nn.Linear(
            projector.shape[1], projector.shape[0], bias=False, device=device, dtype=torch.bfloat16
        )
        self.projector.load_state_dict({"weight": projector})

    def downsampled(self, pixels: Tensor) -> Tensor:
        """The tokens entering the projector, a transposed (non-contiguous) view as the reference leaves them."""
        pixels = pixels.to(torch.bfloat16)
        features = self.encoder(pixels)
        grid = pixels.shape[-1] // self.encoder.patch_size
        features = features.transpose(1, 2).reshape(pixels.shape[0], -1, grid, grid)
        features = self.encoder.vit_downsampler2(self.encoder.vit_downsampler1(features))
        return features.flatten(2).transpose(1, 2)

    def forward(self, pixels: Tensor) -> Tensor:
        return self.projector(self.downsampled(pixels))


class _SGLangTower(nn.Module):
    """SGLang's ``PerceptionEncoder`` and projector, as ``Step3p7ForConditionalGeneration`` runs them."""

    def __init__(self, checkpoint: str, hf_config, weights: dict[str, Tensor], device: torch.device) -> None:
        super().__init__()
        from sglang.srt import runtime_context
        from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
        from sglang.srt.layers.linear import ColumnParallelLinear
        from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
        from sglang.srt.model_loader.weight_utils import default_weight_loader
        from sglang.srt.models.step3_vl_10b import PerceptionEncoder
        from sglang.srt.server_args import ServerArgs
        from transformers.activations import ACT2FN

        runtime_context.publish(ServerArgs(model_path=checkpoint, trust_remote_code=True), role="scheduler")
        initialize_bf16_gemm_config()
        # its row-parallel linears consult the TP group even at TP 1
        init_distributed_environment(
            world_size=1, rank=0, local_rank=device.index, distributed_init_method="tcp://127.0.0.1:29517"
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        # SGLang builds the model on the GPU in the checkpoint dtype; its RoPE table is fp32 regardless
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            self.vision_model = PerceptionEncoder(
                hf_config.vision_config, ACT2FN[hf_config.vision_config.hidden_act], use_data_parallel=True
            )
            self.vit_large_projector = ColumnParallelLinear(
                4 * hf_config.vision_config.width,
                hf_config.text_config.hidden_size,
                bias=False,
                gather_output=False,
                tp_rank=0,
                tp_size=1,
            )
        torch.set_default_dtype(default_dtype)
        params = dict(self.named_parameters())
        for name, weight in weights.items():
            for old, new in _SGLANG_RENAMES:
                name = name.replace(old, new)
            param = params.pop(name)
            getattr(param, "weight_loader", default_weight_loader)(param, weight.to(device))
        assert not params, f"SGLang tower weights left unloaded: {sorted(params)[:8]}"

    def forward(self, pixels: Tensor) -> Tensor:
        features = self.vision_model(pixels.type(self.vision_model.dtype))
        return self.vit_large_projector(features.contiguous())[0]


def _diff(actual: Tensor, expected: Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_value": float(expected.float().abs().max()),
        "mean_abs_value": float(expected.float().abs().mean()),
    }


def _check_rope_matches_sglang(sglang: _SGLangTower, head_dim: int, device: torch.device) -> None:
    rope = sglang.vision_model.transformer.resblocks[0].rope
    native = rope.max_grid_height
    torch.testing.assert_close(
        rope_angles(native, native, head_dim, device=device), rope.freqs_cache[0, 0], rtol=0, atol=0
    )
    grid = 504 // 14
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    looked_up = rope.freqs_cache[0, 0].index_select(0, (rows * native + cols).flatten().to(device))
    torch.testing.assert_close(rope_angles(grid, grid, head_dim, device=device), looked_up, rtol=0, atol=0)


def _bf16_rope_angles(*args, **kwargs) -> Tensor:
    return rope_angles(*args, **kwargs).bfloat16()


def _compare(pixels: Tensor, trainer: Step5VisionEncoder, reference: _ReferenceTower, sglang: _SGLangTower) -> dict:
    features = {"trainer": trainer._encode(pixels), "reference": reference(pixels), "sglang": sglang(pixels)}
    # exact only up to the projector: the reference feeds it a transposed view, which rounds differently
    with mock.patch.object(step5_vision, "rope_angles", _bf16_rope_angles):
        torch.testing.assert_close(trainer.vision_model(pixels), reference.downsampled(pixels), rtol=0, atol=0)
        with_bf16_angles = trainer._encode(pixels)
    # the noise floor: the same tower with another attention kernel
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with_flash_attention = trainer._encode(pixels)
    return {
        "shape": list(features["trainer"].shape),
        "trainer_vs_reference": _diff(features["trainer"], features["reference"]),
        "trainer_vs_sglang": _diff(features["trainer"], features["sglang"]),
        "reference_vs_sglang": _diff(features["reference"], features["sglang"]),
        "trainer_with_bf16_rope_vs_reference": _diff(with_bf16_angles, features["reference"]),
        "trainer_flash_vs_default_attention": _diff(with_flash_attention, features["trainer"]),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--ref-code", required=True, help="directory holding StepFun's vision_encoder.py")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    hf_config = load_hf_config(args.hf_checkpoint)
    config = Step5VisionConfig.from_hf(hf_config)
    weights = _tower_weights(Path(args.hf_checkpoint))
    raw_vision_config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())["vision_config"]
    trainer = load_step5_vision_encoder(args.hf_checkpoint, config, device=device)
    reference = _ReferenceTower(args.ref_code, raw_vision_config, weights, device)
    sglang = _SGLangTower(args.hf_checkpoint, hf_config, weights, device)
    _check_rope_matches_sglang(sglang, config.width // config.heads, device)

    torch.manual_seed(1709)
    report = {"tensors": len(weights), "device": torch.cuda.get_device_name(device), "cases": {}}
    for case, side, batch in _CASES:
        report["cases"][case] = _compare(torch.randn(batch, 3, side, side, device=device), trainer, reference, sglang)
        print(json.dumps({case: report["cases"][case]}), flush=True)
    report["exact"] = [
        "rope_angles == SGLang's RoPE table and its crop-grid lookup",
        "trainer with BF16 RoPE angles == reference, up to the projector input",
    ]
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

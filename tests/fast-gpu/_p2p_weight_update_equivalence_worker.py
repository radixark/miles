"""One case of test_p2p_weight_update_equivalence.py: two engines start from the same checkpoint; one takes
sglang's own weight update, the other a p2p update (`ModelReplica` bytes written raw into its published storage)
from the same HF tensors, twice in a row. Prints PASS when every tensor and scalar of the two engines is equal
after each update."""

import argparse
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from sglang.srt import server_args as server_args_module
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_loader.loader import DefaultModelLoader, post_load_weights
from sglang.srt.server_args import ServerArgs

from miles.backends.megatron_utils.megatron_to_hf.processors import quantizer_fp8, quantizer_mxfp8, quantizer_nvfp4
from miles.backends.training_utils.weight_update.protocols.utils.loader_probe import HfTensorSpec
from miles.backends.training_utils.weight_update.protocols.utils.model_param_stager import ModelParamStager
from miles.backends.training_utils.weight_update.protocols.utils.model_replica import (
    ModelReplica,
    RolloutEngineRankConfig,
    build_model_replica,
    pack_into_buffers,
)
from miles.kernels.quant.fp8_blockwise import fp8_blockwise_cast

CUDA = torch.device("cuda")
TP_SIZE = 2
NUM_LAYERS = 2
BUFFER_NBYTES = 512 * 1024**2
# a byte the loader leaves unwritten reaches the engine as the fill, and differs from the reference
POISON_FILLS = (0xA5, 0x5A)

QUANT_CONFIGS = {
    "bf16": None,
    "fp8_block": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
    },
    # miles' mxfp8 recipes keep kv_b_proj in bf16
    "mxfp8": {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "ignored_layers": [f"model.layers.{layer}.self_attn.kv_b_proj" for layer in range(NUM_LAYERS)],
    },
    "nvfp4": {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "ignore": [
            "lm_head",
            "model.layers.*.self_attn*",
            "model.layers.*.mlp.shared_experts*",
            "model.layers.0.mlp*",
            "model.layers.*.mlp.gate",
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True, help="GLM-5.2's config.json")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fmt", choices=list(QUANT_CONFIGS), required=True)
    parser.add_argument("--moe-runner-backend", required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)

    model_config_json = _write_checkpoint(args.config_dir, args.model_dir, args.fmt)
    server_args = ServerArgs(
        model_path=str(args.model_dir),
        tp_size=TP_SIZE,
        skip_tokenizer_init=True,
        moe_runner_backend=args.moe_runner_backend,
    )
    server_args_module.set_global_server_args_for_scheduler(server_args)
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    quantizer_args = SimpleNamespace(
        sglang_moe_runner_backend=server_args.moe_runner_backend, sglang_moe_a2a_backend=server_args.moe_a2a_backend
    )
    parallelism = RankParallelismConfig(
        tp_size=TP_SIZE,
        tp_rank=args.rank,
        moe_tp_size=TP_SIZE,
        moe_tp_rank=args.rank,
        attn_tp_size=TP_SIZE,
        attn_tp_rank=args.rank,
        world_size=TP_SIZE,
        global_rank=args.rank,
        local_rank=args.rank,
    )

    model_config = ModelConfig.from_server_args(server_args)
    reference_engine, p2p_engine = (_build_engine(model_config, parallelism) for _ in range(2))
    _assert_identical(reference_engine, p2p_engine, "at startup")
    published_locations_by_name = _get_published_locations(p2p_engine)
    model_replica = build_model_replica(
        RolloutEngineRankConfig(runner_role="target", parallelism=parallelism, server_args=server_args),
        str(args.model_dir),
        transfer_buffer_device=torch.device("cpu"),
    )
    buffer_nbytes = max(
        BUFFER_NBYTES, max(layout.occupied_nbytes for layout in model_replica.transfer_buffer_param_layouts.values())
    )
    buffer = torch.empty(buffer_nbytes, dtype=torch.uint8, pin_memory=True)

    for version, fill in enumerate(POISON_FILLS, start=1):
        hf_tensors = _quantize(args.fmt, "reload", _make_random_hf_tensors(model_config_json, version), quantizer_args)

        # sglang's own update, as WeightUpdater runs it; the DeepSeek loader runs post_load_weights itself
        with ParallelismContext(parallelism):
            DefaultModelLoader.restore_weights_before_loading(reference_engine, CUDA)
            reference_engine.load_weights(iter(hf_tensors))
            DefaultModelLoader.postprocess_weights(reference_engine, CUDA)

        with ParallelismContext(parallelism):
            DefaultModelLoader.restore_weights_before_loading(p2p_engine, CUDA)
        assert _get_published_locations(p2p_engine) == published_locations_by_name, "restore moved published storage"
        _write_p2p_update(model_replica, buffer, fill, hf_tensors, p2p_engine, published_locations_by_name)
        with ParallelismContext(parallelism):
            post_load_weights(p2p_engine)
            DefaultModelLoader.postprocess_weights(p2p_engine, CUDA)
        assert (
            _get_published_locations(p2p_engine) == published_locations_by_name
        ), "postprocess moved published storage"

        _assert_identical(reference_engine, p2p_engine, f"after update {version}")
    print("PASS")


def _write_checkpoint(config_dir: Path, model_dir: Path, fmt: str) -> dict:
    """GLM-5.2 cut to a dense and an MoE layer with 32 experts, random weights in `fmt`'s checkpoint form."""
    model_config_json = json.loads((config_dir / "config.json").read_text())
    model_config_json.update(
        num_hidden_layers=NUM_LAYERS,
        first_k_dense_replace=1,
        mlp_layer_types=["dense", "sparse"],
        # layer 1 reuses layer 0's top-k indices, so only layer 0 has an indexer
        indexer_types=["full", "shared"],
        n_routed_experts=32,
        num_nextn_predict_layers=0,
        vocab_size=32768,
    )
    # token ids of the full vocabulary
    for key in ("pad_token_id", "eos_token_id", "quantization_config"):
        model_config_json.pop(key, None)
    if QUANT_CONFIGS[fmt] is not None:
        model_config_json["quantization_config"] = QUANT_CONFIGS[fmt]
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(model_config_json, indent=2))
    # the checkpoint form needs no quantizer args
    checkpoint = _quantize(
        fmt, "checkpoint", _make_random_hf_tensors(model_config_json, version=0), quantizer_args=None
    )
    save_file({name: tensor.contiguous().cpu() for name, tensor in checkpoint}, model_dir / "model.safetensors")
    return model_config_json


def _make_random_hf_tensors(config: dict, version: int) -> list[tuple[str, torch.Tensor]]:
    return [
        (name, _make_random_tensor(name, shape, dtype, version))
        for name, (shape, dtype) in _get_hf_shapes(config).items()
    ]


def _get_hf_shapes(config: dict) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    qk_nope, qk_rope, v_head = config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"]
    q_lora, kv_lora = config["q_lora_rank"], config["kv_lora_rank"]
    bf16 = torch.bfloat16

    def mlp(prefix: str, intermediate: int) -> dict:
        return {
            f"{prefix}gate_proj.weight": ((intermediate, hidden), bf16),
            f"{prefix}up_proj.weight": ((intermediate, hidden), bf16),
            f"{prefix}down_proj.weight": ((hidden, intermediate), bf16),
        }

    shapes = {"model.embed_tokens.weight": ((config["vocab_size"], hidden), bf16)}
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        attention = prefix + "self_attn."
        shapes |= {
            prefix + "input_layernorm.weight": ((hidden,), bf16),
            prefix + "post_attention_layernorm.weight": ((hidden,), bf16),
            attention + "q_a_proj.weight": ((q_lora, hidden), bf16),
            attention + "q_a_layernorm.weight": ((q_lora,), bf16),
            attention + "q_b_proj.weight": ((heads * (qk_nope + qk_rope), q_lora), bf16),
            attention + "kv_a_proj_with_mqa.weight": ((kv_lora + qk_rope, hidden), bf16),
            attention + "kv_a_layernorm.weight": ((kv_lora,), bf16),
            attention + "kv_b_proj.weight": ((heads * (qk_nope + v_head), kv_lora), bf16),
            attention + "o_proj.weight": ((hidden, heads * v_head), bf16),
        }
        if config["indexer_types"][layer] == "full":
            index_heads, index_head_dim = config["index_n_heads"], config["index_head_dim"]
            shapes |= {
                attention + "indexer.wq_b.weight": ((index_heads * index_head_dim, q_lora), bf16),
                attention + "indexer.wk.weight": ((index_head_dim, hidden), bf16),
                attention + "indexer.k_norm.weight": ((index_head_dim,), bf16),
                attention + "indexer.k_norm.bias": ((index_head_dim,), bf16),
                attention + "indexer.weights_proj.weight": ((index_heads, hidden), bf16),
            }
        if layer < config["first_k_dense_replace"]:
            shapes |= mlp(prefix + "mlp.", config["intermediate_size"])
            continue
        num_experts, intermediate = config["n_routed_experts"], config["moe_intermediate_size"]
        shapes[prefix + "mlp.gate.weight"] = ((num_experts, hidden), bf16)
        shapes[prefix + "mlp.gate.e_score_correction_bias"] = ((num_experts,), torch.float32)
        for expert in range(num_experts):
            shapes |= mlp(f"{prefix}mlp.experts.{expert}.", intermediate)
        shapes |= mlp(prefix + "mlp.shared_experts.", intermediate * config["n_shared_experts"])
    shapes["model.norm.weight"] = ((hidden,), bf16)
    shapes["lm_head.weight"] = ((config["vocab_size"], hidden), bf16)
    return shapes


def _make_random_tensor(name: str, shape: tuple[int, ...], dtype: torch.dtype, version: int) -> torch.Tensor:
    generator = torch.Generator(device=CUDA).manual_seed(zlib.crc32(name.encode()) * 131 + version)
    return (torch.randn(shape, generator=generator, device=CUDA) * 0.02).to(dtype)


def _quantize(
    fmt: str, purpose: str, hf_tensors: list[tuple[str, torch.Tensor]], quantizer_args: SimpleNamespace | None
) -> list[tuple[str, torch.Tensor]]:
    """bf16 HF tensors in `fmt`: the checkpoint form, or the form miles' quantizers send in an update (`reload`)."""
    quantized = []
    gate_or_up_by_expert = {}
    for name, weight in hf_tensors:
        if not _is_quantized(fmt, name, tuple(weight.shape)):
            quantized.append((name, weight))
        elif fmt == "fp8_block" and purpose == "checkpoint":
            qweight, scale = fp8_blockwise_cast(weight, [128, 128])
            quantized += [(name, qweight), (name.replace(".weight", ".weight_scale_inv"), scale)]
        elif fmt == "fp8_block":
            quantized += quantizer_fp8._quantize_param(quantizer_args, name, weight, [128, 128])
        elif fmt == "mxfp8":
            quantized += quantizer_mxfp8._quantize_param(name, weight)
        else:
            assert fmt == "nvfp4", fmt
            expert = name.rsplit(".", 2)[0]
            if name.endswith(("gate_proj.weight", "up_proj.weight")):
                gate_or_up_by_expert.setdefault(expert, []).append((name, weight))
                if len(gate_or_up_by_expert[expert]) < 2:
                    continue
                pair = gate_or_up_by_expert.pop(expert)
            else:
                pair = [(name, weight)]
            quantized += quantizer_nvfp4._quantize_moe_params(pair, [])
            if purpose == "checkpoint":
                for pair_name, _ in pair:
                    # gate and up read the same input, so calibration gives them one input_scale
                    calibration_key = (
                        expert if pair_name.endswith(("gate_proj.weight", "up_proj.weight")) else pair_name
                    )
                    input_scale = 0.25 + (zlib.crc32(calibration_key.encode()) % 1000) / 4000
                    quantized.append(
                        (
                            pair_name.replace(".weight", ".input_scale"),
                            torch.tensor(input_scale, dtype=torch.float32, device=CUDA),
                        )
                    )
    assert not gate_or_up_by_expert, f"gate or up without its pair: {list(gate_or_up_by_expert)[:4]}"
    return quantized


def _is_quantized(fmt: str, name: str, shape: tuple[int, ...]) -> bool:
    if fmt == "bf16" or not name.endswith(".weight") or len(shape) != 2:
        return False
    if (
        name in ("model.embed_tokens.weight", "lm_head.weight")
        or "layernorm" in name
        or name.endswith("mlp.gate.weight")
        # GLM-5.2's indexer is interleaved, so miles' quantizers keep these two in bf16
        or ".indexer.wk." in name
        or ".indexer.weights_proj." in name
    ):
        return False
    if fmt == "nvfp4":
        return ".mlp.experts." in name
    if fmt == "mxfp8":
        return ".kv_b_proj." not in name
    return True


def _build_engine(model_config: ModelConfig, parallelism: RankParallelismConfig) -> torch.nn.Module:
    with ParallelismContext(parallelism):
        return DefaultModelLoader(LoadConfig(load_format="auto")).load_model(
            model_config=model_config, device_config=DeviceConfig(device="cuda")
        )


def _get_published_locations(engine: torch.nn.Module) -> dict[str, tuple[int, int]]:
    return {
        name: (param.data_ptr(), param.numel() * param.element_size()) for name, param in engine.named_parameters()
    }


def _write_p2p_update(
    model_replica: ModelReplica,
    buffer: torch.Tensor,
    fill: int,
    hf_tensors: list[tuple[str, torch.Tensor]],
    engine: torch.nn.Module,
    published_locations_by_name: dict[str, tuple[int, int]],
) -> None:
    hf_name_mapping = model_replica.map_hf_names(
        {name: HfTensorSpec(tuple(tensor.shape), tensor.dtype) for name, tensor in hf_tensors}
    )
    stager = ModelParamStager(hf_name_mapping)
    params_by_name = dict(engine.named_parameters())
    for hf_tensor in hf_tensors:
        if hf_tensor[0] not in hf_name_mapping.param_names_by_hf_name:
            continue
        ready_hf_tensors_by_param_group = stager.stage([hf_tensor])
        for param_groups in pack_into_buffers(
            ready_hf_tensors_by_param_group, model_replica.transfer_buffer_param_layouts, buffer.numel()
        ):
            buffer.fill_(fill)
            param_names = [name for param_group in param_groups for name in param_group]
            param_bytes_by_name = model_replica.load_into(
                buffer,
                param_names,
                [t for param_group in param_groups for t in ready_hf_tensors_by_param_group[param_group]],
            )
            for name, param_bytes in param_bytes_by_name.items():
                _write_param_bytes(params_by_name[name], published_locations_by_name[name], param_bytes)
    stager.assert_all_done()


def _write_param_bytes(
    param: torch.nn.Parameter, published_location: tuple[int, int], param_bytes: torch.Tensor
) -> None:
    """Writes `param_bytes` at the published address, as Mooncake does."""
    address, nbytes = published_location
    assert param_bytes.numel() == nbytes, f"{param_bytes.numel()} bytes for a {nbytes}-byte published param"
    storage = param.untyped_storage()
    target = torch.empty(0, dtype=torch.uint8, device=CUDA).set_(storage, address - storage.data_ptr(), (nbytes,))
    target.copy_(param_bytes)


def _assert_identical(engine_a: torch.nn.Module, engine_b: torch.nn.Module, when: str) -> None:
    tensors_a, scalars_a = _collect_module_state(engine_a)
    tensors_b, scalars_b = _collect_module_state(engine_b)
    assert tensors_a.keys() == tensors_b.keys(), f"{when}: {sorted(tensors_a.keys() ^ tensors_b.keys())[:10]}"
    differing = []
    for key, tensor_a in tensors_a.items():
        tensor_b = tensors_b[key]
        if (tensor_a.shape, tensor_a.dtype) != (tensor_b.shape, tensor_b.dtype):
            differing.append(
                f"{key} {tuple(tensor_a.shape)} {tensor_a.dtype} vs {tuple(tensor_b.shape)} {tensor_b.dtype}"
            )
        elif not torch.equal(_view_as_bytes(tensor_a), _view_as_bytes(tensor_b.to(tensor_a.device))):
            differing.append(key)
    differing += [
        f"{key} {scalars_a.get(key)!r} vs {scalars_b.get(key)!r}"
        for key in scalars_a.keys() | scalars_b.keys()
        if scalars_a.get(key) != scalars_b.get(key)
    ]
    assert not differing, f"{when}, {len(differing)} differ: {differing[:10]}"


def _collect_module_state(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Every tensor and scalar a forward may read: each module's params, buffers and tensor attributes (also inside
    dicts, such as permute index caches), and each param's attributes."""
    tensors, scalars = {}, {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for registry in (module._parameters, module._buffers):
            tensors.update({prefix + key: value for key, value in registry.items() if value is not None})
        for key, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                tensors.setdefault(prefix + key, value)
            elif isinstance(value, dict) and key not in ("_parameters", "_buffers"):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        tensors.setdefault(f"{prefix}{key}[{sub_key}]", sub_value)
            elif isinstance(value, bool | int | float | str) and not key.startswith("__"):
                scalars[prefix + key] = value
        for key, param in module._parameters.items():
            if param is None:
                continue
            for attribute, value in vars(param).items():
                if isinstance(value, torch.Tensor):
                    tensors.setdefault(f"{prefix}{key}.{attribute}", value)
                elif isinstance(value, bool | int | float | str):
                    scalars[f"{prefix}{key}.{attribute}"] = value
    return tensors, scalars


def _view_as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


if __name__ == "__main__":
    main()

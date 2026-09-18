import argparse
import contextlib
import json
import os
import pickle
import re
import shutil
import time

import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from typing_extensions import override

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding
from miles.utils.hf_config import load_hf_config


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


@contextlib.contextmanager
def _megatron_classes_as_dummies():
    """Unpickle the raw torch_dist checkpoint without importing Megatron's argument classes.

    Scoped to the raw path on purpose: the bridge path hands the checkpoint to megatron.bridge, which
    needs those classes intact.
    """
    original = pickle.Unpickler
    pickle.Unpickler = UnpicklerWrapper
    try:
        yield
    finally:
        pickle.Unpickler = original


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == num_experts
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            expert_param = param[expert_id]
            yield expert_name, expert_param
    else:
        yield name, param


def get_layer_param(args, name, param):
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        assert param.shape[0] == num_layers
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            layer_param = param[layer_id]
            yield from get_expert_param(args, layer_name, layer_param)
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args, state_dict):
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)


def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None):
    # for miles update_weight compatible
    args.sglang_enable_ep_moe = False

    print(f"start saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # 2GB
    current_size = 0
    total_size = 0
    modeltensors = [{}]
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        converted_named_tensors = convert_to_hf(args, model_name, name, param)
        for converted_name, converted_param in converted_named_tensors:
            tensor_size = converted_param.numel() * converted_param.element_size()
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param
            current_size += tensor_size
            total_size += tensor_size

    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}

    num_files = len(modeltensors)
    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        for key in tensors.keys():
            metadata["weight_map"][key] = filename
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    json.dump(metadata, open(index_filepath, "w"), indent=2)
    print(f"{index_filepath} saved.")

    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        t = time.time()
        filepath = os.path.join(output_dir, filename)
        safetensors.torch.save_file(tensors, filepath)
        print(f"{filename} saved in {time.time() - t:.2f} sec.")


def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)


MEGATRON_TO_HF_MODES = ("raw", "bridge")


def convert_with_bridge(input_dir: str, output_dir: str, origin_hf_dir: str) -> None:
    """Export a megatron.bridge-built checkpoint (Qwen3-VL and friends) through the bridge itself.

    The raw converters in miles.backends.megatron_utils.megatron_to_hf only know the plain decoder
    layout, so a bridge checkpoint dies there with "Unknown parameter name" (#634). AutoBridge.export_ckpt
    loads the torch_dist checkpoint under a temporary gloo group and writes safetensors on CPU, so no
    GPU is needed here either.
    """
    try:
        from megatron.bridge import AutoBridge
    except ImportError as exc:
        raise RuntimeError(
            "--megatron-to-hf-mode bridge needs megatron.bridge, which this environment does not provide; "
            "run the conversion inside the miles image or install Megatron-Bridge"
        ) from exc
    bridge = AutoBridge.from_hf_pretrained(origin_hf_dir, trust_remote_code=True)
    bridge.export_ckpt(megatron_path=input_dir, hf_path=output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=MEGATRON_TO_HF_MODES,
        default="raw",
        help="raw: the per-model converters under miles.backends.megatron_utils.megatron_to_hf; "
        "bridge: AutoBridge.export_ckpt, required for checkpoints trained with --megatron-to-hf-mode bridge "
        "(needs --origin-hf-dir).",
    )
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        default=None,
        help="use the origin hf dir to copy files like tokenizer, config.json, etc.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5 * 1024**3,
        help="Chunk size for saving tensors, default is 2GB.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocab size for removing padding, if applicable. If not provided, no padding will be removed.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    if args.megatron_to_hf_mode == "bridge":
        if args.origin_hf_dir is None:
            raise ValueError("--megatron-to-hf-mode bridge needs --origin-hf-dir to build the bridge from")
        convert_with_bridge(args.input_dir, args.output_dir, args.origin_hf_dir)
        copy_assets(args.origin_hf_dir, args.output_dir)
        return

    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError(
            "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
        )

    if args.model_name is None:
        hf_config = load_hf_config(args.origin_hf_dir)
        args.model_name = type(hf_config).__name__.lower()

    state_dict = {}
    print(f"loading model from {args.input_dir}")
    t = time.time()
    with _megatron_classes_as_dummies():
        megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
        dist_cp.state_dict_loader._load_state_dict(
            state_dict,
            storage_reader=WrappedStorageReader(args.input_dir),
            planner=EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
    print(f"model loaded in {time.time()-t:.2f} sec.")

    save_tensors(megatron_args, args.model_name, state_dict, args.output_dir, args.chunk_size, args.vocab_size)

    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)


if __name__ == "__main__":
    main()

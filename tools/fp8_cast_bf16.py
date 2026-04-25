# Adapt from https://github.com/alibaba/Pai-Megatron-Patch/blob/2b201af08336dea0403df7c6b497c964cf5a2e75/toolkits/model_checkpoints_convertor/deepseek/fp8_cast_bf16.py
import json
import os
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import load_file, save_file
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from tile_kernels.quant import cast_back
from tqdm import tqdm


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantize a 2D FP8 weight matrix back to bf16 using a 128x128 block scale.

    Backed by ``tile_kernels.quant.cast_back`` so it shares the same dequant
    implementation as the rest of the DeepSeek stack.
    """
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    return cast_back((x, s), fmt='bf16', x_block_size=(block_size, block_size))


def main(fp8_path, bf16_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    os.system("cp -rf " + fp8_path + "/config.json " + bf16_path)
    os.system("cp -rf " + fp8_path + "/*.py " + bf16_path)
    os.system("cp -rf " + fp8_path + "/tokenizer* " + bf16_path)
    os.system("cp -rf " + fp8_path + "/chat_template* " + bf16_path)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file) as f:
        model_index = json.load(f)
    weight_map_raw = model_index["weight_map"]
    weight_map_renamed = {
        DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(tensor_name): file_name
        for tensor_name, file_name in weight_map_raw.items()
    }

    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        file_name = weight_map_renamed[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")

        loaded_file_dict_raw = loaded_files[file_name]
        loaded_file_dict_renamed = {
            DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(tensor_name): tensor
            for tensor_name, tensor in loaded_file_dict_raw.items()
        }

        return loaded_file_dict_renamed[tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        print(f"Handling file: {safetensor_file}")
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name_raw, weight in current_state_dict.items():
            weight_name = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(weight_name_raw)

            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map_renamed:
            weight_map_renamed.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map_renamed}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)

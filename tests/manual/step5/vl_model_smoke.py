"""Validate VL forward/backward, image-feature placement, and frozen-tower exclusion.

Exercise mixed and text-only micro-batches built by the real data path. Checkpoint
loading and sequence-parallel invocations are documented in the README.
"""

import json
import os

import numpy as np
import torch  # before anything that imports transformer_engine
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import get_model
from PIL import Image
from tools.convert_hf_to_torch_dist import add_conversion_args, get_args

import miles_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.megatron_to_hf.step5 import convert_step5_to_hf
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.training_utils.data.multimodal import expand_multimodal_rollout_data_in_place
from miles.backends.training_utils.data.rollout import DataIterator, get_batch
from miles.utils import chat_template_utils
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.processing_utils import extract_multimodal_train_inputs, load_processor, load_tokenizer
from miles_plugins.models.step5.vl_model import Step5VLModel

_IMAGE_TOKEN = 128001
# Sample 0: 20 text tokens, one image with one crop (81 + 169 <im_patch> tokens), 26 text tokens.
# Sample 1: 40 text tokens. 336 tokens in all, divisible by any TP size up to 8.
_SAMPLE_LENGTHS = (296, 40)
_IMAGE_SPAN = (20, 20 + 81 + 169)


def _batch(device: torch.device) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(0)
    input_ids = torch.randint(1000, 100000, (sum(_SAMPLE_LENGTHS),), generator=generator)
    input_ids[_IMAGE_SPAN[0] : _IMAGE_SPAN[1]] = _IMAGE_TOKEN
    cu_seqlens = torch.tensor([0, _SAMPLE_LENGTHS[0], sum(_SAMPLE_LENGTHS)], dtype=torch.int32, device=device)
    return {
        "input_ids": input_ids[None].to(device),
        "packed_seq_params": PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(_SAMPLE_LENGTHS),
            max_seqlen_kv=max(_SAMPLE_LENGTHS),
            qkv_format="thd",
        ),
        "images": {
            "step5_pixel_values": torch.randn(1, 3, 728, 728, generator=generator).to(device),
            "step5_patch_pixel_values": torch.randn(1, 3, 504, 504, generator=generator).to(device),
            "step5_num_patches": torch.tensor([1], device=device),
            "step5_image_positions": torch.arange(*_IMAGE_SPAN, device=device),
        },
    }


def _load_weights(model, args) -> None:
    if args.load is not None:
        load_checkpoint(model, None, None, strict=True)
        return
    bridge = AutoBridge.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge.load_weights(model, args.hf_checkpoint, memory_efficient=True)


def _check_tower_is_hidden(args, model, vl_model: Step5VLModel) -> dict:
    tower = vl_model._vision.encoder
    tower_ids = {id(param) for param in tower.parameters()}
    assert not tower_ids & {id(param) for param in vl_model.parameters()}
    assert not any(key.startswith(("vision_model.", "vit_", "_vision")) for key in vl_model.state_dict())
    synced = [name for name, _ in named_params_and_buffers(args, model)]
    assert not any("vision" in name or "vit_" in name for name in synced)
    if args.tensor_model_parallel_size == 1:  # the converter expects TP-gathered weights
        for name, param in named_params_and_buffers(args, model):
            convert_step5_to_hf(args, name, param)
    assert all(not param.requires_grad for param in tower.parameters())
    return {
        "tower_params": sum(param.numel() for param in tower.parameters()),
        "tower_gib": sum(param.numel() * param.element_size() for param in tower.parameters()) / 2**30,
        "synced_names": len(synced),
    }


def _decoder_input(model, vl_model: Step5VLModel, batch: dict, images: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The logits and the (sequence-gathered) hidden states entering the decoder."""
    captured = {}
    hook = vl_model.decoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: captured.update(hidden=kwargs["hidden_states"]), with_kwargs=True
    )
    logits = model(
        input_ids=batch["input_ids"],
        position_ids=None,
        attention_mask=None,
        packed_seq_params=batch["packed_seq_params"],
        **images,
    )
    hook.remove()
    hidden = captured["hidden"]
    if vl_model.config.sequence_parallel:
        hidden = tensor_parallel.gather_from_sequence_parallel_region(hidden, tensor_parallel_output_grad=False)
    return logits, hidden


def _text_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Mean next-token loss over targets that are text tokens inside the same sample."""
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    losses = tensor_parallel.vocab_parallel_cross_entropy(logits.float().transpose(0, 1), labels.transpose(0, 1))
    targets = labels.transpose(0, 1) != _IMAGE_TOKEN
    targets[_SAMPLE_LENGTHS[0] - 1] = targets[-1] = False
    return losses[targets].mean()


def _check_gradients(vl_model: Step5VLModel) -> dict:
    trainable = [(name, param) for name, param in vl_model.named_parameters() if param.requires_grad]
    missing = [name for name, param in trainable if param.grad is None]
    assert not missing, f"no gradient for {missing[:8]}"
    assert all(torch.isfinite(param.grad).all() for _, param in trainable)
    assert all(param.grad is None for param in vl_model._vision.encoder.parameters())
    word_embeddings = vl_model.embedding.word_embeddings
    start = word_embeddings.vocab_start_index
    if start <= _IMAGE_TOKEN < word_embeddings.vocab_end_index:
        assert not word_embeddings.weight.grad[_IMAGE_TOKEN - start].any(), "the <im_patch> row got a gradient"
    return {
        "trainable_params": len(trainable),
        "params_with_nonzero_grad": sum(bool(param.grad.any()) for _, param in trainable),
    }


# (width, height) per rollout sample, None for text: two crops, text, no crops.
_ROLLOUT_IMAGES = ((1000, 300), None, (256, 256))
_COLLATED_KEYS = ["tokens", "loss_masks", "total_lengths", "response_lengths", "multimodal_train_inputs"]


def _rollout_data(hf_checkpoint: str, device: torch.device) -> dict:
    """Rollout samples as the data side hands them to the trainer: processor output, text prompts without it."""
    tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
    processor = load_processor(hf_checkpoint)
    rng = np.random.RandomState(0)
    tokens, response_lengths, inputs = [], [], []
    for size in _ROLLOUT_IMAGES:
        images = [] if size is None else [Image.fromarray(rng.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8))]
        content = [{"type": "text", "text": "Find x.\n\n"}] + [{"type": "image", "image": image} for image in images]
        prompt = chat_template_utils.apply_chat_template(
            [{"role": "user", "content": content}], tokenizer=tokenizer, tokenize=False, add_generation_prompt=True
        )
        if images:
            output = processor(text=prompt, images=images)
            prompt_ids = output["input_ids"][0]
            inputs.append({key: value.to(device) for key, value in extract_multimodal_train_inputs(output).items()})
        else:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            inputs.append(None)
        response = rng.randint(1000, 100000, size=16).tolist()
        tokens.append(torch.tensor(prompt_ids + response, device=device))
        response_lengths.append(len(response))
    return {
        "tokens": tokens,
        "loss_masks": [torch.ones(length, dtype=torch.int, device=device) for length in response_lengths],
        "total_lengths": [len(t) for t in tokens],
        "response_lengths": response_lengths,
        "multimodal_train_inputs": inputs,
    }


def _check_collated_micro_batches(model, vl_model: Step5VLModel, hf_checkpoint: str, device: torch.device) -> dict:
    rollout_data = _rollout_data(hf_checkpoint, device)
    expand_multimodal_rollout_data_in_place(rollout_data, qkv_format="thd")
    iterator = DataIterator(rollout_data, micro_batch_indices=[[0, 1, 2], [1]])
    report = {}
    for name in ("mixed", "text_only"):
        collated = get_batch(iterator, _COLLATED_KEYS, qkv_format="thd")
        images = collated["multimodal_train_inputs"]
        cu_seqlens, max_seqlen = collated["cu_seqlens"], collated["max_seqlen"]
        batch = {
            "input_ids": collated["tokens"],
            "packed_seq_params": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
                qkv_format="thd",
            ),
        }
        with torch.no_grad():
            _, text_hidden = _decoder_input(model, vl_model, batch, images={})
            _, hidden = _decoder_input(model, vl_model, batch, images=images)
            image_rows = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
            if images:
                features = vl_model._vision.encoder.features(
                    images["step5_pixel_values"], images["step5_patch_pixel_values"], images["step5_num_patches"]
                )
                image_rows[images["step5_image_positions"]] = True
                torch.testing.assert_close(hidden[image_rows, 0], features.to(hidden.dtype), rtol=0, atol=0)
        torch.testing.assert_close(hidden[~image_rows], text_hidden[~image_rows], rtol=0, atol=0)
        report[f"{name}_micro_batch"] = {
            "tokens": collated["tokens"].shape[1],
            "image_tokens": int(image_rows.sum()),
            "crops": int(images["step5_num_patches"].sum()) if images else 0,
            "image_inputs": sorted(images),
        }
    assert report["text_only_micro_batch"]["image_inputs"] == [], "a text-only micro-batch carried image inputs"
    return report


def main() -> None:
    configure_logger_raw()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    args = get_args(add_conversion_args)
    init(args)

    allocated = torch.cuda.memory_allocated()
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    vl_model = model[0].module
    assert isinstance(vl_model, Step5VLModel)
    enforce_marked_param_dtypes(model)
    _load_weights(model, args)
    report = _check_tower_is_hidden(args, model, vl_model)
    report["model_gib_allocated"] = (torch.cuda.memory_allocated() - allocated) / 2**30

    batch = _batch(torch.device("cuda", local_rank))
    with torch.no_grad():
        _, text_hidden = _decoder_input(model[0], vl_model, batch, images={})
        features = vl_model._vision.encoder.features(
            batch["images"]["step5_pixel_values"],
            batch["images"]["step5_patch_pixel_values"],
            batch["images"]["step5_num_patches"],
        )
    logits, hidden = _decoder_input(model[0], vl_model, batch, images=batch["images"])
    image_rows = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
    image_rows[batch["images"]["step5_image_positions"]] = True
    torch.testing.assert_close(hidden[image_rows, 0], features.to(hidden.dtype), rtol=0, atol=0)
    torch.testing.assert_close(hidden[~image_rows], text_hidden[~image_rows], rtol=0, atol=0)

    loss = _text_loss(logits, batch["input_ids"])
    loss.backward()
    report |= _check_gradients(vl_model)
    report |= _check_collated_micro_batches(model[0], vl_model, args.hf_checkpoint, batch["input_ids"].device)
    report |= {
        "loss": loss.item(),
        "decoder_input_dtype": str(hidden.dtype),
        "tp": parallel_state.get_tensor_model_parallel_world_size(),
        "sequence_parallel": vl_model.config.sequence_parallel,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    if dist.get_rank() == 0:
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

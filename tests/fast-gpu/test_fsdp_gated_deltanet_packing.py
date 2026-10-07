"""Packed documents must stay independent through Qwen3.5's GatedDeltaNet and vision tower on the real kernels.

The CPU tests cannot check this: without FLA, causal_conv1d and flash-attn, HF's torch fallbacks ignore
``cu_seqlens`` and ``seq_idx``, so a leak between packed documents is invisible there.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-b-2-gpu-h200", labels=["fsdp"], hardware=["hopper"])

import itertools
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
from miles.backends.fsdp_utils.adaptations.specs import resolve_arch_adapter

# Measured on H200 (transformers 5.12.1, fla 0.5.2, causal_conv1d 1.6.1, flash-attn 2.7.4), both head sizes:
# packed log-probs equal the solo runs bitwise, so the floor sets their tolerance; packed gradients sit
# within 4.1e-3 of the solo sum, about the bf16 rounding of that sum (2.7e-3 to 3.3e-3). Without the packing
# kwargs the later documents move by 0.035 to 0.23 in log-prob and 0.30 to 0.99 in relative gradient, at
# least 28x the tolerance.
_NOISE_FLOOR = 1e-3
_NOISE_MULTIPLE = 4
_LEAK_MARGIN = 10

_VOCAB = 512
# 128 is the linear-attention head size of the released Qwen3.5 checkpoints, which FLA 0.5 runs on its
# FlashQLA kernel on Hopper; every other size runs on FLA's Triton kernels.
_LINEAR_HEAD_DIMS = pytest.mark.parametrize("linear_head_dim", [64, 128], ids=lambda dim: f"linear_head_dim_{dim}")


def _text_config(linear_head_dim):
    return dict(
        vocab_size=_VOCAB,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=linear_head_dim,
        linear_value_head_dim=linear_head_dim,
    )


_IMAGE_TOKEN, _VIDEO_TOKEN, _VISION_START, _VISION_END = 508, 509, 510, 511
# a 1x4x4 patch grid; spatial_merge_size=2 merges it into 4 image tokens
_IMAGE_GRID_THW = (1, 4, 4)
_IMAGE_TOKENS = 4
_PATCH_DIM = 3 * 2 * 2 * 2  # in_channels * temporal_patch_size * patch_size**2


def _tiny_text_model(linear_head_dim):
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM._from_config(
        Qwen3_5TextConfig(**_text_config(linear_head_dim)),
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    return model.cuda().eval()


def _tiny_vision_language_model(linear_head_dim):
    torch.manual_seed(0)
    text_config = _text_config(linear_head_dim)
    config = Qwen3_5Config(
        text_config=text_config,
        vision_config=dict(
            depth=2,
            hidden_size=128,
            intermediate_size=256,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            out_hidden_size=text_config["hidden_size"],
            num_position_embeddings=16,
        ),
        image_token_id=_IMAGE_TOKEN,
        video_token_id=_VIDEO_TOKEN,
        vision_start_token_id=_VISION_START,
        vision_end_token_id=_VISION_END,
    )
    model = Qwen3_5ForConditionalGeneration._from_config(
        config, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    )
    assert model.config.text_config._attn_implementation == "flash_attention_2"
    assert model.config.vision_config._attn_implementation == "flash_attention_2"
    return model.cuda().eval()


def _random_tokens(length, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(1, _IMAGE_TOKEN, (length,), device="cuda", generator=generator)


def _packed_batch(docs, padded_length):
    """What `get_batch` hands the FSDP actor in thd format: documents back to back, then pad tokens (id 0,
    position 0) recorded as one trailing segment."""
    lengths = [doc.numel() for doc in docs]
    pad = padded_length - sum(lengths)
    cu_seqlens_host = tuple(itertools.accumulate([*lengths, pad], initial=0))
    zeros = torch.zeros(pad, dtype=torch.long, device="cuda")
    return dict(
        tokens=torch.cat([*docs, zeros]).view(1, -1),
        position_ids=torch.cat([*(torch.arange(n, device="cuda") for n in lengths), zeros]).view(1, -1),
        cu_seqlens=torch.tensor(cu_seqlens_host, dtype=torch.int32, device="cuda"),
        cu_seqlens_host=cu_seqlens_host,
        max_seqlen=max(end - start for start, end in itertools.pairwise(cu_seqlens_host)),
    )


def _actor_for(model):
    actor = object.__new__(FSDPTrainRayActor)
    actor.arch_adapter = resolve_arch_adapter(model.config)
    return actor


def _stock_inputs(batch):
    """The packed row as stock HF sees it: boundaries only through position_ids."""
    inputs = dict(input_ids=batch["tokens"], position_ids=batch["position_ids"], attention_mask=None)
    inputs.update(batch.get("multimodal_train_inputs") or {})
    return inputs


def _token_logprobs(model, inputs):
    """log p(token t+1 | tokens <= t) for every position of the single row."""
    logits = model(**inputs).logits
    targets = inputs["input_ids"][:, 1:, None]
    return torch.log_softmax(logits.float(), dim=-1)[:, :-1].gather(-1, targets)[0, :, 0]


def _per_document(row_logprobs, batch, num_docs):
    """Split a packed row's log-probs into each document's within-document predictions."""
    bounds = itertools.pairwise(batch["cu_seqlens_host"][: num_docs + 1])
    return [row_logprobs[start : end - 1] for start, end in bounds]


def _max_gaps(packed_docs, solo_docs):
    return [(packed - solo).abs().max().item() for packed, solo in zip(packed_docs, solo_docs, strict=True)]


def _tolerance(noise):
    """A tolerance from a gap measured where no leak is possible."""
    return max(_NOISE_MULTIPLE * noise, _NOISE_FLOOR)


@_LINEAR_HEAD_DIMS
@torch.no_grad()
def test_packed_text_documents_match_each_document_run_alone(linear_head_dim):
    """GatedDeltaNet carrying recurrence or conv state from one packed document into the next."""
    model = _tiny_text_model(linear_head_dim)
    docs = [_random_tokens(length, seed) for seed, length in enumerate((37, 64, 129))]
    batch = _packed_batch(docs, padded_length=256)
    assert batch["cu_seqlens_host"] == (0, 37, 101, 230, 256) and batch["max_seqlen"] == 129

    solo = [_token_logprobs(model, dict(input_ids=doc.view(1, -1))) for doc in docs]
    packed = _per_document(_token_logprobs(model, _actor_for(model)._get_model_inputs_args(batch)), batch, len(docs))
    stock = _per_document(_token_logprobs(model, _stock_inputs(batch)), batch, len(docs))

    gaps, stock_gaps = _max_gaps(packed, solo), _max_gaps(stock, solo)
    tolerance = _tolerance(gaps[0])  # the first document has no predecessor to leak from
    print(f"packed-vs-solo max |dlogp| per doc {gaps}; without packing kwargs {stock_gaps}; tolerance {tolerance}")
    assert max(gaps[1:]) <= tolerance
    assert min(stock_gaps[1:]) >= _LEAK_MARGIN * tolerance  # the comparison detects a leak


def _grads(model, loss_fn):
    model.zero_grad(set_to_none=True)
    loss_fn().backward()
    return {name: param.grad.float().clone() for name, param in model.named_parameters() if param.grad is not None}


def _sum_grads(grads_list):
    return {name: sum(grads[name] for grads in grads_list) for name in grads_list[0]}


def _max_relative_error(grads, reference):
    assert grads.keys() == reference.keys()
    return max(((grads[name] - ref).norm() / ref.norm()).item() for name, ref in reference.items() if ref.norm() > 0)


@_LINEAR_HEAD_DIMS
def test_packed_gradients_match_the_sum_of_each_document_run_alone(linear_head_dim):
    """Backward under gradient checkpointing mixing gradient between packed documents."""
    model = _tiny_text_model(linear_head_dim)
    model.gradient_checkpointing_enable()
    model.train()
    docs = [_random_tokens(length, seed) for seed, length in enumerate((37, 64, 129))]
    batch = _packed_batch(docs, padded_length=256)
    packing_inputs = _actor_for(model)._get_model_inputs_args(batch)

    def packed_loss(inputs, docs_in_loss):
        return lambda: sum(
            _per_document(_token_logprobs(model, inputs), batch, len(docs))[i].sum() for i in docs_in_loss
        )

    def solo_loss(doc):
        return lambda: _token_logprobs(model, dict(input_ids=doc.view(1, -1))).sum()

    all_docs = range(len(docs))
    solo_sum = _sum_grads([_grads(model, solo_loss(doc)) for doc in docs])
    packed = _grads(model, packed_loss(packing_inputs, all_docs))
    # Gradients are linear in the loss, so this gap is only bf16 rounding from adding three bf16 gradients
    # where one backward rounds once; the solo sum carries the same rounding.
    rounding = _max_relative_error(
        packed, _sum_grads([_grads(model, packed_loss(packing_inputs, [i])) for i in all_docs])
    )
    error = _max_relative_error(packed, solo_sum)
    stock_error = _max_relative_error(_grads(model, packed_loss(_stock_inputs(batch), all_docs)), solo_sum)

    tolerance = _tolerance(rounding)
    print(
        f"max per-param relative grad error vs the solo sum: packed {error}, without packing kwargs "
        f"{stock_error}; rounding of the sum {rounding}; tolerance {tolerance}"
    )
    assert error <= tolerance
    assert stock_error >= _LEAK_MARGIN * tolerance


def _image_document(text_length, seed):
    image = torch.tensor([_VISION_START, *[_IMAGE_TOKEN] * _IMAGE_TOKENS, _VISION_END], device="cuda")
    return torch.cat([image, _random_tokens(text_length, seed)])


def _image_inputs(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_patches = _IMAGE_GRID_THW[0] * _IMAGE_GRID_THW[1] * _IMAGE_GRID_THW[2]
    return dict(
        pixel_values=torch.randn(num_patches, _PATCH_DIM, device="cuda", generator=generator, dtype=torch.bfloat16),
        image_grid_thw=torch.tensor([_IMAGE_GRID_THW], device="cuda"),
    )


@_LINEAR_HEAD_DIMS
@torch.no_grad()
def test_packed_image_document_matches_the_image_document_run_alone(linear_head_dim):
    """The vision tower raising on the LM's packing kwargs, or an image document reading its predecessor's state."""
    model = _tiny_vision_language_model(linear_head_dim)
    adapter = resolve_arch_adapter(model.config)
    docs = [_random_tokens(37, seed=0), _image_document(text_length=58, seed=1)]
    batch = _packed_batch(docs, padded_length=128)
    batch["multimodal_train_inputs"] = _image_inputs(seed=2)
    packing_inputs = _actor_for(model)._get_model_inputs_args(batch)

    # Pins the upstream HF bug the guard works around: this turns red once HF stops forwarding the
    # language model's kwargs into the vision tower's flash attention.
    with pytest.raises(TypeError, match="multiple values for keyword argument 'cu_seq_lens_q'"):
        model(**packing_inputs)

    adapter.patch_model(model, None)
    # the actor hands HF 1-D text positions, so without them HF would build multimodal RoPE positions instead
    solo = [
        _token_logprobs(model, dict(input_ids=docs[0].view(1, -1))),
        _token_logprobs(
            model,
            dict(
                input_ids=docs[1].view(1, -1),
                position_ids=torch.arange(docs[1].numel(), device="cuda").view(1, -1),
                **batch["multimodal_train_inputs"],
            ),
        ),
    ]
    packed = _per_document(_token_logprobs(model, packing_inputs), batch, len(docs))
    stock = _per_document(_token_logprobs(model, _stock_inputs(batch)), batch, len(docs))

    gaps, stock_gaps = _max_gaps(packed, solo), _max_gaps(stock, solo)
    tolerance = _tolerance(gaps[0])  # the text document comes first, so nothing can leak into it
    print(
        f"packed-vs-solo max |dlogp| [text, image] {gaps}; without packing kwargs {stock_gaps}; tolerance {tolerance}"
    )
    assert gaps[1] <= tolerance
    assert stock_gaps[1] >= _LEAK_MARGIN * tolerance


def test_building_packing_kwargs_never_syncs_the_device():
    """A host-device sync on every micro-batch from sizing `seq_idx` with a device-side length."""
    actor = object.__new__(FSDPTrainRayActor)
    actor.arch_adapter = resolve_arch_adapter(SimpleNamespace(model_type="qwen3_5"))
    docs = [_random_tokens(length, seed) for seed, length in enumerate((37, 64, 129))]
    batch = _packed_batch(docs, padded_length=256)

    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        inputs = actor._get_model_inputs_args(batch)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    expected_seq_idx = [doc for doc, length in enumerate((37, 64, 129, 26)) for _ in range(length)]
    assert inputs["seq_idx"].tolist() == [expected_seq_idx]
    assert inputs["cu_seq_lens_q"] is batch["cu_seqlens"] and inputs["max_length_q"] == 129


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

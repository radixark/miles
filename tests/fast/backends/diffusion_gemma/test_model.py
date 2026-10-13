import pytest
import torch
from transformers import DiffusionGemmaConfig, DiffusionGemmaTextConfig, Gemma4VisionConfig
from transformers.cache_utils import DynamicCache
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion as HFModel

from miles.backends.fsdp_utils.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion


def tiny_config():
    return DiffusionGemmaConfig(
        vision_config=Gemma4VisionConfig(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            position_embedding_size=16,
        ),
        text_config=DiffusionGemmaTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            num_global_key_value_heads=1,
            num_experts=3,
            top_k_experts=2,
            moe_intermediate_size=12,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
        ),
    )


def inputs(batch_size=2):
    clean = torch.tensor([[3, 4, 5, 6]]).expand(batch_size, -1)
    noisy = torch.tensor([[7, 8]]).expand(batch_size, -1)
    encoder_mask = torch.ones(batch_size, 1, 4, 4, dtype=torch.bool).tril()
    decoder_mask = torch.ones(batch_size, 1, 2, 6, dtype=torch.bool)
    return dict(
        input_ids=clean,
        decoder_input_ids=noisy,
        encoder_attention_mask={key: encoder_mask for key in ("sliding_attention", "full_attention")},
        decoder_attention_mask={key: decoder_mask for key in ("sliding_attention", "full_attention")},
        position_ids=torch.arange(4).expand(batch_size, -1),
        decoder_position_ids=torch.arange(4, 6).expand(batch_size, -1),
        self_conditioning_mask=torch.tensor([False, True])[:batch_size],
    )


def test_single_stack_shapes_and_encoder_kv_gradient():
    model = DiffusionGemmaForBlockDiffusion(tiny_config())
    calls = []
    kv_tensors = []

    def record_layer(module, args, kwargs, output):
        calls.append(torch.is_grad_enabled())
        if kwargs.get("encoder_mode"):
            for tensor in output[1]:
                tensor.retain_grad()
                kv_tensors.append(tensor)

    handle = model.model.decoder.layers[0].register_forward_hook(record_layer, with_kwargs=True)
    encoder_logits, decoder_logits = model(**inputs())
    handle.remove()
    assert encoder_logits.shape == (2, 4, 32)
    assert decoder_logits.shape == (2, 2, 32)
    assert calls == [True, False, True]
    decoder_logits.square().mean().backward()
    assert all(tensor.grad is not None and tensor.grad.abs().sum() > 0 for tensor in kv_tensors)
    assert model.model.decoder.layers[0].self_attn.k_proj.weight.grad.abs().sum() > 0
    assert not any("encoder.language_model" in name for name, _ in model.named_parameters())


def additive(mask):
    return torch.zeros(mask.shape).masked_fill(~mask, torch.finfo(torch.float32).min)


@pytest.mark.parametrize("self_conditioning", [False, True])
def test_hf_parity(self_conditioning):
    torch.manual_seed(2)
    reference_config = tiny_config()
    reference_config._attn_implementation = "eager"
    hf = HFModel(reference_config).eval()
    model = DiffusionGemmaForBlockDiffusion(tiny_config()).eval()
    model.load_state_dict(hf.state_dict(), strict=False)
    data = inputs()
    if not self_conditioning:
        data["self_conditioning_mask"] = torch.zeros(2, dtype=torch.bool)
    encoder_masks = {key: additive(mask) for key, mask in data["encoder_attention_mask"].items()}
    decoder_masks = {key: additive(mask) for key, mask in data["decoder_attention_mask"].items()}
    cache = DynamicCache()
    enc = hf.model.encoder.language_model(
        input_ids=data["input_ids"],
        attention_mask=encoder_masks,
        position_ids=data["position_ids"],
        past_key_values=cache,
        use_cache=True,
    ).last_hidden_state
    dec = hf.model.decoder(
        decoder_input_ids=data["decoder_input_ids"],
        decoder_attention_mask=decoder_masks,
        decoder_position_ids=data["decoder_position_ids"],
        past_key_values=cache,
    ).last_hidden_state
    if self_conditioning:
        first_logits = torch.tanh(hf.lm_head(dec).float() / hf.final_logit_softcapping) * hf.final_logit_softcapping
        dec = hf.model.decoder(
            decoder_input_ids=data["decoder_input_ids"],
            decoder_attention_mask=decoder_masks,
            decoder_position_ids=data["decoder_position_ids"],
            past_key_values=cache,
            self_conditioning_logits=first_logits,
            self_conditioning_mask=data["self_conditioning_mask"],
        ).last_hidden_state
    expected_encoder = hf.lm_head(enc)
    expected_decoder = hf.lm_head(dec)
    cap = hf.final_logit_softcapping
    expected_encoder = torch.tanh(expected_encoder / cap) * cap
    expected_decoder = torch.tanh(expected_decoder / cap) * cap
    actual_encoder, actual_decoder = model(**data)
    torch.testing.assert_close(actual_encoder, expected_encoder)
    torch.testing.assert_close(actual_decoder, expected_decoder)


def test_native_checkpoint_load_preserves_encoder_scalar_and_save_roundtrip(tmp_path):
    hf = HFModel(tiny_config())
    hf.model.encoder.language_model.layers[0].layer_scalar.fill_(0.7)
    hf.model.decoder.layers[0].layer_scalar.fill_(1.3)
    hf.save_pretrained(tmp_path / "native")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(tmp_path / "native")
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, hf.get_parameter(name))
    assert model.model.decoder.layers[0].encoder_layer_scalar.item() == pytest.approx(0.7)
    assert model.model.decoder.layers[0].layer_scalar.item() == pytest.approx(1.3)
    model.save_pretrained(tmp_path / "shared")
    loaded = DiffusionGemmaForBlockDiffusion.from_pretrained(tmp_path / "shared")
    for actual, expected in zip(loaded(**inputs()), model(**inputs()), strict=True):
        torch.testing.assert_close(actual, expected)


def test_bfloat16_logits_use_hf_float32_softcap():
    model = DiffusionGemmaForBlockDiffusion(tiny_config()).to(torch.bfloat16)
    assert all(logits.dtype == torch.float32 for logits in model(**inputs()))


def test_checkpointed_decoder_loss_keeps_encoder_gradient():
    torch.manual_seed(4)
    model = DiffusionGemmaForBlockDiffusion(tiny_config())
    baseline = DiffusionGemmaForBlockDiffusion(tiny_config())
    baseline.load_state_dict(model.state_dict())
    model.gradient_checkpointing_enable()
    model(**inputs())[1].square().mean().backward()
    baseline(**inputs())[1].square().mean().backward()
    for (_, actual), (_, expected) in zip(model.named_parameters(), baseline.named_parameters(), strict=True):
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad)


def test_sdpa_backend_dispatches_without_materializing_attention_scores(monkeypatch):
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["scale"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    config = tiny_config()
    config._attn_implementation = "sdpa"
    model = DiffusionGemmaForBlockDiffusion(config)
    model(**inputs())[1].sum().backward()
    assert calls == [1.0] * 6


def test_reentrant_checkpointing_rejected_for_nested_encoder_kv():
    model = DiffusionGemmaForBlockDiffusion(tiny_config())
    with pytest.raises(ValueError, match="non-reentrant"):
        model.gradient_checkpointing_enable({"use_reentrant": True})


@pytest.mark.parametrize("checkpointing", [False, True])
def test_fsdp2_bfloat16_policy_preserves_fp32_buffers_and_backward(tmp_path, checkpointing):
    # Fake collectives exercise actual FSDP2 materialization on one CPU rank;
    # this checks parameter/buffer dtype handling, not multi-rank communication.
    import torch.distributed as dist
    import torch.testing._internal.distributed.fake_pg  # noqa: F401
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dist.init_process_group("fake", rank=0, world_size=1, init_method=f"file://{tmp_path / 'init'}")
    try:
        model = DiffusionGemmaForBlockDiffusion(tiny_config()).float()
        if checkpointing:
            model.gradient_checkpointing_enable()
        mesh = init_device_mesh("cpu", (1,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.decoder.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        encoder_logits, decoder_logits = model(**inputs())
        (encoder_logits.square().mean() + decoder_logits.square().mean()).backward()
        assert all(layer.layer_scalar.dtype == torch.float32 for layer in model.model.decoder.layers)
        assert all(layer.encoder_layer_scalar.dtype == torch.float32 for layer in model.model.decoder.layers)
        key_gradient = model.model.decoder.layers[0].self_attn.k_proj.weight.grad
        assert key_gradient is not None
        assert torch.isfinite(key_gradient.to_local()).all()
        assert key_gradient.to_local().abs().sum() > 0
        assert torch.isfinite(decoder_logits).all()
    finally:
        dist.destroy_process_group()

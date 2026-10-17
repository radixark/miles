"""Step5 VL model: image features scattered into the embeddings, and the tower kept out of the model."""

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.training")
from megatron.core import parallel_state  # noqa: E402
from megatron.core.models.gpt import GPTModel  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from miles_plugins.models.step5.vision import Step5VisionConfig, Step5VisionEncoder  # noqa: E402
from miles_plugins.models.step5.vl_model import Step5VLModel, scatter_image_features  # noqa: E402

_IMAGE_TOKEN = 7


def test_features_replace_the_embeddings_at_their_positions():
    embeddings = torch.randn(6, 1, 4)
    features = torch.randn(2, 4, dtype=torch.bfloat16)

    out = scatter_image_features(embeddings, features, torch.tensor([4, 1]), shard_start=0)

    expected = embeddings.clone()
    expected[4, 0], expected[1, 0] = features[0].float(), features[1].float()
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_no_image_features_leave_the_embeddings_unchanged():
    embeddings = torch.randn(6, 1, 4)
    out = scatter_image_features(embeddings, torch.empty(0, 4), torch.empty(0, dtype=torch.long), shard_start=0)
    torch.testing.assert_close(out, embeddings, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [2, 3])
def test_sequence_parallel_slices_reassemble_the_full_sequence(tp_size):
    embeddings, features = torch.randn(12, 1, 4), torch.randn(5, 4)
    positions = torch.tensor([11, 0, 5, 6, 3])
    local_len = 12 // tp_size

    slices = [
        scatter_image_features(chunk, features, positions, shard_start=rank * local_len)
        for rank, chunk in enumerate(embeddings.chunk(tp_size))
    ]

    full = scatter_image_features(embeddings, features, positions, shard_start=0)
    torch.testing.assert_close(torch.cat(slices), full, rtol=0, atol=0)


def test_gradients_reach_only_the_text_rows():
    embeddings = torch.randn(5, 1, 3, requires_grad=True)
    scatter_image_features(embeddings, torch.randn(2, 3), torch.tensor([0, 3]), shard_start=0).sum().backward()
    assert embeddings.grad[:, 0, 0].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]


@pytest.fixture(scope="module")
def model_parallel(tmp_path_factory):
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        rendezvous = tmp_path_factory.mktemp("step5-vl") / "process-group"
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        if owns_process_group:
            dist.destroy_process_group()


def _gpt_kwargs() -> dict:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=24,
        num_attention_heads=2,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp32_residual_connection=True,
    )
    return dict(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=16,
        max_sequence_length=32,
        position_embedding_type="rope",
    )


def _vl_model() -> Step5VLModel:
    encoder = Step5VisionEncoder(
        Step5VisionConfig(width=8, layers=1, heads=2, patch_size=2, image_size=8, mlp_width=16, text_hidden_size=24)
    )
    return Step5VLModel(vision_encoder=encoder.requires_grad_(False), image_token_id=_IMAGE_TOKEN, **_gpt_kwargs())


def test_the_tower_is_not_part_of_the_model(model_parallel):
    model = _vl_model()
    encoder = model._vision.encoder

    assert model.state_dict().keys() == GPTModel(**_gpt_kwargs()).state_dict().keys()
    tower_params = {id(param) for param in encoder.parameters()}
    assert not tower_params & {id(param) for param in model.parameters()}
    assert not any(module is encoder for module in model.modules())
    # Module-wide calls skip the tower: this one was built in fp32 and stays there.
    model.bfloat16()
    assert all(param.dtype == torch.bfloat16 for param in model.parameters())
    assert all(param.dtype == torch.float32 for param in encoder.parameters())


def test_image_features_replace_the_im_patch_embeddings(model_parallel):
    model = _vl_model()
    input_ids = torch.tensor([[1, _IMAGE_TOKEN, _IMAGE_TOKEN, 2, _IMAGE_TOKEN]])
    features = torch.randn(3, 24, dtype=torch.bfloat16)
    text = model.embedding(input_ids=input_ids, position_ids=None)

    mixed = model._with_image_features(text, input_ids, features, torch.tensor([1, 2, 4]))

    assert mixed.dtype == torch.float32  # the fp32 residual stream
    torch.testing.assert_close(mixed[[1, 2, 4], 0], features.float(), rtol=0, atol=0)
    torch.testing.assert_close(mixed[[0, 3]], text[[0, 3]], rtol=0, atol=0)


def test_image_positions_must_hold_im_patch_tokens(model_parallel):
    model = _vl_model()
    input_ids = torch.tensor([[1, _IMAGE_TOKEN, 2]])
    text = model.embedding(input_ids=input_ids, position_ids=None)
    with pytest.raises(AssertionError, match="<im_patch>"):
        model._with_image_features(text, input_ids, torch.randn(1, 24), torch.tensor([2]))

"""Inkling screenshots through the TITO session server in token space.

The same path as the Qwen tests in test_token_space_images.py, with Inkling's token-space strategy: one
in-vocabulary placeholder per image, which the engine expands to a run of itself, one token per hMLP patch.
"""

import asyncio
import json

import pytest
import torch

from tests.fast.rollout.session.test_token_space_images import (
    TokenSpaceEngine,
    _chat,
    _data_url,
    _engine_args,
    _image_urls,
    _make_core,
    _model_files,
    _samples,
    _user,
)

MODEL = "thinkingmachines/Inkling-Small"
IMAGE_PLACEHOLDER = "<|unused_200054|>"
END_MESSAGE = "<|end_message|>"


@pytest.fixture(scope="module")
def model_path():
    return _model_files(MODEL)


@pytest.fixture(scope="module")
def tokenizer(model_path):
    """The Inkling tokenizer with the fixed template, as ``--tito-model inkling`` installs it."""
    from transformers import AutoTokenizer

    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, InklingTITOTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.chat_template = (TEMPLATE_DIR / InklingTITOTokenizer.FIXED_TEMPLATE.template).read_text(encoding="utf-8")
    return tok


@pytest.fixture(scope="module")
def trainer_processor(model_path):
    from miles.backends.training_utils.data.token_space_media_process import build_trainer_processor

    return build_trainer_processor(
        _engine_args(model_path), mm_processor_worker_num=1, mm_io_worker_num=1, device=torch.device("cpu")
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_inkling_images_flow_through_the_session_server_in_token_space(
    version, tokenizer, trainer_processor, tmp_path
):
    from miles.backends.training_utils.data.token_space_media_process import (
        compute_multimodal_train_inputs,
        resolved_multimodal_train_inputs,
    )

    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    engine = TokenSpaceEngine(trainer_processor.core, tokenizer, end_token=END_MESSAGE)
    core, registry = _make_core(version, tokenizer, engine, tmp_path, model=MODEL, tito_model="inkling")
    session_id = registry.create_session()

    turn1 = [_user("look", _data_url(64, 64, (200, 10, 10)))]
    response = asyncio.run(_chat(core, session_id, turn1))
    assert response.status_code == 200, response.body
    assistant = json.loads(response.body)["choices"][0]["message"]
    turn2 = turn1 + [assistant, _user("next", _data_url(96, 64, (5, 5, 5)))]
    assert asyncio.run(_chat(core, session_id, turn2)).status_code == 200

    body1, body2 = engine.bodies
    assert body1["mm_token_expansion_start_len"] == 0 and body1["input_ids"].count(image_id) == 1
    start = body2["mm_token_expansion_start_len"]
    assert body2["input_ids"][:start] == engine.prompts[0] + engine.output_ids, "turn 1 is sent expanded"
    assert body2["input_ids"][start:].count(image_id) == 1, "only the new image is left to expand"

    (sample,) = _samples(core, session_id, version)
    assert sample.tokens == engine.prompts[1] + engine.output_ids
    assert sample.tokens.count(image_id) > 2, "each placeholder became a run of itself, one token per patch"
    assert sample.multimodal_inputs == {"images": _image_urls(body2["messages"])}

    # The trainer rebuilds from the references and the expanded tokens what the engine computed.
    built = compute_multimodal_train_inputs(
        trainer_processor,
        [sample.multimodal_inputs],
        [torch.tensor(sample.tokens)],
        mm_processor_worker_num=1,
        with_sglang_placement=False,
        device=torch.device("cpu"),
    )
    (train_inputs,) = resolved_multimodal_train_inputs(built.multimodal_train_inputs)
    assert torch.equal(train_inputs["vision_patches_bthwc"], engine.outputs[-1]["vision_patches_bthwc"])

"""Server constraints, full model argument resolution, and the renderer projection."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.request_args import (
    apply_session_sampling_defaults,
    filter_turn_args,
    prepare_chat_request,
    resolve_request_args_by_config,
)
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, extract_template_args
from miles.utils.lora import LORA_ADAPTER_NAME


class TestResolveRequestArgsByConfig:
    def test_default_config_body_and_key_order(self):
        request_args = {"model": "m", "temperature": 0.7, "unknown": {"x": 1}, "messages": []}

        wire, _ = resolve_request_args_by_config(request_args, make_session_server_config())

        assert wire == {
            "model": "m",
            "temperature": 0.7,
            "unknown": {"x": 1},
            "messages": [],
            "logprobs": True,
            "return_meta_info": True,
            "no_stop_trim": False,
            "return_routed_experts": False,
            "return_indexer_topk": False,
        }
        assert list(wire)[:4] == ["model", "temperature", "unknown", "messages"]
        assert wire is request_args

    def test_replay_flags_follow_the_launch_flags(self):
        config = make_session_server_config(use_rollout_routing_replay=True, use_rollout_indexer_replay=True)
        wire, _ = resolve_request_args_by_config({"return_routed_experts": False}, config)
        assert wire["return_routed_experts"] is True
        assert wire["return_indexer_topk"] is True

    @pytest.mark.parametrize("field", ["input_ids", "routed_experts_start_len", "logprob_start_len", "lora_path"])
    def test_client_tito_control_fields_are_rejected(self, field):
        with pytest.raises(MessageValidationError, match=f"{field}="):
            resolve_request_args_by_config({field: 1}, make_session_server_config())

    def test_lora_path_follows_lora_rollout_enabled(self):
        wire, _ = resolve_request_args_by_config({}, make_session_server_config(lora_rank=8))
        assert wire["lora_path"] == LORA_ADAPTER_NAME
        wire, _ = resolve_request_args_by_config({}, make_session_server_config(lora_rank=8, lora_train_only=True))
        assert "lora_path" not in wire
        wire, _ = resolve_request_args_by_config({}, make_session_server_config())
        assert "lora_path" not in wire

    def test_model_adapter_suffix_is_rejected_only_with_lora_rollout(self):
        with pytest.raises(MessageValidationError, match="LoRA adapter"):
            resolve_request_args_by_config({"model": "base:adapter"}, make_session_server_config(lora_rank=8))
        wire, _ = resolve_request_args_by_config({"model": "base:adapter"}, make_session_server_config())
        assert wire["model"] == "base:adapter"

    @pytest.mark.parametrize("kwargs", ["oops", [], False, 1])
    def test_malformed_kwargs_are_refused(self, kwargs):
        with pytest.raises(MessageValidationError, match="chat_template_kwargs must be an object"):
            resolve_request_args_by_config({"chat_template_kwargs": kwargs}, make_session_server_config())

    def test_tools_must_be_top_level(self):
        with pytest.raises(MessageValidationError, match="tools belongs at the top level"):
            resolve_request_args_by_config({"chat_template_kwargs": {"tools": []}}, make_session_server_config())

    def test_control_field_errors_precede_template_shape_errors(self):
        with pytest.raises(MessageValidationError, match="input_ids="):
            resolve_request_args_by_config(
                {"input_ids": [1], "chat_template_kwargs": "oops"}, make_session_server_config()
            )

    @pytest.mark.parametrize("field", ["input_ids", "routed_experts_start_len", "logprob_start_len", "lora_path"])
    def test_null_control_fields_are_removed(self, field):
        wire, _ = resolve_request_args_by_config({field: None}, make_session_server_config())
        assert field not in wire

    def test_selected_lora_path_is_accepted(self):
        wire, _ = resolve_request_args_by_config(
            {"lora_path": LORA_ADAPTER_NAME}, make_session_server_config(lora_rank=8)
        )
        assert wire["lora_path"] == LORA_ADAPTER_NAME


class TestPrepareChatRequest:
    LAUNCH = {"enable_thinking": False}
    TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]

    @staticmethod
    def _tito(**kwargs) -> TITOTokenizer:
        return TITOTokenizer(MagicMock(), **kwargs)

    def test_wire_carries_the_template_args_the_prompt_is_rendered_with(self):
        client_args = {"messages": [], "tools": self.TOOLS, "chat_template_kwargs": {"enable_thinking": True}}

        prepared = prepare_chat_request(
            client_args,
            self._tito(chat_template_kwargs=self.LAUNCH),
            config=make_session_server_config(),
            turn_args=None,
        )

        assert prepared.template_args == {"enable_thinking": True, "tools": self.TOOLS}
        assert prepared.body["tools"] == self.TOOLS
        assert prepared.body["chat_template_kwargs"] == {"enable_thinking": True}
        assert prepared.body["logprobs"] is True  # resolve_request_args_by_config ran on the same body

    @pytest.mark.parametrize("client_tools", [None, [], [{"function": {"name": "override"}}]])
    def test_launch_tools_stay_top_level_and_client_tools_override_them(self, client_tools):
        launch_kwargs = {**self.LAUNCH, "tools": self.TOOLS}
        original = deepcopy(launch_kwargs)
        client_args = {} if client_tools is None else {"tools": client_tools}
        tito = self._tito(chat_template_kwargs=launch_kwargs)
        prepared = prepare_chat_request(client_args, tito, config=make_session_server_config(), turn_args=None)
        expected_tools = client_tools or self.TOOLS
        assert prepared.body["tools"] == expected_tools
        assert prepared.body["chat_template_kwargs"] == self.LAUNCH
        assert prepared.template_args == {**self.LAUNCH, "tools": expected_tools}
        continued = prepare_chat_request({}, tito, config=make_session_server_config(), turn_args=prepared.body)
        assert continued.body["tools"] == expected_tools
        assert continued.template_args == prepared.template_args
        assert launch_kwargs == original

    def test_inherited_tools_reach_the_wire_and_empty_args_are_retained(self):
        recorded = {"chat_template_kwargs": self.LAUNCH, "tools": self.TOOLS}
        prepared = prepare_chat_request(
            {"messages": [], "tools": []},
            self._tito(chat_template_kwargs=self.LAUNCH),
            config=make_session_server_config(),
            turn_args=recorded,
        )
        assert prepared.body["tools"] == self.TOOLS
        assert prepared.body["chat_template_kwargs"] == self.LAUNCH

        prepared = prepare_chat_request(
            {"messages": [], "tools": [], "chat_template_kwargs": {}},
            self._tito(),
            config=make_session_server_config(),
            turn_args=None,
        )
        assert prepared.template_args == {}
        assert prepared.body["tools"] is None
        assert prepared.body["chat_template_kwargs"] == {}

    def test_a_refused_request_is_a_400(self):
        with pytest.raises(MessageValidationError, match="tools changed") as excinfo:
            prepare_chat_request(
                {"messages": [], "tools": self.TOOLS},
                self._tito(chat_template_kwargs=self.LAUNCH),
                config=make_session_server_config(),
                turn_args={"chat_template_kwargs": self.LAUNCH},
            )
        assert excinfo.value.status_code == 400

    def test_model_receives_and_updates_full_args_without_mutating_client_input(self):
        class Model(TITOTokenizer):
            def resolve_request_args(self, request_args, *, turn_args):
                assert request_args["logprobs"] is True
                assert "stream" not in request_args
                assert super().resolve_request_args(request_args, turn_args=turn_args) is request_args
                request_args["temperature"] = 0.3
                request_args["unknown"]["values"].append(2)
                request_args["chat_template_kwargs"]["model_option"] = True
                return request_args

        client_args = {"temperature": 0.8, "unknown": {"values": [1]}, "stream": True}
        original = deepcopy(client_args)
        prepared = prepare_chat_request(
            client_args,
            Model(MagicMock(), chat_template_kwargs={"model_option": False}),
            config=make_session_server_config(),
            turn_args=None,
        )
        assert client_args == original
        assert prepared.body["temperature"] == 0.3
        assert prepared.body["unknown"] == {"values": [1, 2]}
        assert prepared.template_args == {"model_option": True}
        assert prepared.client_stream is True


def test_template_projection_only_selects_render_fields():
    request_args = {
        "model": "m",
        "temperature": 0.7,
        "input_ids": [1],
        "tools": [{"name": "f"}],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert extract_template_args(request_args) == {"enable_thinking": True, "tools": [{"name": "f"}]}
    assert extract_template_args({"temperature": 0.7}) == {}


@pytest.mark.parametrize("field", ["input_ids", "messages"])
def test_filter_turn_args_drops_payload_before_copying_and_isolates_metadata(field):
    payload = MagicMock()
    payload.__deepcopy__ = MagicMock(side_effect=AssertionError("excluded payload must not be copied"))
    turn_args = {field: payload, "temperature": 0.7, "chat_template_kwargs": {"nested": [1]}}

    metadata = filter_turn_args(turn_args)

    assert metadata == {"temperature": 0.7, "chat_template_kwargs": {"nested": [1]}}
    metadata["chat_template_kwargs"]["nested"].append(2)
    assert turn_args["chat_template_kwargs"] == {"nested": [1]}
    assert turn_args[field] is payload


def test_filter_turn_args_accepts_an_explicit_drop_list():
    turn_args = {"input_ids": [1], "messages": [{"role": "user", "content": "hi"}], "seed": 42}
    assert filter_turn_args(turn_args, drop_keys=("input_ids",)) == {"messages": turn_args["messages"], "seed": 42}
    assert filter_turn_args(turn_args, drop_keys=()) == turn_args


@pytest.mark.parametrize("sampling", [{}, {"temperature": 0.0}, {"temperature": 1.2, "top_p": 0.8, "top_k": 16}])
def test_eval_keeps_sampling_resolution_and_overrides_model_replay(sampling):
    class Model(TITOTokenizer):
        def resolve_request_args(self, request_args, *, turn_args):
            request_args = super().resolve_request_args(request_args, turn_args=turn_args)
            request_args.setdefault("temperature", 0.7)
            request_args.setdefault("top_p", 0.9)
            request_args.setdefault("top_k", 32)
            request_args.update(return_sampling_mask=True, return_routed_experts=True, return_indexer_topk=True)
            request_args["routed_experts_start_len"] = 7
            return request_args

    config = make_session_server_config(use_rollout_routing_replay=True, use_rollout_indexer_replay=True)
    client_args = {**sampling, "return_sampling_mask": True, "routed_experts_start_len": 99}
    history = {"temperature": 0.3, "chat_template_kwargs": {}}
    original, original_history = deepcopy(client_args), deepcopy(history)
    prepared = prepare_chat_request(client_args, Model(MagicMock()), config=config, turn_args=history, evaluation=True)
    assert {key: prepared.body[key] for key in ("temperature", "top_p", "top_k")} == {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 32,
        **sampling,
    }
    assert all(
        prepared.body[key] is False for key in ("return_sampling_mask", "return_routed_experts", "return_indexer_topk")
    )
    assert "routed_experts_start_len" not in prepared.body
    assert client_args == original and history == original_history


def test_session_sampling_defaults_fill_only_omitted_fields():
    """Session values land before model rules, so the resolved request carries them; explicit values win."""
    client_args = {"temperature": 0.1, "top_p": None}
    original = deepcopy(client_args)
    prepared = prepare_chat_request(
        client_args,
        TITOTokenizer(MagicMock()),
        config=make_session_server_config(),
        turn_args=None,
        sampling_defaults={"temperature": 0.6, "top_p": 0.9, "top_k": 20},
    )
    assert {key: prepared.body[key] for key in ("temperature", "top_p", "top_k")} == {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 20,
    }
    assert client_args == original


def test_without_session_sampling_defaults_omitted_fields_stay_unset():
    prepared = prepare_chat_request(
        {}, TITOTokenizer(MagicMock()), config=make_session_server_config(), turn_args=None
    )
    assert not {"temperature", "top_p", "top_k"} & set(prepared.body)


def test_apply_session_sampling_defaults_treats_none_as_unset():
    request_args = {"temperature": None, "top_k": -1}
    apply_session_sampling_defaults(request_args, {"temperature": 0.6, "top_p": 0.9, "top_k": 20})
    assert request_args == {"temperature": 0.6, "top_k": -1, "top_p": 0.9}


@pytest.mark.parametrize("evaluation", [False, True])
def test_template_compatibility_applies_to_train_and_eval(evaluation):
    with pytest.raises(MessageValidationError, match="tools changed"):
        prepare_chat_request(
            {"tools": [{"type": "function", "function": {"name": "new_tool"}}]},
            TITOTokenizer(MagicMock()),
            config=make_session_server_config(),
            turn_args={"chat_template_kwargs": {}},
            evaluation=evaluation,
        )

"""Unit tests for ``miles.rollout.session.request_args``: the two field functions,
``decide_chat_request_args``, which applies them to a client body, and
``prepare_chat_request``, which adds the template args and puts them on the wire."""

import logging
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.request_args import (
    decide_chat_request_args,
    prepare_chat_request,
    server_first,
    server_strict,
)
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer
from miles.utils.lora import LORA_ADAPTER_NAME

ARGS_LOGGER = "miles.rollout.session.request_args"


class TestServerFirst:
    def test_replaces_a_different_value_and_logs_why(self, caplog):
        wire = {"logprobs": False}
        with caplog.at_level(logging.WARNING, logger=ARGS_LOGGER):
            server_first(wire, "logprobs", True, why="TITO reads logprobs")
        assert wire == {"logprobs": True}
        assert "logprobs=False from the client replaced by True: TITO reads logprobs" in caplog.text

    def test_is_silent_when_the_client_agrees_or_says_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger=ARGS_LOGGER):
            for wire in ({"logprobs": True}, {}, {"logprobs": None}):
                server_first(wire, "logprobs", True, why="tito")
                assert wire == {"logprobs": True}
        assert caplog.text == ""

    def test_a_none_server_value_takes_the_field_off_the_wire(self, caplog):
        wire = {"flag": True}
        with caplog.at_level(logging.WARNING, logger=ARGS_LOGGER):
            server_first(wire, "flag", None, why="unset")
        assert wire == {}
        assert "flag=True from the client replaced by None: unset" in caplog.text


class TestServerStrict:
    def test_rejects_a_different_value_with_why(self):
        with pytest.raises(MessageValidationError) as excinfo:
            server_strict({"input_ids": [1, 2]}, "input_ids", None, why="rendered by the session server")
        assert str(excinfo.value) == "input_ids=[1, 2] is not accepted: rendered by the session server"
        assert excinfo.value.status_code == 400

    def test_accepts_the_same_value_or_silence(self):
        wire = {"lora_path": "adapter"}
        server_strict(wire, "lora_path", "adapter", why="training picks it")
        assert wire == {"lora_path": "adapter"}
        wire = {}
        server_strict(wire, "lora_path", "adapter", why="training picks it")
        assert wire == {"lora_path": "adapter"}

    def test_a_none_server_value_keeps_the_field_off_the_wire(self):
        wire = {"lora_path": None, "model": "m"}
        server_strict(wire, "lora_path", None, why="no lora")
        assert wire == {"model": "m"}


class TestDecideChatRequestArgs:
    def test_default_config_body_and_key_order(self):
        client = {"model": "m", "temperature": 0.7, "unknown": {"x": 1}, "messages": []}

        wire = decide_chat_request_args(client, make_session_server_config())

        assert wire == {
            **client,
            "logprobs": True,
            "return_meta_info": True,
            "no_stop_trim": False,
            "return_routed_experts": False,
            "return_indexer_topk": False,
        }
        assert list(wire)[:4] == ["model", "temperature", "unknown", "messages"]
        assert client == {"model": "m", "temperature": 0.7, "unknown": {"x": 1}, "messages": []}  # not mutated

    def test_replay_flags_follow_the_launch_flags(self):
        config = make_session_server_config(use_rollout_routing_replay=True, use_rollout_indexer_replay=True)
        wire = decide_chat_request_args({"return_routed_experts": False}, config)
        assert wire["return_routed_experts"] is True
        assert wire["return_indexer_topk"] is True

    @pytest.mark.parametrize("field", ["input_ids", "routed_experts_start_len", "logprob_start_len", "lora_path"])
    def test_client_tito_control_fields_are_rejected(self, field):
        with pytest.raises(MessageValidationError, match=f"{field}="):
            decide_chat_request_args({field: 1}, make_session_server_config())

    def test_lora_path_follows_lora_rollout_enabled(self):
        assert decide_chat_request_args({}, make_session_server_config(lora_rank=8))["lora_path"] == LORA_ADAPTER_NAME
        assert "lora_path" not in decide_chat_request_args(
            {}, make_session_server_config(lora_rank=8, lora_train_only=True)
        )
        assert "lora_path" not in decide_chat_request_args({}, make_session_server_config())

    def test_model_adapter_suffix_is_rejected_only_with_lora_rollout(self):
        with pytest.raises(MessageValidationError, match="LoRA adapter"):
            decide_chat_request_args({"model": "base:adapter"}, make_session_server_config(lora_rank=8))
        assert (
            decide_chat_request_args({"model": "base:adapter"}, make_session_server_config())["model"]
            == "base:adapter"
        )


class TestPrepareChatRequest:
    LAUNCH = {"enable_thinking": False}
    TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]

    @staticmethod
    def _tito(**kwargs) -> TITOTokenizer:
        return TITOTokenizer(MagicMock(), **kwargs)

    def test_wire_carries_the_template_args_the_prompt_is_rendered_with(self):
        client = {"messages": [], "tools": self.TOOLS, "chat_template_kwargs": {"enable_thinking": True}}

        prepared = prepare_chat_request(
            client, self._tito(chat_template_kwargs=self.LAUNCH), config=make_session_server_config(), turn_args=None
        )

        assert prepared.template_args == {"enable_thinking": True, "tools": self.TOOLS}
        assert prepared.body["tools"] == self.TOOLS
        assert prepared.body["chat_template_kwargs"] == {"enable_thinking": True}
        assert prepared.body["logprobs"] is True  # decide_chat_request_args ran on the same body

    def test_inherited_tools_reach_the_wire_and_empty_template_args_leave_it(self):
        recorded = {**self.LAUNCH, "tools": self.TOOLS}
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
        assert "tools" not in prepared.body and "chat_template_kwargs" not in prepared.body

    def test_a_refused_request_is_a_400(self):
        with pytest.raises(MessageValidationError, match="was rendered with") as excinfo:
            prepare_chat_request(
                {"messages": [], "chat_template_kwargs": {"enable_thinking": True}},
                self._tito(chat_template_kwargs=self.LAUNCH),
                config=make_session_server_config(),
                turn_args=self.LAUNCH,
            )
        assert excinfo.value.status_code == 400

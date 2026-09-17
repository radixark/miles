"""Server-owned request fields and input validation."""

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.request_args import resolve_request_args_by_config
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

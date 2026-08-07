from __future__ import annotations

from argparse import Namespace

import pytest
from pydantic import ValidationError

from miles.rollout.session.config import SessionServerConfig, compute_session_server_config


_ARGS_TO_CONFIG_FIELD = {
    "miles_router_timeout": "timeout",
    "hf_checkpoint": "hf_checkpoint",
    "chat_template_path": "chat_template_path",
    "tito_model": "tito_model",
    "apply_chat_template_kwargs": "apply_chat_template_kwargs",
    "use_rollout_routing_replay": "use_rollout_routing_replay",
    "use_rollout_indexer_replay": "use_rollout_indexer_replay",
    "sglang_speculative_algorithm": "sglang_speculative_algorithm",
    "num_layers": "num_layers",
    "moe_router_topk": "moe_router_topk",
    "save_debug_trajectory_data": "save_debug_trajectory_data",
    "lora_rank": "lora_rank",
    "lora_adapter_path": "lora_adapter_path",
}

_CALL_SITE_FIELDS = ("host", "port", "instance_id", "backend_url")

_DISTINCT_ARGS_VALUES = dict(
    miles_router_timeout=31.5,
    hf_checkpoint="/fake/model",
    chat_template_path="/fake/chat_template.jinja",
    tito_model="tito-xyz",
    apply_chat_template_kwargs={"enable_thinking": True},
    use_rollout_routing_replay=True,
    use_rollout_indexer_replay=True,
    sglang_speculative_algorithm="EAGLE",
    num_layers=61,
    moe_router_topk=8,
    save_debug_trajectory_data="/fake/traj",
    lora_rank=32,
    lora_adapter_path="/fake/adapters/x",
)


def _make_args(**overrides) -> Namespace:
    defaults = dict(_DISTINCT_ARGS_VALUES)
    defaults.update(overrides)
    return Namespace(**defaults)


class TestComputeSessionServerConfig:
    def test_every_config_field_has_a_known_source(self):
        """Adding a config field without extending this test's mapping must fail here."""
        covered = set(_CALL_SITE_FIELDS) | set(_ARGS_TO_CONFIG_FIELD.values())
        assert covered == set(SessionServerConfig.model_fields)

    def test_call_site_fields_are_copied(self):
        """Host, port, instance id, and backend url come from the caller, not from args."""
        config = compute_session_server_config(
            _make_args(), host="10.0.0.1", port=5001, instance_id="abc", backend_url="http://10.0.0.2:3000"
        )
        assert config.host == "10.0.0.1"
        assert config.port == 5001
        assert config.instance_id == "abc"
        assert config.backend_url == "http://10.0.0.2:3000"

    @pytest.mark.parametrize("args_attr,config_field", sorted(_ARGS_TO_CONFIG_FIELD.items()))
    def test_each_args_field_is_copied(self, args_attr: str, config_field: str):
        """Every args-derived field maps one-to-one, using values no hardcoded default would produce."""
        config = compute_session_server_config(
            _make_args(), host="10.0.0.1", port=5001, instance_id="abc", backend_url="http://10.0.0.2:3000"
        )
        assert getattr(config, config_field) == _DISTINCT_ARGS_VALUES[args_attr]


_COMPLETE_CONFIG_KWARGS = dict(
    host="127.0.0.1",
    port=5001,
    instance_id=None,
    backend_url="http://127.0.0.1:3000",
    timeout=None,
    hf_checkpoint=None,
    chat_template_path=None,
    tito_model="default",
    apply_chat_template_kwargs=None,
    use_rollout_routing_replay=False,
    use_rollout_indexer_replay=False,
    sglang_speculative_algorithm=None,
    num_layers=None,
    moe_router_topk=None,
    save_debug_trajectory_data=None,
    lora_rank=0,
    lora_adapter_path=None,
)


class TestSessionServerConfig:
    def test_complete_kwargs_cover_every_field(self):
        """The per-field omission cases are only meaningful if the full kwargs validate."""
        assert set(_COMPLETE_CONFIG_KWARGS) == set(SessionServerConfig.model_fields)
        assert SessionServerConfig(**_COMPLETE_CONFIG_KWARGS).port == 5001

    @pytest.mark.parametrize("missing", sorted(SessionServerConfig.model_fields))
    def test_every_field_is_required(self, missing: str):
        """Omitting any single field must fail, so none can be silently defaulted."""
        kwargs = {name: value for name, value in _COMPLETE_CONFIG_KWARGS.items() if name != missing}
        with pytest.raises(ValidationError):
            SessionServerConfig(**kwargs)

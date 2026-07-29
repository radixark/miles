from __future__ import annotations

from argparse import Namespace

import pytest
from pydantic import ValidationError

from miles.rollout.session.config import SessionServerConfig, compute_session_server_config


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        miles_router_timeout=None,
        hf_checkpoint="/fake/model",
        chat_template_path=None,
        tito_model="default",
        apply_chat_template_kwargs={},
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
        num_layers=None,
        moe_router_topk=None,
        save_debug_trajectory_data=None,
        lora_rank=0,
        lora_adapter_path=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestComputeSessionServerConfig:
    def test_fields_are_copied_from_args_and_call_site(self):
        """Args fields map one-to-one; host, port, and instance id come from the caller."""
        config = compute_session_server_config(
            _make_args(miles_router_timeout=30.0, use_rollout_routing_replay=True),
            host="10.0.0.1",
            port=5001,
            instance_id="abc",
            backend_url="http://10.0.0.2:3000",
        )
        assert config.host == "10.0.0.1"
        assert config.port == 5001
        assert config.instance_id == "abc"
        assert config.backend_url == "http://10.0.0.2:3000"
        assert config.timeout == 30.0
        assert config.hf_checkpoint == "/fake/model"
        assert config.tito_model == "default"
        assert config.use_rollout_routing_replay is True
        assert config.use_rollout_indexer_replay is False


class TestSessionServerConfig:
    def test_every_field_is_required(self):
        """No field may be silently defaulted, including the nullable ones."""
        with pytest.raises(ValidationError):
            SessionServerConfig(
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
            )

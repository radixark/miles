from argparse import Namespace
from types import SimpleNamespace

import pytest

import miles.rollout.generate_utils.weight_version_partition as weight_version_partition
import miles.rollout.sglang_rollout as sglang_rollout
from miles.rollout.generate_utils.weight_version_partition import (
    WEIGHT_VERSION_EXTRA_KEY_METADATA_KEY,
    format_weight_version_extra_key,
    latest_weight_version,
    observe_weight_version,
)
from miles.utils.types import Sample


def test_format_maps_none_to_version_zero():
    """A never-updated engine and version 0 share one namespace."""
    assert format_weight_version_extra_key(None) == "weight-version:0"
    assert format_weight_version_extra_key(7) == "weight-version:7"


def test_observe_keeps_monotonic_max(monkeypatch):
    """Belief only moves forward, even when a lagging engine reports an older version."""
    monkeypatch.setattr(weight_version_partition, "_latest_weight_version", None)
    observe_weight_version({})
    assert latest_weight_version() is None
    observe_weight_version({"weight_version": "3"})
    assert latest_weight_version() == 3
    observe_weight_version({"weight_version": "2"})
    assert latest_weight_version() == 3
    observe_weight_version({"weight_version": 5})
    assert latest_weight_version() == 5


def test_observe_ignores_unparseable_versions(monkeypatch):
    """Non-numeric engine versions leave the belief untouched."""
    monkeypatch.setattr(weight_version_partition, "_latest_weight_version", 4)
    observe_weight_version({"weight_version": "default"})
    assert latest_weight_version() == 4
    observe_weight_version({"weight_version": None})
    assert latest_weight_version() == 4


class _FakeTokenizer:
    def encode(self, prompt: str, add_special_tokens: bool) -> list[int]:
        return [1, 2, 3]


def _make_args() -> Namespace:
    return Namespace(
        ci_test=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_policy="round_robin",
        sglang_speculative_algorithm=None,
        use_rollout_routing_replay=False,
        lora_rank=0,
        lora_configs=None,
    )


def _make_state(args: Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        args=args,
        tokenizer=_FakeTokenizer(),
        processor=None,
    )


def _install_fake_post(monkeypatch: pytest.MonkeyPatch, payloads: list[dict], weight_version: str):
    async def fake_post(url, payload, headers=None):
        payloads.append(payload)
        return {
            "text": "ok",
            "meta_info": {
                "weight_version": weight_version,
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [(-0.1, 11)],
            },
        }

    monkeypatch.setattr(sglang_rollout, "post", fake_post)


@pytest.mark.asyncio
async def test_generate_locks_extra_key_across_turns(monkeypatch):
    """A sample's first turn tags the current belief and later turns reuse it even after the belief advances."""
    args = _make_args()
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _make_state(args))
    monkeypatch.setattr(weight_version_partition, "_latest_weight_version", None)
    payloads: list[dict] = []
    _install_fake_post(monkeypatch, payloads, weight_version="4")

    sample = Sample(prompt="q")
    await sglang_rollout.generate(args, sample, {"max_new_tokens": 8})
    assert payloads[0]["extra_key"] == "weight-version:0"
    assert latest_weight_version() == 4

    sample.status = Sample.Status.PENDING
    await sglang_rollout.generate(args, sample, {"max_new_tokens": 8})
    assert payloads[1]["extra_key"] == "weight-version:0"
    assert sample.metadata[WEIGHT_VERSION_EXTRA_KEY_METADATA_KEY] == "weight-version:0"


@pytest.mark.asyncio
async def test_generate_tags_new_samples_with_latest_belief(monkeypatch):
    """A fresh sample starts in the namespace of the latest observed weight version."""
    args = _make_args()
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: _make_state(args))
    monkeypatch.setattr(weight_version_partition, "_latest_weight_version", 9)
    payloads: list[dict] = []
    _install_fake_post(monkeypatch, payloads, weight_version="9")

    await sglang_rollout.generate(args, Sample(prompt="q"), {"max_new_tokens": 8})
    assert payloads[0]["extra_key"] == "weight-version:9"

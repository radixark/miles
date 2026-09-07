import asyncio
import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout import sglang_rollout
from miles.utils.types import AdapterRef, Sample


def _make_generate_args() -> Namespace:
    return Namespace(
        ci_test=False,
        sglang_router_ip="router",
        sglang_router_port=30000,
        sglang_router_policy="round_robin",
        sglang_speculative_algorithm=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        lora_rank=0,
        lora_adapter_path=None,
    )


class TestGenerateExtraKey:
    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        payloads: list[dict] = []
        tokenizer = SimpleNamespace(encode=lambda prompt, add_special_tokens: [1, 2, 3])
        state = SimpleNamespace(tokenizer=tokenizer, processor=None)

        async def fake_post(url: str, payload: dict, headers: dict | None = None) -> dict:
            payloads.append(payload)
            return {"text": "", "meta_info": {"finish_reason": {"type": "stop"}}}

        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda state_args: state)
        monkeypatch.setattr(sglang_rollout, "post", fake_post)
        return payloads

    async def test_a_started_sample_partitions_the_cache_by_its_kv_cache_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default generate path sends extra_key the namespace itself for a started sample."""
        payloads = self._patch(monkeypatch)
        sample = Sample(prompt="p", kv_cache_namespace="train:-:7")

        await sglang_rollout.generate(_make_generate_args(), sample, {"max_new_tokens": 4})

        assert [payload["extra_key"] for payload in payloads] == ["train:-:7"]

    async def test_an_unstarted_sample_sends_no_extra_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a kv_cache_namespace the payload is not partitioned."""
        payloads = self._patch(monkeypatch)

        await sglang_rollout.generate(_make_generate_args(), Sample(prompt="p"), {"max_new_tokens": 4})

        assert len(payloads) == 1
        assert "extra_key" not in payloads[0]

    async def test_the_multi_lora_adapter_key_wins_over_the_kv_cache_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-LoRA sample keeps its adapter:v<version> key instead of the namespace one."""
        payloads = self._patch(monkeypatch)

        class FakeAdaptersCache:
            async def get(self, name: str) -> SimpleNamespace:
                return SimpleNamespace(version=3)

        monkeypatch.setattr("miles.ray.multi_lora.controller.AdaptersCache", FakeAdaptersCache)
        sample = Sample(prompt="p", kv_cache_namespace="train:-:7", adapter=AdapterRef(name="alpha", slot=0))

        await sglang_rollout.generate(_make_generate_args(), sample, {"max_new_tokens": 4})

        assert [payload["extra_key"] for payload in payloads] == ["alpha:v3"]


class TestAbort:
    async def test_abort_survives_one_unresponsive_worker_and_finishes_cleanup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One failed worker abort still runs the agent hook and drains partial samples."""
        args = Namespace(
            partial_rollout=True,
            sglang_router_ip="router",
            sglang_router_port=30000,
            use_miles_router=True,
        )
        sample = SimpleNamespace(response="partial", metadata={})
        state = SimpleNamespace(args=args, aborted=False)
        hook_calls: list[Namespace] = []
        posted_urls: list[str] = []
        hook_finished = asyncio.Event()

        async def fake_get(url: str) -> dict[str, list[str]]:
            return {"urls": ["http://healthy", "http://unresponsive"]}

        async def fake_post(url: str, payload: dict[str, bool]) -> None:
            posted_urls.append(url)
            if "unresponsive" in url:
                raise ConnectionError("worker cannot answer")

        async def fake_agent_abort_hook(hook_args: Namespace) -> None:
            hook_calls.append(hook_args)
            hook_finished.set()

        async def finish_group() -> list[SimpleNamespace]:
            await hook_finished.wait()
            return [sample]

        state.pendings = {asyncio.create_task(finish_group())}
        await asyncio.sleep(0)
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda state_args: state)
        monkeypatch.setattr(sglang_rollout, "get", fake_get)
        monkeypatch.setattr(sglang_rollout, "post", fake_post)
        monkeypatch.setattr(sglang_rollout, "call_agent_abort_hook", fake_agent_abort_hook)

        with caplog.at_level(logging.WARNING, logger=sglang_rollout.__name__):
            aborted_groups = await sglang_rollout.abort(args, rollout_id=23)

        assert posted_urls == ["http://healthy/abort_request", "http://unresponsive/abort_request"]
        assert hook_calls == [args]
        assert aborted_groups == [[sample]]
        assert sample.metadata["start_rollout_id"] == 23
        assert "Failed to abort worker at http://unresponsive: worker cannot answer" in caplog.text

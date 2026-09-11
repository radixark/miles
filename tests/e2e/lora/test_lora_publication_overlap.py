"""Real Miles NCCL and IPC publication while A@1/B/C keep decoding.

The tensor source is synthetic; WeightUpdater, transport, SGLang and generation
are real. This checks publication, not optimizer or Tinker SDK training E2E.
"""

import asyncio
import concurrent.futures
import json
import os
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.utils.sglang_server import start_sglang_server

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase, WeightUpdatePlacement
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo

register_cuda_ci(est_time=240, suite="stage-c-2-gpu-h200", labels=["lora"], hardware=["hopper"])

_NAMES = ("A@1", "B", "C")
_UPDATES = (("A@2", 4), ("B@2", 5))
_TOKENS = 4096


class _AdapterIterator(HfWeightIteratorBase):
    def _iter_hf_param_units(self, weights, *, materialize):
        raise AssertionError("Publishing an adapter must not read the base")

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        if not materialize:
            return
        rng = torch.Generator().manual_seed(adapter.seed)
        for layer in range(28):
            for module, width in (("q_proj", 2048), ("k_proj", 1024), ("v_proj", 1024)):
                prefix = f"model.layers.{layer}.self_attn.{module}"
                a = torch.randn(8, 1024, generator=rng, dtype=torch.bfloat16) * 0.04
                b = torch.randn(width, 8, generator=rng, dtype=torch.bfloat16) * 0.04
                yield [(f"{prefix}.lora_A.weight", a.cuda()), (f"{prefix}.lora_B.weight", b.cuda())]


@dataclass(frozen=True)
class _PublicationClient(SGLangApiClient):
    # The name streaming in the open staged session (register→end), and the
    # names whose sessions already ran the mid-stream probe.
    _streaming: list = field(default_factory=list, compare=False)
    _probed_sessions: set[str] = field(default_factory=set, compare=False)

    async def _make_request(self, endpoint, payload=None):
        assert endpoint not in {"pause_generation", "continue_generation", "flush_cache", "update_weight_version"}
        started = time.monotonic()
        result = await super()._make_request(endpoint, payload)
        if endpoint == "register_lora_adapter" and (payload or {}).get("defer_publish"):
            self._streaming.append(payload["lora_name"])
        session_id = self._streaming[-1] if self._streaming else None
        if endpoint == "end_weight_update":
            self._streaming.clear()
        if session_id in {name for name, _ in _UPDATES}:
            print(f"{self.server_url} {endpoint}: {time.monotonic() - started:.3f}s", flush=True)
            if endpoint in {"update_weights_from_tensor", "update_weights_from_distributed"}:
                if session_id not in self._probed_sessions:
                    self._probed_sessions.add(session_id)
                    await asyncio.to_thread(self._probe_new_c_request, session_id)
        return result

    def _probe_new_c_request(self, session_id):
        # A bucket has arrived, but this session cannot finish before C does.
        response = requests.post(
            f"{self.server_url}/generate",
            json={
                "text": "Continue counting integers, separated by commas: 1, 2, 3,",
                "lora_path": "C",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32, "ignore_eos": True},
            },
            timeout=30,
        )
        response.raise_for_status()
        assert response.json()["meta_info"]["completion_tokens"] == 32, response.text
        print(
            json.dumps({"engine": self.server_url, "new_C_completed_during_session": session_id}),
            flush=True,
        )


def _make_updater(urls, colocate):
    singleton = GroupInfo(rank=0, size=1, group=None)
    replicas = GroupInfo(rank=dist.get_rank(), size=dist.get_world_size(), group=dist.group.WORLD)
    parallel = ParallelState(
        intra_dp=replicas,
        intra_dp_cp=replicas,
        cp=singleton,
        tp=singleton,
        pp=singleton,
        ep=singleton,
        etp=singleton,
        indep_dp=singleton,
    )
    set_parallel_state(parallel)
    args = Namespace(
        colocate=colocate,
        update_weight_transfer_mode="broadcast",
        rollout_num_gpus_per_engine=1,
        actor_num_nodes=1,
        actor_num_gpus_per_node=dist.get_world_size(),
        offload_rollout=False,
        lora_rank=8,
        check_weight_update_equal=False,
        check_lora_weight_equal=False,
        update_weight_buffer_size=512 * 1024,
        pause_generation_mode="retract",
    )
    updater = WeightUpdater(
        args,
        [],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
        iterator_factory=lambda args, model, required_placement, **kwargs: _AdapterIterator(
            args, model, placement=WeightUpdatePlacement(gather_pp=True), **kwargs
        ),
        parallel_state=parallel,
        is_lora=True,
        lora_sync_config={
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 8,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
        },
    )
    updater.connect_rollout_engines([_PublicationClient(url) for url in urls])
    return updater


def _stream(url, name, started, progress):
    tokens = []
    with requests.post(
        f"{url}/generate",
        json={
            "text": "Continue counting integers, separated by commas: 1, 2, 3,",
            "lora_path": name,
            "stream": True,
            "sampling_params": {"temperature": 0, "max_new_tokens": _TOKENS, "ignore_eos": True},
        },
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1024):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            assert "error" not in chunk, chunk
            tokens.extend(chunk.get("output_ids", []))
            progress[(url, name)] = len(tokens)
            if len(tokens) >= 8:
                started.set()
    assert len(tokens) == _TOKENS, len(tokens)
    return tokens, time.monotonic()


def _publish_with_live_requests(updater, urls):
    pairs = [(url, name) for url in urls for name in _NAMES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        baseline = {pair: pool.submit(_stream, *pair, threading.Event(), {}) for pair in pairs}
        expected = {pair: future.result(timeout=120)[0] for pair, future in baseline.items()}
        started = {pair: threading.Event() for pair in pairs}
        progress = {}
        active = {pair: pool.submit(_stream, *pair, started[pair], progress) for pair in pairs}
        for event in started.values():
            assert event.wait(30), "Generation did not start"
        assert all(not future.done() for future in active.values())
        dist.barrier()
        begin = time.monotonic()
        publications = []
        for name, seed in _UPDATES:
            updater.push_adapter(name, SimpleNamespace(rank=8, alpha=8, seed=seed))
            assert all(not future.done() for future in active.values()), progress
            publications.append({"name": name, "elapsed": time.monotonic() - begin})
        published = time.monotonic()
        assert all(not future.done() for future in active.values()), progress
        progress_at_publish = {f"engine-{urls.index(url)}/{name}": count for (url, name), count in progress.items()}
        for url in urls:
            for name, _ in _UPDATES:
                response = requests.post(
                    f"{url}/generate",
                    json={
                        "text": "Hello",
                        "lora_path": name,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                    },
                    timeout=30,
                )
                response.raise_for_status()
        for pair, future in active.items():
            tokens, finished = future.result(timeout=120)
            assert published < finished, pair
            assert tokens == expected[pair], f"Old version changed during publication: {pair}"
    print(
        json.dumps(
            {
                "publication_seconds": published - begin,
                "publications": publications,
                "tokens_at_publish": progress_at_publish,
                "engines": len(urls),
                "old_version_tokens_unchanged": True,
            }
        ),
        flush=True,
    )


def _rank_main(rank, urls, store_path, colocate):
    torch.cuda.set_device(rank)
    monkey_patch_torch_reductions()
    dist.init_process_group(
        "nccl",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2 if colocate else 1,
        timeout=timedelta(seconds=90),
    )
    init_gloo_group()
    try:
        updater = _make_updater(urls, colocate)
        for seed, name in enumerate(_NAMES, start=1):
            updater.push_adapter(name, SimpleNamespace(rank=8, alpha=8, seed=seed))
        if rank == 0:
            _publish_with_live_requests(updater, urls)
        else:
            dist.barrier()
            for name, seed in _UPDATES:
                updater.push_adapter(name, SimpleNamespace(rank=8, alpha=8, seed=seed))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _stop_server(server):
    kill_process_tree(server.process.pid)
    server.stop()


class TestLoRAPublicationOverlap(unittest.TestCase):
    def setUp(self):
        # Both processes must inherit the same NCCL options before importing CUDA.
        env = patch.dict(os.environ, {"NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_two_engine_publication_overlaps_existing_generations(self):
        self._run(colocate=True)

    def test_distributed_publication_overlaps_existing_generations(self):
        self._run(colocate=False)

    def _run(self, colocate):
        urls = []
        for gpu in range(2) if colocate else [1]:
            server = start_sglang_server(
                model_path=os.environ.get("MILES_LORA_PUBLICATION_TEST_MODEL", "Qwen/Qwen3-0.6B"),
                enable_deterministic_inference=False,
                extra_args=[
                    "--base-gpu-id",
                    str(gpu),
                    "--enable-lora",
                    "--max-lora-rank",
                    "8",
                    "--max-loras-per-batch",
                    "4",
                    "--lora-target-modules",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "--lora-backend",
                    "triton",
                    "--mem-fraction-static",
                    "0.15",
                    "--context-length",
                    "8192",
                    "--cuda-graph-max-bs-decode",
                    "4",
                    "--max-running-requests",
                    "8",
                    "--incremental-streaming-output",
                    "--random-seed",
                    "11",
                ],
            )
            self.addCleanup(_stop_server, server)
            urls.append(server.base_url)
        with tempfile.TemporaryDirectory(prefix="miles-lora-overlap-") as directory:
            workers = mp.spawn(
                _rank_main, args=(urls, f"{directory}/rendezvous", colocate), nprocs=2 if colocate else 1, join=False
            )
            try:
                deadline = time.monotonic() + 180
                while not workers.join(timeout=1):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Publication workers did not finish within 180 seconds")
            finally:
                for process in workers.processes:
                    if process.is_alive():
                        process.terminate()
                for process in workers.processes:
                    process.join(timeout=10)


if __name__ == "__main__":
    unittest.main()

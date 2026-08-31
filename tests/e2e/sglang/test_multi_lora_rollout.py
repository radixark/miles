"""Rollout-to-handoff E2E: real SGLang multi-LoRA generation through whole-batch admission; no trainer/optimizer/publication."""

from __future__ import annotations

import json
import math
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=420, suite="stage-b-2-gpu-h200", labels=["sglang"])

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

MODEL = "Qwen/Qwen3-0.6B"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_zero_lora(model_name: str, out_dir: Path, rank: int = 4) -> Path:
    from safetensors.torch import save_file

    from miles.utils.hf_config import load_hf_config

    config = load_hf_config(model_name, trust_remote_code=True)
    hidden = config.hidden_size
    head_dim = getattr(config, "head_dim", hidden // config.num_attention_heads)
    q_out = config.num_attention_heads * head_dim
    v_out = config.num_key_value_heads * head_dim
    tensors = {}
    for layer in range(config.num_hidden_layers):
        prefix = f"base_model.model.model.layers.{layer}.self_attn"
        tensors[f"{prefix}.q_proj.lora_A.weight"] = torch.zeros(rank, hidden, dtype=torch.bfloat16)
        tensors[f"{prefix}.q_proj.lora_B.weight"] = torch.zeros(q_out, rank, dtype=torch.bfloat16)
        tensors[f"{prefix}.v_proj.lora_A.weight"] = torch.zeros(rank, hidden, dtype=torch.bfloat16)
        tensors[f"{prefix}.v_proj.lora_B.weight"] = torch.zeros(v_out, rank, dtype=torch.bfloat16)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    adapter_config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": 2 * rank,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "base_model_name_or_path": model_name,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(adapter_config))
    return out_dir


@pytest.fixture(scope="module")
def sglang_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlrollout")
    lora_dir = _write_zero_lora(MODEL, root / "lora")
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--enable-lora",
            "--lora-paths",
            f"__miles_slot_0={lora_dir}",
            f"__miles_slot_1={lora_dir}",
            "--max-loras-per-batch",
            "2",
            "--max-lora-rank",
            "4",
            "--lora-target-modules",
            "q_proj",
            "v_proj",
            "--mem-fraction-static",
            "0.4",
            "--disable-cuda-graph",
            "--log-level",
            "warning",
        ],
    )
    try:
        deadline = time.time() + 600
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"sglang server exited early with {proc.returncode}") from None
                assert time.time() < deadline, "sglang server did not become healthy in time"
                time.sleep(2)
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def make_args(port: int, prompt_data: str):
    argv = [
        "pytest",
        "--train-backend",
        "fsdp",
        "--rollout-batch-size",
        "4",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--hf-checkpoint",
        MODEL,
        "--prompt-data",
        prompt_data,
        "--rm-type",
        "math",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        str(port),
        "--rollout-max-response-len",
        "8",
        "--global-batch-size",
        "64",
    ]
    from miles.utils.arguments import parse_args
    from miles.utils.http_utils import init_http_client

    with patch("sys.argv", argv):
        args = parse_args()
    args.multi_lora_dp_size = 1
    args.multi_lora_max_coalesce_wait_s = 0.2
    args.max_weight_staleness = None
    init_http_client(args)
    return args


def make_run(name: str, slot: int, data: str, rollout_batch_size: int, version: int = 1):
    from miles.utils.adapter_config import AdapterRun, AdapterRunConfig

    config = AdapterRunConfig(
        data=data, rank=4, alpha=8, rollout_batch_size=rollout_batch_size, n_samples_per_prompt=2
    )
    return AdapterRun(name=name, config=config, slot=slot, version=version, step=0, registration_id=f"reg-{name}")


def wait_until(condition, timeout=120.0, message="condition"):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, f"timeout waiting for {message}"
        time.sleep(0.2)


def test_multi_lora_rollout(sglang_server, tmp_path, monkeypatch):
    import miles.rollout.multi_lora.async_rollout as mlar
    import miles.rollout.multi_lora.data_source as mlds
    from miles.rollout.multi_lora.async_rollout import AsyncMultiLoRAWorker, generate_rollout_multi_lora_async
    from miles.rollout.sglang_rollout import call_sglang_generate_endpoint, parse_output_token_logprobs
    from miles.utils.async_utils import run
    from miles.utils.multi_lora import slot_lora_name
    from miles.utils.processing_utils import load_tokenizer
    from miles.utils.types import Sample

    for name in ("a", "b"):
        rows = [{"text": f"{name} question {i}: what is {i}+{i}?"} for i in range(8)]
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    args = make_args(sglang_server, str(tmp_path / "a.jsonl"))
    tokenizer = load_tokenizer(MODEL, trust_remote_code=True)

    runs = {
        "A": make_run("A", 0, str(tmp_path / "a.jsonl"), rollout_batch_size=2),
        "B": make_run("B", 1, str(tmp_path / "b.jsonl"), rollout_batch_size=3),
    }

    def snapshot():
        return {"active": dict(runs), "retiring": {}, "cleanup": []}

    monkeypatch.setattr(mlds, "fetch_snapshot", snapshot)
    monkeypatch.setattr(mlar, "fetch_snapshot", snapshot)

    class FakeCache:
        async def get(self, name):
            return runs.get(name)

    class _Remote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *call_args, **call_kwargs):
            async def _call():
                return self._fn(*call_args, **call_kwargs)

            return _call()

    controller = SimpleNamespace(
        snapshot=_Remote(lambda: snapshot()),
        record_batch_adapters=_Remote(lambda *a: None),
        resolve_num_step=SimpleNamespace(remote=lambda *a: None),
    )
    monkeypatch.setattr(mlar, "AdaptersCache", FakeCache)
    monkeypatch.setattr(mlar, "get_multi_lora_controller", lambda: controller)
    monkeypatch.setattr(mlar, "tracking", SimpleNamespace(log=lambda *a, **k: None))
    monkeypatch.setattr(mlds, "get_multi_lora_controller", lambda: controller)
    monkeypatch.setattr(mlds, "ray", SimpleNamespace(get=lambda ref: None))

    async def real_generate(gen_args, group, sampling_params):
        for sample in group:
            ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
            out = await call_sglang_generate_endpoint(
                gen_args,
                input_ids=ids,
                sampling_params={"max_new_tokens": 8, "temperature": 0.0},
                lora_path=slot_lora_name(sample.adapter.slot),
            )
            tokens, log_probs = parse_output_token_logprobs(out)
            assert tokens and len(log_probs) == len(tokens)
            assert all(math.isfinite(lp) for lp in log_probs)
            sample.tokens = ids + tokens
            sample.response = out["text"]
            sample.response_length = len(tokens)
            sample.rollout_log_probs = log_probs
            sample.status = Sample.Status.COMPLETED
            sample.reward = 1.0
        return group

    data_source = mlds.MultiLoRAAsyncDataSource(args)
    worker = AsyncMultiLoRAWorker.get_or_create(args, data_source, real_generate)
    try:
        quotas = {"A": 2, "B": 3}
        seen_sizes: list[dict[str, int]] = []

        def full():
            sizes = worker.queue_sizes()
            seen_sizes.append(sizes)
            return sizes.get("A", 0) == 2 and sizes.get("B", 0) == 3 and not worker.inflight_groups

        wait_until(full, message="both adapters to fill their whole batch")
        assert all(
            sizes.get(name, 0) <= quota for sizes in seen_sizes for name, quota in quotas.items()
        ), "producer overfilled an adapter beyond its open batch quota"

        straggler = worker.buffers["B"].get(1)[0]
        groups, counts = worker.get_groups(snapshot(), 1024, {})
        assert counts == {"A": 2}, "a partial adapter batch must never reach the handoff"
        assert worker.awaiting_publication["A"].version == 1
        worker.buffers["B"].put(straggler)

        output = run(generate_rollout_multi_lora_async(args, 0, data_source, real_generate))
        b_groups = output.samples
        assert len(b_groups) == 3 and {g[0].adapter.name for g in b_groups} == {"B"}
        head = b_groups[0][0]
        assert head.metadata["step_adapter_names"] == ["B"]
        assert head.metadata["step_slots"] == [1]
        for group in b_groups:
            for sample in group:
                assert sample.metadata["adapter_global_batch_size"] == 6
                assert sample.metadata["registration_id"] == "reg-B"
                assert sample.metadata["slot_version"] == 1
                assert sample.index is not None and sample.response_length > 0
                assert all(math.isfinite(lp) for lp in sample.rollout_log_probs)
        assert [g[0].index for g in b_groups] == sorted(g[0].index for g in b_groups)

        time.sleep(1.0)
        assert worker.get_groups(snapshot(), 1024, {})[1] == {}, "blocked adapters must stay blocked"

        runs["A"] = make_run("A", 0, str(tmp_path / "a.jsonl"), rollout_batch_size=2, version=2)
        wait_until(lambda: worker.queue_sizes().get("A", 0) == 2, message="A to refill after version advance")

        runs["B"] = make_run("B", 1, str(tmp_path / "b.jsonl"), rollout_batch_size=3, version=2)
        wait_until(
            lambda: worker.queue_sizes().get("A", 0) == 2 and worker.queue_sizes().get("B", 0) == 3,
            message="both adapters ready for the coalesced round",
        )
        output = run(generate_rollout_multi_lora_async(args, 1, data_source, real_generate))
        names = [g[0].adapter.name for g in output.samples]
        assert names == ["A"] * 2 + ["B"] * 3, "complete adapter batches must coalesce into one multi-slot batch"
        head = output.samples[0][0]
        assert head.metadata["step_adapter_names"] == ["A", "B"]
        assert head.metadata["step_slots"] == [0, 1]
    finally:
        AsyncMultiLoRAWorker.stop_global()

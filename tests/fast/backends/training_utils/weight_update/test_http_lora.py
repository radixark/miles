import base64
import io
import json
import os
import pickle
from argparse import Namespace
from unittest.mock import patch

import httpx
import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols import http_lora
from miles.backends.training_utils.weight_update.protocols.http_lora import UpdateWeightHttpLora
from miles.utils.lora.utils import LORA_ADAPTER_NAME

_MODULE = "miles.backends.training_utils.weight_update.protocols.http_lora"
_HF_KEY = "model.layers.0.self_attn.q_proj.lora_A.weight"


def _http_error(status_code: int, text: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://engine")
    return httpx.HTTPStatusError(text, request=request, response=httpx.Response(status_code, text=text))


class _FakeEngine:
    """Models SGLang's LoRA registry: a held name rejects a plain load; upsert replaces it only if supported."""

    def __init__(self, url: str = "http://e:1", supports_upsert: bool = True, held: dict | None = None) -> None:
        self.server_url = url
        self.supports_upsert = supports_upsert
        self.served: dict[str, object] = dict(held or {})
        self.calls: list[str] = []
        self.load_error: httpx.HTTPStatusError | None = None
        self.pause_error: Exception | None = None

    async def load_lora_adapter(self, lora_name, lora_path, pinned=False, upsert=False, timeout=None):
        return self._load("load", lora_name, lora_path, upsert)

    async def load_lora_adapter_from_tensors(
        self, lora_name, config_dict, serialized_named_tensors=None, upsert=False, timeout=None
    ):
        return self._load("load_from_tensors", lora_name, (config_dict, serialized_named_tensors), upsert)

    def _load(self, route: str, lora_name: str, payload: object, upsert: bool) -> dict:
        self.calls.append(f"{route}+upsert" if upsert else route)
        if self.load_error is not None:
            raise self.load_error
        if lora_name in self.served and not (upsert and self.supports_upsert):
            raise _http_error(400, f"LoRA with name {lora_name} already exists. Loaded LoRAs: [{lora_name}]")
        self.served[lora_name] = payload
        return {"success": True}

    async def unload_lora_adapter(self, lora_name, timeout=None):
        self.calls.append("unload")
        if lora_name not in self.served:
            raise _http_error(400, f"LoRA with name {lora_name} does not exist. Loaded LoRAs: []")
        del self.served[lora_name]

    async def pause_generation(self, mode="retract"):
        self.calls.append("pause")
        if self.pause_error is not None:
            raise self.pause_error

    async def flush_cache(self):
        self.calls.append("flush")

    async def continue_generation(self):
        self.calls.append("resume")

    async def update_weight_version(self, weight_version, abort_all_requests=False):
        self.calls.append(f"version={weight_version}")


def _make_args(tmp_path, **overrides) -> Namespace:
    args = Namespace(
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_adapter_targets=["q_proj", "v_proj"],
        update_weight_disk_dir=str(tmp_path),
        custom_update_weight_post_write_path=None,
        http_lora_ship="path",
        fully_async=False,
        pause_generation_mode="retract",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _connect(args: Namespace, engines: list[_FakeEngine], gpu_counts: list[int] | None = None) -> UpdateWeightHttpLora:
    protocol = UpdateWeightHttpLora(args)
    with patch(f"{_MODULE}.dist.get_rank", return_value=0):
        protocol.connect(
            engines,
            gpu_counts or [1] * len(engines),
            None,
            parallel_state=None,
            placement=WeightUpdatePlacement(gather_pp=True),
            selector="all",
        )
    return protocol


def _sync(protocol: UpdateWeightHttpLora, weight_version: int, adapter_value: float = 1.0) -> None:
    protocol.send_bucket([(f"{LORA_ADAPTER_NAME}:{_HF_KEY}", torch.full((16, 8), adapter_value))])
    protocol.finalize(weight_version)


def _served_path(engine: _FakeEngine) -> str:
    return os.path.basename(engine.served[LORA_ADAPTER_NAME])


def test_upsert_engine_replaces_in_place_under_a_pause(tmp_path):
    engine = _FakeEngine(supports_upsert=True)
    protocol = _connect(_make_args(tmp_path), [engine])

    _sync(protocol, 1)
    engine.calls.clear()
    _sync(protocol, 2)

    assert _served_path(engine).endswith("_v2")
    assert engine.calls == ["pause", "flush", "load+upsert", "version=2", "resume"]


def test_stock_engine_unloads_and_reloads_unpaused_after_the_first_sync(tmp_path):
    engine = _FakeEngine(supports_upsert=False)
    protocol = _connect(_make_args(tmp_path), [engine])

    _sync(protocol, 1)
    assert engine.calls == ["pause", "flush", "load", "load+upsert", "version=1", "resume"]
    engine.calls.clear()
    _sync(protocol, 2)

    assert _served_path(engine).endswith("_v2")
    assert engine.calls == ["unload", "load", "version=2"]


def test_stock_engine_holding_an_adapter_from_an_earlier_run_serves_the_new_one(tmp_path):
    engine = _FakeEngine(supports_upsert=False, held={LORA_ADAPTER_NAME: "/old/run/v30"})
    protocol = _connect(_make_args(tmp_path), [engine])

    _sync(protocol, 1)

    assert _served_path(engine).endswith("_v1")


def test_fully_async_refuses_a_stock_engine_and_resumes_the_fleet(tmp_path):
    engine = _FakeEngine(supports_upsert=False)
    protocol = _connect(_make_args(tmp_path, fully_async=True), [engine])

    with pytest.raises(RuntimeError, match="fully-async"):
        _sync(protocol, 1)
    assert engine.calls[-1] == "resume"


def test_fully_async_refuses_to_unload_a_stale_adapter(tmp_path):
    engine = _FakeEngine(supports_upsert=False, held={LORA_ADAPTER_NAME: "/old/run/v30"})
    protocol = _connect(_make_args(tmp_path, fully_async=True), [engine])

    with pytest.raises(RuntimeError, match="earlier run"):
        _sync(protocol, 1)
    assert "unload" not in engine.calls


def test_a_mixed_fleet_falls_back_to_unload_on_every_engine(tmp_path):
    upsert_engine = _FakeEngine("http://a:1", supports_upsert=True)
    stock_engine = _FakeEngine("http://b:1", supports_upsert=False)
    protocol = _connect(_make_args(tmp_path), [upsert_engine, stock_engine])

    _sync(protocol, 1)
    _sync(protocol, 2)

    for engine in (upsert_engine, stock_engine):
        assert _served_path(engine).endswith("_v2")
        assert "pause" not in engine.calls[engine.calls.index("version=1") :]


def test_an_unrelated_load_failure_raises_without_unloading_and_resumes(tmp_path):
    engine = _FakeEngine()
    engine.load_error = _http_error(400, "rank 64 exceeds --max-lora-rank")
    protocol = _connect(_make_args(tmp_path), [engine])

    with pytest.raises(httpx.HTTPStatusError, match="max-lora-rank"):
        _sync(protocol, 1)
    assert "unload" not in engine.calls
    assert engine.calls[-1] == "resume"


def test_a_failed_pause_still_resumes_the_fleet(tmp_path):
    engine = _FakeEngine()
    engine.pause_error = TimeoutError("Timeout while flushing cache")
    protocol = _connect(_make_args(tmp_path), [engine])

    with pytest.raises(TimeoutError):
        _sync(protocol, 1)
    assert engine.calls == ["pause", "resume"]


def test_path_mode_stages_a_peft_dir_with_the_run_lora_config(tmp_path):
    engine = _FakeEngine()
    protocol = _connect(_make_args(tmp_path), [engine])

    _sync(protocol, 1, adapter_value=3.0)

    adapter_dir = engine.served[LORA_ADAPTER_NAME]
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        config = json.load(f)
    assert config["r"] == 16 and config["lora_alpha"] == 32 and config["target_modules"] == ["q_proj", "v_proj"]
    weights = safetensors.torch.load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    assert torch.equal(weights[_HF_KEY], torch.full((16, 8), 3.0))


def test_path_mode_keeps_the_newest_versions_of_this_run_only(tmp_path):
    os.makedirs(tmp_path / f"{LORA_ADAPTER_NAME}_19990101-000000_v30")
    protocol = _connect(_make_args(tmp_path), [_FakeEngine()])

    for version in range(1, 6):
        _sync(protocol, version)

    own_versions = sorted(name for name in os.listdir(tmp_path) if protocol._run_tag in name)
    assert [name.rsplit("_v", 1)[1] for name in own_versions] == ["3", "4", "5"]
    assert os.path.isdir(tmp_path / f"{LORA_ADAPTER_NAME}_19990101-000000_v30")


def test_tensors_mode_sends_one_copy_per_engine_gpu_and_writes_nothing(tmp_path):
    engines = [_FakeEngine("http://a:1"), _FakeEngine("http://b:1")]
    protocol = _connect(
        _make_args(tmp_path, http_lora_ship="tensors", update_weight_disk_dir=None), engines, gpu_counts=[2, 4]
    )

    _sync(protocol, 1, adapter_value=5.0)

    for engine, gpu_count in zip(engines, [2, 4], strict=True):
        config, serialized = engine.served[LORA_ADAPTER_NAME]
        assert config["r"] == 16
        assert len(serialized) == gpu_count
        tensors = pickle.loads(base64.b64decode(serialized[0]))
        assert torch.equal(tensors[_HF_KEY], torch.full((16, 8), 5.0))
    assert not any(tmp_path.iterdir())


def _foreign_load_from_bytes(b):
    return torch.load(io.BytesIO(b), weights_only=True)


def test_tensors_payload_names_the_stock_torch_loader_under_a_patched_torch(monkeypatch):
    monkeypatch.setattr(torch.storage, "_load_from_bytes", _foreign_load_from_bytes)
    assert b"_foreign_load_from_bytes" in pickle.dumps({"w": torch.ones(2, 2)})

    payload = http_lora._pickle_with_torch_globals({"w": torch.ones(2, 2)})

    assert b"_foreign_load_from_bytes" not in payload
    assert torch.storage._load_from_bytes is _foreign_load_from_bytes
    monkeypatch.undo()
    assert torch.equal(pickle.loads(payload)["w"], torch.ones(2, 2))

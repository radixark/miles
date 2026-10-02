"""The load path must REPLACE, never silently skip, and never deadlock a paused engine.

The updater hands a transport a stable adapter name every sync
(``LORA_ADAPTER_NAME``, or ``slot_lora_name(slot)`` under multi-LoRA), so every
sync after the first names an adapter the engine already holds. SGLang's registry
raises ``LoRA with name X already exists``. Treated as success, v1 loads and every
later version is refused by the engine yet logged as a successful sync.

Replacing via unload is only safe UNPAUSED: unload waits for the adapter's usage
counter, which a request releases only when it FINISHES, and a paused engine under
retract/in_place never finishes them. So the transport probes upsert once, pauses
only when upsert (or the first sync) makes that safe, and refuses the unpaused
fallback under --fully-async.
"""

from __future__ import annotations

import os
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols import http_lora as mod
from miles.backends.training_utils.weight_update.protocols.http_lora import UpdateWeightHttpLora

CONFLICT = "LoRA with name miles_lora already exists. Loaded LoRAs: dict_keys(['miles_lora'])"
BODY1 = {"lora_name": "miles_lora", "lora_path": "/d/v1", "pinned": False}
BODY2 = {"lora_name": "miles_lora", "lora_path": "/d/v2", "pinned": False}
ABSENT = "LoRA with name miles_lora does not exist. Loaded LoRAs: dict_keys([])"


class _Resp:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _FakeClient:
    """Replays a scripted response per (route, has-upsert) and records every call."""

    def __init__(self, script):
        self._script = script
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):  # noqa: A002
        route = url.rsplit("/", 1)[-1]
        self.calls.append((route, json))
        key = (route, "upsert" in json)
        seq = self._script.get(key, self._script.get(route))
        return seq.pop(0) if isinstance(seq, list) else seq


def _protocol(tmp_path, fully_async=False, supported=None, ship="path", tp=1) -> UpdateWeightHttpLora:
    p = UpdateWeightHttpLora.__new__(UpdateWeightHttpLora)  # skip __init__/engine wiring
    p.args = Namespace(
        lora_rank=32, lora_alpha=64.0, lora_dropout=0.0, fully_async=fully_async, pause_generation_mode="retract"
    )
    p._stage_root = str(tmp_path)
    p._ship = ship
    p._load_route = "/load_lora_adapter" if ship == "path" else "/load_lora_adapter_from_tensors"
    p._engine_tp = tp
    p._keep_versions = 3
    p._load_timeout_s = 5.0
    p._upsert_supported = supported
    p.rollout_engines = [Namespace(server_url="http://e:1")]
    return p


def _patch_httpx(monkeypatch, client):
    import httpx

    monkeypatch.setattr(httpx, "Client", lambda **kw: client)


def _routes(c):
    return [r for r, _ in c.calls]


# ------------------------------------------------------------------ first sync


def test_first_sync_on_a_fork_installs_then_confirms_upsert(tmp_path, monkeypatch):
    c = _FakeClient({("load_lora_adapter", False): _Resp(200), ("load_lora_adapter", True): _Resp(200)})
    _patch_httpx(monkeypatch, c)
    assert _protocol(tmp_path)._load_one("http://e:1", "miles_lora", BODY1) is True
    assert _routes(c) == ["load_lora_adapter", "load_lora_adapter"]


def test_first_sync_on_a_stock_engine_installs_then_learns_it_cannot_upsert(tmp_path, monkeypatch):
    # pydantic ignores the unknown `upsert` key, so the probe surfaces as a name conflict
    c = _FakeClient({("load_lora_adapter", False): _Resp(200), ("load_lora_adapter", True): _Resp(400, CONFLICT)})
    _patch_httpx(monkeypatch, c)
    assert _protocol(tmp_path)._load_one("http://e:1", "miles_lora", BODY1) is False
    assert "unload_lora_adapter" not in _routes(c), "must not unload while paused for the first sync"


def test_a_stale_adapter_on_a_stock_engine_fails_loudly_instead_of_deadlocking(tmp_path, monkeypatch):
    """Leftover from a previous run + no upsert: we are paused and cannot unload."""
    c = _FakeClient(
        {("load_lora_adapter", False): _Resp(400, CONFLICT), ("load_lora_adapter", True): _Resp(400, CONFLICT)}
    )
    _patch_httpx(monkeypatch, c)
    with pytest.raises(RuntimeError, match="earlier run"):
        _protocol(tmp_path)._load_one("http://e:1", "miles_lora", BODY1)
    assert "unload_lora_adapter" not in _routes(c)


# ------------------------------------------------------------------ later syncs


def test_later_syncs_with_upsert_are_one_call_and_never_unload(tmp_path, monkeypatch):
    c = _FakeClient({("load_lora_adapter", True): _Resp(200)})
    _patch_httpx(monkeypatch, c)
    _protocol(tmp_path, supported=True)._load_one("http://e:1", "miles_lora", BODY2)
    assert _routes(c) == ["load_lora_adapter"] and c.calls[0][1]["upsert"] is True


def test_later_syncs_without_upsert_unload_then_load_the_new_version(tmp_path, monkeypatch):
    """THE regression: sync 2+ must change the served adapter."""
    c = _FakeClient({"unload_lora_adapter": _Resp(200), ("load_lora_adapter", False): _Resp(200)})
    _patch_httpx(monkeypatch, c)
    _protocol(tmp_path, supported=False)._load_one("http://e:1", "miles_lora", BODY2)
    assert _routes(c) == ["unload_lora_adapter", "load_lora_adapter"]
    assert c.calls[-1][1]["lora_path"] == "/d/v2"
    assert "upsert" not in c.calls[-1][1]


def test_unloading_an_absent_adapter_is_benign(tmp_path, monkeypatch):
    c = _FakeClient({"unload_lora_adapter": _Resp(400, ABSENT), ("load_lora_adapter", False): _Resp(200)})
    _patch_httpx(monkeypatch, c)
    _protocol(tmp_path, supported=False)._load_one("http://e:1", "miles_lora", BODY2)
    assert _routes(c)[-1] == "load_lora_adapter"


def test_a_genuine_unload_failure_is_an_error(tmp_path, monkeypatch):
    c = _FakeClient({"unload_lora_adapter": _Resp(500, "internal error")})
    _patch_httpx(monkeypatch, c)
    with pytest.raises(RuntimeError, match="unload failed"):
        _protocol(tmp_path, supported=False)._load_one("http://e:1", "miles_lora", BODY2)


def test_an_unrelated_load_failure_never_triggers_an_unload(tmp_path, monkeypatch):
    c = _FakeClient({("load_lora_adapter", False): _Resp(500, "rank 64 exceeds --max-lora-rank")})
    _patch_httpx(monkeypatch, c)
    with pytest.raises(RuntimeError, match="load failed"):
        _protocol(tmp_path)._load_one("http://e:1", "miles_lora", BODY1)
    assert "unload_lora_adapter" not in _routes(c)


# ----------------------------------------------------------- fleet-level policy


def test_fully_async_refuses_a_fleet_that_cannot_upsert(tmp_path, monkeypatch):
    c = _FakeClient({("load_lora_adapter", False): _Resp(200), ("load_lora_adapter", True): _Resp(400, CONFLICT)})
    _patch_httpx(monkeypatch, c)
    with pytest.raises(RuntimeError, match="fully-async"):
        _protocol(tmp_path, fully_async=True)._load_everywhere("miles_lora", BODY1)


def test_synchronous_rl_accepts_a_fleet_that_cannot_upsert(tmp_path, monkeypatch):
    c = _FakeClient({("load_lora_adapter", False): _Resp(200), ("load_lora_adapter", True): _Resp(400, CONFLICT)})
    _patch_httpx(monkeypatch, c)
    p = _protocol(tmp_path, fully_async=False)
    p._load_everywhere("miles_lora", BODY1)
    assert p._upsert_supported is False


def test_a_mixed_fleet_is_treated_as_unable_to_upsert(tmp_path, monkeypatch):
    p = _protocol(tmp_path)
    p.rollout_engines = [Namespace(server_url="http://a:1"), Namespace(server_url="http://b:1")]
    monkeypatch.setattr(p, "_load_one", lambda base, *_: base.startswith("http://a"))
    p._load_everywhere("miles_lora", BODY1)
    assert p._upsert_supported is False, "one engine that cannot upsert makes pausing unsafe fleet-wide"


# ------------------------------------------------------------- pause discipline


def _finalize_with(p, monkeypatch, buffered=True):
    events = []
    monkeypatch.setattr(mod, "pause_engines", lambda args, engines: events.append("pause"))
    monkeypatch.setattr(mod, "resume_engines", lambda engines: events.append("resume"))
    monkeypatch.setattr(mod, "set_weight_version", lambda engines, v: events.append(f"version={v}"))
    monkeypatch.setattr(p, "_write_adapter", lambda name, v, t: str(p._stage_root))
    monkeypatch.setattr(p, "_load_everywhere", lambda name, d: events.append("load"))
    monkeypatch.setattr(p, "_prune_old_versions", lambda *a: None)
    p.is_sender = True
    p._post_write_hook = None
    p.update_weight_metrics = {}
    p._buf = {"miles_lora:m.layers.0.q.lora_A.weight": torch.zeros(2, 2)} if buffered else {}
    p.finalize(7)
    return events


def test_the_first_sync_is_paused_since_nothing_can_be_unloaded_yet(tmp_path, monkeypatch):
    assert _finalize_with(_protocol(tmp_path, supported=None), monkeypatch) == ["pause", "load", "version=7", "resume"]


def test_upsert_syncs_are_paused_so_the_cache_flush_and_version_label_are_right(tmp_path, monkeypatch):
    assert _finalize_with(_protocol(tmp_path, supported=True), monkeypatch) == ["pause", "load", "version=7", "resume"]


def test_the_unload_fallback_is_never_paused(tmp_path, monkeypatch):
    """Pausing here deadlocks: unload waits on requests a paused engine cannot finish."""
    events = _finalize_with(_protocol(tmp_path, supported=False), monkeypatch)
    assert "pause" not in events and "resume" not in events
    assert events == ["load", "version=7"]


def test_the_engines_are_resumed_even_when_the_load_raises(tmp_path, monkeypatch):
    p = _protocol(tmp_path, supported=True)
    events = []
    monkeypatch.setattr(mod, "pause_engines", lambda args, engines: events.append("pause"))
    monkeypatch.setattr(mod, "resume_engines", lambda engines: events.append("resume"))
    monkeypatch.setattr(mod, "set_weight_version", lambda engines, v: None)
    monkeypatch.setattr(p, "_write_adapter", lambda name, v, t: str(p._stage_root))

    def boom(name, d):
        raise RuntimeError("engine down")

    monkeypatch.setattr(p, "_load_everywhere", boom)
    p.is_sender, p._post_write_hook, p.update_weight_metrics = True, None, {}
    p._buf = {"miles_lora:k.lora_A.weight": torch.zeros(2, 2)}
    with pytest.raises(RuntimeError):
        p.finalize(1)
    assert events == ["pause", "resume"], "a failed swap must not leave the fleet paused"


def test_the_engines_are_resumed_even_when_the_pause_itself_fails(tmp_path, monkeypatch):
    """pause_engines pauses and then flushes; stock SGLang refuses the flush while
    retracted requests are queued, so under --fully-async the helper can raise with
    the fleet already paused. Seen live: a run died at sync 3 and the engine stayed
    paused until someone called /continue_generation by hand."""
    p = _protocol(tmp_path, supported=True)
    events = []

    def paused_then_flush_failed(args, engines):
        events.append("pause")
        raise TimeoutError("Timeout while flushing cache: Flush cache failed.")

    monkeypatch.setattr(mod, "pause_engines", paused_then_flush_failed)
    monkeypatch.setattr(mod, "resume_engines", lambda engines: events.append("resume"))
    monkeypatch.setattr(mod, "set_weight_version", lambda engines, v: None)
    monkeypatch.setattr(p, "_write_adapter", lambda name, v, t: str(p._stage_root))
    monkeypatch.setattr(p, "_load_everywhere", lambda name, d: events.append("load"))
    p.is_sender, p._post_write_hook, p.update_weight_metrics = True, None, {}
    p._buf = {"miles_lora:k.lora_A.weight": torch.zeros(2, 2)}
    with pytest.raises(TimeoutError):
        p.finalize(1)
    assert events == ["pause", "resume"], "a failed pause must not leave the fleet paused"


# ------------------------------------------------------------------ the rest


def test_rank_comes_from_the_tensors_not_the_flags(tmp_path):
    p = _protocol(tmp_path)
    p.args.lora_rank = 999
    cfg = p._adapter_config(
        {
            "m.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(16, 2048),
            "m.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(2048, 16),
        }
    )
    assert cfg["r"] == 16 and isinstance(cfg["lora_alpha"], float)


def test_old_versions_are_pruned_but_the_current_one_survives(tmp_path):
    p = _protocol(tmp_path)
    for v in range(1, 7):
        os.makedirs(tmp_path / f"miles_lora_v{v}")
    p._prune_old_versions("miles_lora", current=6)
    assert sorted(int(d.split("_v")[-1]) for d in os.listdir(tmp_path)) == [4, 5, 6]


# ------------------------------------------------------------- tensors mode


def _finalize_real(p, monkeypatch, client):
    """finalize() with the real writer and loader, a fake engine, and the pause frame stubbed."""
    _patch_httpx(monkeypatch, client)
    monkeypatch.setattr(mod, "pause_engines", lambda args, engines: None)
    monkeypatch.setattr(mod, "resume_engines", lambda engines: None)
    monkeypatch.setattr(mod, "set_weight_version", lambda engines, v: None)
    p.is_sender, p.update_weight_metrics = True, {}
    p._buf = {
        "miles_lora:model.layers.0.self_attn.q_proj.lora_A.weight": torch.arange(64, dtype=torch.bfloat16).reshape(
            32, 2
        )
    }
    p.finalize(1)


def test_tensors_mode_posts_the_weights_themselves_and_needs_no_files_or_hook(tmp_path, monkeypatch):
    """--http-lora-ship tensors: the adapter travels in the request body, one copy per TP
    rank, as a plain pickle that any host can open. No staging dir, no post-write hook."""
    import base64
    import pickle

    p = _protocol(tmp_path, ship="tensors", tp=2)
    p._stage_root = None

    def must_not_ship(*a):
        raise AssertionError("the post-write hook is a path-mode concern")

    p._post_write_hook = must_not_ship
    c = _FakeClient(
        {("load_lora_adapter_from_tensors", False): _Resp(200), ("load_lora_adapter_from_tensors", True): _Resp(200)}
    )
    _finalize_real(p, monkeypatch, c)

    assert _routes(c) == ["load_lora_adapter_from_tensors", "load_lora_adapter_from_tensors"]  # install, then probe
    body = c.calls[0][1]
    assert body["lora_name"] == "miles_lora" and body["config_dict"]["r"] == 32
    assert len(body["serialized_named_tensors"]) == 2, "one identical copy per TP rank"
    got = pickle.loads(base64.b64decode(body["serialized_named_tensors"][0]))
    assert torch.equal(
        got["model.layers.0.self_attn.q_proj.lora_A.weight"], torch.arange(64, dtype=torch.bfloat16).reshape(32, 2)
    ), "the payload must carry the bytes, not a shared-memory handle"
    assert not any(tmp_path.iterdir()), "nothing written to disk without a staging dir"
    assert p._upsert_supported is True


def test_tensors_mode_keeps_versioned_files_when_a_dir_is_given(tmp_path, monkeypatch):
    p = _protocol(tmp_path, ship="tensors")
    c = _FakeClient(
        {("load_lora_adapter_from_tensors", False): _Resp(200), ("load_lora_adapter_from_tensors", True): _Resp(200)}
    )
    _finalize_real(p, monkeypatch, c)
    assert (tmp_path / "miles_lora_v1" / "adapter_model.safetensors").exists()
    assert (tmp_path / "miles_lora_v1" / "adapter_config.json").exists()


def test_tensors_mode_probes_and_falls_back_exactly_like_path_mode(tmp_path, monkeypatch):
    """Same registry, same 'already exists', same decision: a stock engine gets unload+reload in sync RL."""
    c = _FakeClient(
        {
            ("load_lora_adapter_from_tensors", False): _Resp(200),
            ("load_lora_adapter_from_tensors", True): _Resp(400, CONFLICT),
        }
    )
    _patch_httpx(monkeypatch, c)
    p = _protocol(tmp_path, ship="tensors")
    body = {"lora_name": "miles_lora", "config_dict": {}, "serialized_named_tensors": ["x"], "pinned": False}
    assert p._load_one("http://e:1", "miles_lora", body) is False
    c2 = _FakeClient({"unload_lora_adapter": _Resp(200), ("load_lora_adapter_from_tensors", False): _Resp(200)})
    _patch_httpx(monkeypatch, c2)
    p._upsert_supported = False
    p._load_one("http://e:1", "miles_lora", body)
    assert _routes(c2) == ["unload_lora_adapter", "load_lora_adapter_from_tensors"]


def _foreign_load_from_bytes(b):  # stands in for megatron.core.safe_globals.safe_load_from_bytes
    return torch.load(__import__("io").BytesIO(b), weights_only=True)


def test_tensors_payload_names_torch_not_megatron_even_when_megatron_patched_pickling(monkeypatch):
    """Inside a Megatron trainer torch.storage._load_from_bytes is a Megatron function; a
    naive pickle then references megatron.core.safe_globals, which SGLang's allowlisting
    unpickler rejects (and the scheduler dies). Seen live at the first sync of a run."""
    import pickle
    import torch.storage as ts

    monkeypatch.setattr(ts, "_load_from_bytes", _foreign_load_from_bytes)
    naive = pickle.dumps({"w": torch.ones(2, 2)})
    assert b"_foreign_load_from_bytes" in naive, "precondition: a naive pickle names the foreign loader"
    ours = mod._pickle_with_torch_globals({"w": torch.ones(2, 2)})
    assert b"_foreign_load_from_bytes" not in ours and b"torch.storage" in ours
    assert ts._load_from_bytes is _foreign_load_from_bytes, "the patch is restored afterwards"
    monkeypatch.undo()
    assert torch.equal(pickle.loads(ours)["w"], torch.ones(2, 2)), "stock torch unpickles it"

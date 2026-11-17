"""The sglang-torchtpu APIs miles' TPU path calls; runs where sglang-torchtpu is installed (.github/workflows/tpu-contract.yml)."""

import argparse
import asyncio
import inspect
import pathlib
import re
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang_tpu")

from sglang.srt.entrypoints.http_server import app  # noqa: E402
from sglang.srt.environ import EnvField, Envs  # noqa: E402
from sglang.srt.managers import io_struct  # noqa: E402

from sglang_tpu import shared_slice  # noqa: E402
from sglang_tpu.envs import TpuEnvs  # noqa: E402

from miles.backends.sglang_utils import arguments as sglang_arguments  # noqa: E402
from miles.backends.sglang_utils import sglang_api_client  # noqa: E402
from miles.platforms import tpu_shared_slice  # noqa: E402
from miles.utils.args.runtime import AllConfig  # noqa: E402

_MILES_ROOT = pathlib.Path(__file__).resolve().parents[4] / "miles"
# fields miles sends that sglang-torchtpu does not take; each is LoRA-only, and TPU weight sync rejects LoRA
_LORA_ONLY_FIELDS = {
    "begin_weight_update": {"sync_base"},
    "end_weight_update": {"expected_lora_checksums"},
}
# server args only sglang-miles registers; miles reads them through getattr or only writes them
_SGLANG_MILES_ONLY_ARGS = {"enable_prefill_weight_versions", "enable_prefill_only_deterministic_inference"}


class _RecordingHttpClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    async def post(self, url: str, *, json: dict, **_):
        self.requests.append((url.rsplit("/", 1)[-1], json))
        return _OkResponse()


class _OkResponse:
    status_code = 200
    content = b'{"success": true}'

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"success": True}


# the engine calls of the TPU path: weight-update session, ICI weight sync (protocols/ici.py), version,
# pause/resume, checksum, and the abort every rollout ends with
_CALLS = {
    "begin_weight_update": ("BeginWeightUpdateReqInput", lambda c: c.begin_weight_update(selector="all")),
    "end_weight_update": ("EndWeightUpdateReqInput", lambda c: c.end_weight_update()),
    "init_weights_update_group": (
        "InitWeightsUpdateGroupReqInput",
        lambda c: c.init_weights_update_group(
            master_address="10.0.0.1",
            master_port=30108,
            rank_offset=0,
            world_size=8,
            group_name="miles_ici",
            backend=shared_slice.BACKEND_NAME,
        ),
    ),
    "update_weights_from_distributed": (
        "UpdateWeightsFromDistributedReqInput",
        lambda c: c.update_weights_from_distributed(
            names=["w"], dtypes=[torch.bfloat16], shapes=[torch.Size([2, 3])], group_name="miles_ici"
        ),
    ),
    "update_weight_version": ("UpdateWeightVersionReqInput", lambda c: c.update_weight_version("1")),
    "pause_generation": ("PauseGenerationReqInput", lambda c: c.pause_generation(mode="retract")),
    "weights_checker": ("CheckWeightsReqInput", lambda c: c.check_weights(action="checksum")),
    "abort_request": ("AbortReq", lambda c: c.abort_all_requests()),
}


@pytest.mark.parametrize("endpoint", sorted(_CALLS))
def test_the_engine_parses_every_request_miles_sends(monkeypatch, endpoint):
    """The engine drops fields it does not know without an error, so a renamed field only shows up here."""
    http = _RecordingHttpClient()
    monkeypatch.setattr(sglang_api_client.GeneralHttpClientProvider, "client", classmethod(lambda cls: http))
    request_class, call = _CALLS[endpoint]
    asyncio.run(call(sglang_api_client.SGLangApiClient(server_url="http://engine")))

    (sent_endpoint, payload), *_ = http.requests
    assert sent_endpoint == endpoint
    payload = {k: v for k, v in payload.items() if k not in _LORA_ONLY_FIELDS.get(endpoint, set())}
    getattr(io_struct, request_class)(**payload)


def _served_routes(routes) -> set[tuple[str, str]]:
    served = set()
    for route in routes:
        served |= {(route.path, method) for method in getattr(route, "methods", None) or ()}
        nested = getattr(route, "routes", None) or getattr(getattr(route, "router", None), "routes", None)
        if nested:
            served |= _served_routes(nested)
    return served


def test_the_engine_serves_every_endpoint_miles_calls():
    expected = {(f"/{endpoint}", "POST") for endpoint in _CALLS} | {
        ("/generate", "POST"),
        ("/continue_generation", "POST"),
        ("/flush_cache", "GET"),
        ("/server_info", "GET"),
        ("/health_generate", "GET"),
        ("/model_info", "GET"),
    }
    served = _served_routes(app.routes)
    assert expected <= served, f"sglang-torchtpu no longer serves {sorted(expected - served)}"


def test_the_engine_takes_the_generate_fields_the_tpu_path_sends():
    """Rollout sends these on /generate (generate_endpoint_utils.py, session/core.py for routing replay)."""
    fields = set(inspect.signature(io_struct.GenerateReqInput).parameters)
    sent = {"input_ids", "sampling_params", "return_logprob", "return_routed_experts", "routed_experts_start_len"}
    assert sent <= fields, f"GenerateReqInput lost {sorted(sent - fields)}"


def test_the_engine_registers_the_env_vars_the_launcher_sets():
    """An unregistered SGLANG_* variable is ignored, so a rename silently turns the setting off."""
    registered = {
        field.name for envs in (Envs, TpuEnvs) for field in vars(envs).values() if isinstance(field, EnvField)
    }
    # scripts/tpu/torchtitan_launch.py serve_sglang
    launcher_sets = {"SGLANG_TPU_USE_SEQ_ALONG_LANE_KV"}
    assert launcher_sets <= registered, f"sglang-torchtpu no longer reads {sorted(launcher_sets - registered)}"
    assert shared_slice.ENV_VAR == "SGLANG_TPU_SHARED_SLICE"


class _LoopbackBackend:
    """One slice's send/recv: a send parks the buffer under its tag until the paired recv takes it."""

    def __init__(self) -> None:
        self.in_flight: dict[int, torch.Tensor] = {}

    def send(self, buffers, _dst, tag):
        self.in_flight[tag] = buffers[0].clone()
        return SimpleNamespace(wait=lambda: None)

    def recv(self, buffers, _src, tag):
        buffers[0].copy_(self.in_flight.pop(tag))
        return SimpleNamespace(wait=lambda: None)


class _DictStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def set(self, key, value):
        self.values[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        return self.values[key]

    def delete_key(self, key):
        del self.values[key]


def test_the_engine_reads_back_each_bucket_miles_sends_over_the_shared_slice(monkeypatch):
    """miles packs a bucket and posts its names under a tag; the engine must slice the same bits out by its layout."""
    backend, store = _LoopbackBackend(), _DictStore()
    monkeypatch.setattr(tpu_shared_slice, "_membership", tpu_shared_slice._Membership(backend, store, slice_rank=1))
    receiver = shared_slice.SharedSliceReceiver(backend, store, src_slice_rank=1, slice_rank=5)
    buckets = [
        {"embed": torch.randn(3, 5, dtype=torch.bfloat16), "norm": torch.randn(7, dtype=torch.float32)},
        {"experts": torch.arange(4097, dtype=torch.int32).view(17, 241)},
    ]

    for bucket in buckets:
        tensors = list(bucket.values())
        tpu_shared_slice.send(tpu_shared_slice.pack(tensors), 5, names=list(bucket))
        received = receiver.receive(
            names=["rank 0's names"] * len(bucket),
            dtypes=[str(t.dtype).replace("torch.", "") for t in tensors],
            shapes=[list(t.shape) for t in tensors],
            device="cpu",
        )
        assert [name for name, _ in received] == list(bucket)
        for (_, got), want in zip(received, tensors, strict=True):
            assert torch.equal(got, want)
    assert not store.values, "the engine leaves bucket names behind in the store"


def test_every_server_arg_miles_reads_is_registered():
    parser = argparse.ArgumentParser()
    sglang_arguments.add_sglang_arguments(parser)
    registered = {dest.removeprefix("sglang_") for dest in vars(parser.parse_args([]))}
    # direct reads only: a getattr read with a default already tolerates a missing arg
    read = {
        name
        for path in _MILES_ROOT.rglob("*.py")
        for name in re.findall(r"args\.sglang_([a-z0-9_]+)", re.sub(r"#.*", "", path.read_text()))
    }
    # miles' own config fields named sglang_* (router, scaling, ...) are not engine server args
    miles_owned = {name.removeprefix("sglang_") for name in AllConfig.model_fields if name.startswith("sglang_")}
    missing = read - registered - miles_owned - _SGLANG_MILES_ONLY_ARGS
    assert not missing, (
        f"miles reads --sglang-* args sglang-torchtpu does not register: {sorted(missing)}; "
        "if only sglang-miles has them and the TPU path never reads them, add them to _SGLANG_MILES_ONLY_ARGS"
    )

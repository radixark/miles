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
from sglang.srt.environ import EnvField, Envs, exportable_env_vars  # noqa: E402
from sglang.srt.managers import io_struct  # noqa: E402
from sglang.srt.state_capturer.base import BaseDeviceCache, BaseTopkCapturer  # noqa: E402

from miles.backends.sglang_utils import arguments as sglang_arguments  # noqa: E402
from miles.backends.sglang_utils import sglang_api_client  # noqa: E402
from miles.ray.rollout.external_engine_provider import _engine_activation_fp32  # noqa: E402

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


# the engine calls of the TPU path: weight-update session, shm/disk weight sync, version, pause/resume, checksum,
# and the abort every rollout ends with
_CALLS = {
    "begin_weight_update": ("BeginWeightUpdateReqInput", lambda c: c.begin_weight_update(selector="all")),
    "end_weight_update": ("EndWeightUpdateReqInput", lambda c: c.end_weight_update()),
    "update_weights_from_tensor": (
        "UpdateWeightsFromTensorReqInput",
        lambda c: c.update_weights_from_tensor(serialized_named_tensors=["x"], load_format="flattened_bucket"),
    ),
    "update_weights_from_disk": (
        "UpdateWeightFromDiskReqInput",
        lambda c: c.update_weights_from_disk(model_path="/w/weight_v000001", weight_version="1"),
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


def _served_paths(routes) -> set[str]:
    paths = set()
    for route in routes:
        if hasattr(route, "path"):
            paths.add(route.path)
        nested = getattr(route, "routes", None) or getattr(getattr(route, "router", None), "routes", None)
        if nested:
            paths |= _served_paths(nested)
    return paths


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
    registered = {field.name for field in vars(Envs).values() if isinstance(field, EnvField)}
    launcher_sets = {"SGLANG_TPU_PD_COLOCATED", "SGLANG_EXPOSE_OWN_ENV_VARS", "SGLANG_TPU_ACTIVATION_FP32"}
    assert launcher_sets <= registered, f"sglang-torchtpu no longer reads {sorted(launcher_sets - registered)}"


def test_the_engine_exports_its_fp32_silu_setting(monkeypatch):
    """--activation-in-fp32 refuses an external engine whose SGLANG_TPU_ACTIVATION_FP32 it cannot read."""
    monkeypatch.setenv("SGLANG_TPU_ACTIVATION_FP32", "1")
    server_info = {"internal_states": [{"env_vars": exportable_env_vars()}]}
    assert _engine_activation_fp32(server_info) is True


def test_a_skipped_forward_reports_its_rows_as_unknown():
    """Routing replay on TPU relies on prefill rows coming back as -1 (sgl-project/sglang-torchtpu#1580)."""
    capturer = object.__new__(BaseTopkCapturer)
    capturer.num_layers, capturer.topk_size, capturer.skip_capture = 2, 2, False
    capturer.device_cache = BaseDeviceCache(max_batch_size=4, num_layers=2, topk_size=2, device="cpu", name="t")
    capturer.host_cache = SimpleNamespace(buffer=torch.zeros(8, 2, 2, dtype=torch.int32))

    def forward(out_cache_loc: list[int], *, skip: bool) -> None:
        capturer.skip_capture = skip
        for layer in range(2):
            capturer.capture(layer, torch.tensor([[layer, 5]] * len(out_cache_loc), dtype=torch.int32))
        loc = torch.tensor(out_cache_loc)
        capturer.on_forward_end(
            SimpleNamespace(out_cache_loc=loc, out_cache_loc_virtual=None), can_run_graph=False, cuda_graph_batch=None
        )

    forward([0, 1, 2], skip=True)  # a compiled prefill
    forward([3], skip=False)  # a decode step
    assert not capturer.skip_capture, "the flag must clear after the skipped forward"

    pool = SimpleNamespace(req_to_token=torch.arange(8).unsqueeze(0))
    rows = capturer.get_topk(req_pool_idx=0, seqlen=5, req_to_token_pool=pool)
    assert rows.dtype == torch.int32 and rows.shape == (4, 2, 2), "miles decodes [num_tokens - 1, layers, topk] int32"
    assert (rows[:3] == -1).all()
    assert rows[3].tolist() == [[0, 5], [1, 5]]


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
    missing = read - registered - _SGLANG_MILES_ONLY_ARGS
    assert not missing, f"miles reads --sglang-* args sglang-torchtpu does not register: {sorted(missing)}"

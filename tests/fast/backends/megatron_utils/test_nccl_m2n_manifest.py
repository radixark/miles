"""Critical M2N sender regressions; native NCCL calls are mocked."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols import nccl_m2n
from miles.backends.training_utils.weight_update.protocols import nccl_m2n_manifest as manifest_utils
from miles.backends.training_utils.weight_update.protocols.nccl_m2n import UpdateWeightFromNcclM2N
from miles.backends.training_utils.weight_update.protocols.nccl_m2n_manifest import (
    _build_manifest,
    _split_manifest_by_pp,
)

_FP8_CONFIG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "scale_fmt": "ue8m0",
}


def _payloads():
    """Uneven PP2, TP2/CP2, EP4/ETP1; no model-specific routing assumptions."""
    payloads = []
    for rank in range(8):
        pp, local = divmod(rank, 4)
        specs = []
        for layer in ((0, 1), (2,))[pp]:
            for family in ("dense", "routed_expert"):
                for projection in (1, 2):
                    expert = ".experts" if family == "routed_expert" else ""
                    name = f"module.module.decoder.layers.{layer}.mlp{expert}.linear_fc{projection}.weight"
                    specs.append(
                        {
                            "name": name + (str(local) if expert else ""),
                            "family": family,
                            "layer": layer,
                            "projection": f"fc{projection}",
                            "expert_id": local if expert else None,
                            "dtype": "bfloat16",
                            "local_shape": [512 if projection == 1 else 256, 256],
                            "partition_dim": -1 if expert else projection - 1,
                            "partition_stride": 2 if not expert and projection == 1 else 1,
                        }
                    )
        payloads.append(
            {
                "topology": dict(
                    world_rank=rank,
                    pp_rank=pp,
                    pp_size=2,
                    tp_rank=local % 2,
                    tp_size=2,
                    cp_rank=local // 2,
                    cp_size=2,
                    dense_dp_rank=0,
                    dense_dp_size=1,
                    ep_rank=local,
                    ep_size=4,
                    etp_rank=0,
                    etp_size=1,
                    expert_dp_rank=0,
                    expert_dp_size=1,
                    independent_dp_rank=0,
                    independent_dp_size=1,
                ),
                "specs": specs,
                "update_units": [[spec["name"]] for spec in specs],
            }
        )
    return payloads


@pytest.mark.parametrize("fp8,ep", [(False, 2), (False, 1), (True, 2), (True, 1)])
@pytest.mark.parametrize("scale_format", ["canonical", "ue8m0_unpacked"])
def test_manifest_ownership_shards_replicas_and_fp8_pairs(fp8, ep, scale_format):
    payloads = _payloads()
    config = {**_FP8_CONFIG, "scale_format": scale_format} if fp8 else None
    kwargs = dict(destination_ep_size=ep, quantization_config=config)
    manifest = _build_manifest(payloads, [2, 2], **kwargs)
    assert manifest == _build_manifest(list(reversed(payloads)), [2, 2], **kwargs)
    original = deepcopy(manifest)
    stages = _split_manifest_by_pp(manifest)
    assert manifest == original
    assert list(stages) == [0, 1]
    assert sum(len(s["entries"]) for s in stages.values()) == len(manifest["entries"])
    assert len(manifest["entries"]) == 18
    for pp, stage in stages.items():
        assert stage["source_world_ranks"] == list(range(4 * pp, 4 * pp + 4))
        assert stage["trainer_world_to_comm_rank"] == {str(4 * pp + rank): rank for rank in range(4)}
        assert stage["communicator_world_size"] == 8
        assert stage["manifest_hash"] == manifest_utils._manifest_digest(stage)
        for entry in stage["entries"]:
            assert entry["pp_rank"] == pp
            layer = int(entry["name"].split(".")[2])
            assert pp == (0 if layer < 2 else 1)
            expert = entry["family"] == "routed_expert"
            assert entry["source"]["mesh"] == [list(range(4 if expert else 2))]
            assert set(entry["source"]["names_by_rank"]) == {str(r) for r in range(4 if expert else 2)}
            assert entry["destination"]["mesh"] == [[4, 5], [6, 7]]
            down = "down_proj" in entry["name"]
            dim = (0 if ep == 2 else (2 if down else 1)) if expert else int(down)
            assert entry["destination"]["placements"][1] == {"type": "shard", "dim": dim}
            shape = entry["global_shape"].copy()
            shape[dim] //= 2
            assert entry["destination"]["local_shape"] == shape
        if fp8:
            manifest_utils._validate_fp8_pairs(stage["entries"])
            assert stage["quantization"]["scale_format"] == scale_format
            pairs = {}
            for entry in stage["entries"]:
                pairs.setdefault(entry["pair_id"], {})[entry["tensor_role"]] = entry
            for pair in pairs.values():
                weight, scale = pair["weight"], pair["scale"]
                assert weight["dtype"] == "float8_e4m3fn" and scale["dtype"] == "float32"
                assert scale["global_shape"] == [weight["global_shape"][0], 2, 2]
                assert scale["source"]["names_by_rank"] == weight["source"]["names_by_rank"]
            with pytest.raises(ValueError):
                manifest_utils._validate_fp8_pairs(stage["entries"][:-1])


@pytest.mark.parametrize("fp8,family", [(False, "dense"), (False, "routed_expert"), (True, "routed_expert")])
def test_atomic_fallback_does_not_drop_peer_weights(fp8, family):
    payloads = _payloads()
    blocked = set()
    for payload in payloads[:4]:
        for spec in payload["specs"]:
            if spec["layer"] == 0 and spec["family"] == family and spec["projection"] == "fc1":
                blocked.add(spec["name"])
        payload["update_units"] = [
            unit + ["unsupported.peer"] if unit[0] in blocked else unit for unit in payload["update_units"]
        ]
    manifest = _build_manifest(payloads, [2, 2], quantization_config=_FP8_CONFIG if fp8 else None)
    prefix = "model.layers.0.mlp." + ("experts." if family == "routed_expert" else "")
    names = {e["name"] for e in manifest["entries"]}
    assert prefix + "gate_proj.weight" not in names
    assert prefix + "up_proj.weight" not in names
    assert (prefix + "down_proj.weight" in names) is (not fp8)
    routed = {name for unit in manifest["routed_update_units"] for name in unit}
    assert routed.isdisjoint(blocked)
    # Coalesced expert entries must not leave any peer fc1 marked as routed.
    assert any(name.startswith("model.layers.1.") for name in names)


@pytest.mark.parametrize("scale_format", ["canonical", "ue8m0_unpacked"])
def test_fp8_source_pair_is_quantized_once_per_batch_and_never_reused(scale_format):
    manifest = _build_manifest(
        _payloads(),
        [2],
        quantization_config={**_FP8_CONFIG, "scale_format": scale_format},
    )
    pair_id = "model.layers.0.mlp.experts.gate_proj.weight"
    pair_entries = [entry for entry in manifest["entries"] if entry["pair_id"] == pair_id]
    assert [entry["tensor_role"] for entry in pair_entries] == [
        "weight",
        "scale",
    ]

    source_name = pair_entries[0]["source"]["names_by_rank"]["0"][0]
    fused_fc1 = torch.empty(512, 256, dtype=torch.bfloat16)
    fused_fc1[:256].fill_(1)
    fused_fc1[256:].fill_(9)

    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater._m2n_manifest = {"entries": pair_entries, "quantization": manifest["quantization"]}
    updater._m2n_pg = object()
    updater._m2n_comm_ptr = 123
    updater._m2n_comm_rank = 0
    updater._source_device = "cpu"
    updater._m2n_local_tensors = {source_name: fused_fc1}
    updater._m2n_fp8_pair_cache = {}

    quantized_inputs = []

    def quantize(logical, **kwargs):
        assert kwargs == {"scale_format": scale_format}
        quantized_inputs.append(logical.clone())
        marker = float(logical[0, 0, 0])
        return (
            torch.full(
                logical.shape,
                marker,
                dtype=torch.float8_e4m3fn,
            ),
            torch.full((1, 2, 2), marker, dtype=torch.float32),
        )

    m2n = SimpleNamespace(
        Replicate=Mock(return_value=("replicate",)),
        Shard=Mock(side_effect=lambda dim: ("shard", dim)),
        reshard=Mock(),
    )
    stream = Mock()
    with (
        patch.object(nccl_m2n, "_nccl_m2n", return_value=m2n),
        patch.object(
            nccl_m2n,
            "_quantize_block_fp8",
            side_effect=quantize,
        ),
        patch.object(
            nccl_m2n.torch.cuda,
            "current_stream",
            return_value=stream,
        ),
    ):
        updater._run_m2n_batch()
        assert updater._m2n_fp8_pair_cache == {}

        fused_fc1[:256].fill_(2)
        updater._run_m2n_batch()
        assert updater._m2n_fp8_pair_cache == {}

        m2n.reshard.side_effect = RuntimeError("injected transfer failure")
        with pytest.raises(RuntimeError, match="injected transfer failure"):
            updater._run_m2n_batch()
        assert updater._m2n_fp8_pair_cache == {}

    assert len(quantized_inputs) == 3
    assert torch.all(quantized_inputs[0] == 1)
    assert torch.all(quantized_inputs[1] == 2)
    assert torch.all(quantized_inputs[2] == 2)
    assert m2n.reshard.call_count == 5
    first_weight = m2n.reshard.call_args_list[0].args[0]
    first_scale = m2n.reshard.call_args_list[1].args[0]
    second_weight = m2n.reshard.call_args_list[2].args[0]
    second_scale = m2n.reshard.call_args_list[3].args[0]
    assert first_weight.dtype == torch.float8_e4m3fn
    assert first_scale.dtype == torch.float32
    assert torch.all(first_weight == 1)
    assert torch.all(first_scale == 1)
    assert torch.all(second_weight == 2)
    assert torch.all(second_scale == 2)
    assert stream.synchronize.call_count == 4


@pytest.mark.parametrize("scale_format", ["canonical", "ue8m0_unpacked"])
@pytest.mark.parametrize("fp32_scales", ["0", "1"])
def test_fp8_source_uses_selected_quantizer(scale_format, fp32_scales):
    weight = torch.ones((2, 128, 256), dtype=torch.bfloat16)
    quantized = torch.ones((256, 256), dtype=torch.float8_e4m3fn)
    scales = torch.full((2, 2), 0.5)
    with (
        patch.dict(nccl_m2n.os.environ, NVTE_FP8_BLOCK_SCALING_FP32_SCALES=fp32_scales),
        patch.object(nccl_m2n, "per_block_cast_to_fp8", return_value=(quantized, scales)) as ue8m0,
        patch.object(nccl_m2n, "blockwise_cast_to_fp8_triton", return_value=(quantized, scales)) as canonical,
    ):
        actual_weight, actual_scale = nccl_m2n._quantize_block_fp8(weight, scale_format=scale_format)
        assert actual_weight.shape == weight.shape
        assert actual_scale.shape == (2, 1, 2)
        assert actual_scale.dtype == torch.float32 and actual_scale.is_contiguous()
        assert torch.all(actual_scale == 0.5)
        use_ue8m0 = scale_format == "ue8m0_unpacked" or fp32_scales == "0"
        assert ue8m0.call_count == int(use_ue8m0)
        assert canonical.call_count == int(not use_ue8m0)
        with patch.object(nccl_m2n, "per_block_cast_to_fp8", None):
            if scale_format == "ue8m0_unpacked":
                with pytest.raises(RuntimeError, match="power-of-two"):
                    nccl_m2n._quantize_block_fp8(weight, scale_format=scale_format)
            else:
                nccl_m2n._quantize_block_fp8(weight, scale_format=scale_format)
                assert canonical.call_count == int(not use_ue8m0) + 1


@pytest.mark.parametrize(
    "backend,a2a,requires_ue8m0,scale_format",
    [
        ("triton", "none", True, "canonical"),
        ("deep_gemm", "none", True, "ue8m0_unpacked"),
        ("auto", "deepep", True, "ue8m0_unpacked"),
        ("auto", "none", True, "canonical"),
        ("deep_gemm", "none", False, "canonical"),
    ],
)
def test_fp8_manifest_selection_matches_broadcast(backend, a2a, requires_ue8m0, scale_format):
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.args = SimpleNamespace(
        sglang_moe_runner_backend=backend, sglang_moe_a2a_backend=a2a, megatron_to_hf_mode="direct"
    )
    with (
        patch.object(nccl_m2n, "named_params_and_buffers", return_value=[]),
        patch.object(nccl_m2n.quantizer_fp8, "should_deepgemm_weight_requant_ue8m0", return_value=requires_ue8m0),
    ):
        updater.configure_model(SimpleNamespace(model=None, quantization_config=_FP8_CONFIG))
    assert updater._m2n_quantization["scale_format"] == scale_format


@pytest.mark.parametrize("peer_issue", [None, "missing UE8M0 quantizer", "different NCCL M2N FP8 formats"])
def test_ue8m0_wire_format_requires_all_trainers_to_support_it(peer_issue):
    payloads = _payloads()
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.args = SimpleNamespace(sglang_ep_size=2)
    updater._m2n_quantization = manifest_utils._fp8_manifest_quantization(
        {**_FP8_CONFIG, "scale_format": "ue8m0_unpacked"}
    )
    updater._trainer_payload = Mock(return_value=payloads[0])

    def gather(records, local, **kwargs):
        assert local["error"] is None
        records[:] = [
            {"payload": payload, "quantization": updater._m2n_quantization, "error": None} for payload in payloads
        ]
        if peer_issue == "missing UE8M0 quantizer":
            records[1] = {"payload": None, "error": "trainer rank 1: missing UE8M0 quantizer"}
        elif peer_issue:
            records[1]["quantization"] = {**updater._m2n_quantization, "scale_format": "canonical"}

    with (
        patch.object(nccl_m2n, "per_block_cast_to_fp8", Mock()),
        patch.object(nccl_m2n, "get_gloo_group"),
        patch.object(nccl_m2n.dist, "get_world_size", return_value=8),
        patch.object(nccl_m2n.dist, "get_rank", return_value=0),
        patch.object(nccl_m2n.dist, "all_gather_object", side_effect=gather),
        patch.object(nccl_m2n.dist, "broadcast_object_list"),
    ):
        if peer_issue:
            with pytest.raises(RuntimeError, match=peer_issue):
                updater._negotiate_manifest([2, 2])
        else:
            manifest = updater._negotiate_manifest([2, 2])
            assert manifest["quantization"]["scale_format"] == "ue8m0_unpacked"


@pytest.mark.parametrize("local_pp", [0, 1, 2])
def test_concurrent_wave_dispatches_one_rpc_and_only_local_selected_stage(local_pp):
    stages = _split_manifest_by_pp(_build_manifest(_payloads(), [2, 2]))
    wave = {stage: stages[stage] for stage in (0, 1)}
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater._m2n_group_names = {stage: f"pp-{stage}" for stage in stages}
    updater._m2n_group_name = f"pp-{local_pp}"
    updater._run_m2n_batch = Mock()
    updater._selector = "all"
    updater.rollout_engines = [Mock(), Mock()]
    with (
        patch.object(nccl_m2n.dist, "get_rank", return_value=4 * local_pp),
        patch.object(nccl_m2n, "_collect_errors", side_effect=lambda error: [error] if error else []),
        patch.object(nccl_m2n.async_utils, "submit"),
        patch.object(nccl_m2n.async_utils, "wait_futures", return_value=[{"success": True}] * 2),
    ):
        updater._update_m2n_stages(wave)
    if local_pp in wave:
        updater._run_m2n_batch.assert_called_once_with(wave[local_pp])
    else:
        updater._run_m2n_batch.assert_not_called()
    for engine in updater.rollout_engines:
        if local_pp != 0:
            engine.update_weights_from_distributed.assert_not_called()
            continue
        engine.update_weights_from_distributed.assert_called_once()
        payload = engine.update_weights_from_distributed.call_args.kwargs
        assert payload["group_name"] == "pp-0"
        assert payload["m2n_group_names"] == ["pp-0", "pp-1"]
        assert payload["names"] == [entry["name"] for stage in wave.values() for entry in stage["entries"]]


@pytest.mark.parametrize("concurrency,fail", [(1, False), (2, False), (3, False), (2, True)])
def test_pp_waves_cover_uneven_tail_and_stop_on_failure(concurrency, fail):
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.args = SimpleNamespace(m2n_pp_concurrency=concurrency)
    updater._m2n_manifest = {"entries": []}
    updater._m2n_group_names = {stage: f"pp-{stage}" for stage in range(5)}
    updater._m2n_stage_manifests = {stage: {} for stage in range(5)}
    updater._engine_lock = MagicMock()
    waves = []

    def dispatch(stages):
        waves.append(list(stages))
        if fail:
            raise RuntimeError("wave failure")

    updater._update_m2n_stage = lambda stage, manifest: dispatch({stage: manifest})
    updater._update_m2n_stages = dispatch
    with (
        patch.object(nccl_m2n.dist, "get_rank", return_value=0),
        patch.object(nccl_m2n, "_collect_errors", side_effect=lambda error: [error] if error else []),
    ):
        if fail:
            with pytest.raises(RuntimeError, match="wave failure"):
                updater._update_bulk_weights()
            assert waves == [[0, 1]]
        else:
            assert updater._update_bulk_weights() is True
            assert waves == [list(range(start, min(start + concurrency, 5))) for start in range(0, 5, concurrency)]
    updater._engine_lock.__enter__.assert_called_once()
    updater._engine_lock.__exit__.assert_called_once_with(None, None, None)


def test_sender_orders_dense_expert_source_handoffs_inside_one_pp_group():
    manifest = _split_manifest_by_pp(_build_manifest(_payloads(), [2, 2]))[0]
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater._m2n_manifest = manifest
    updater._m2n_pg = object()
    updater._m2n_comm_ptr = 123
    updater._m2n_comm_rank = 0
    updater._m2n_fp8_pair_cache = {}
    updater._source_tensor = Mock(return_value=object())
    m2n = Mock()
    events = []
    m2n.reshard.side_effect = lambda *args, **kwargs: events.append(kwargs["src_mesh"])
    with (
        patch.object(nccl_m2n, "_nccl_m2n", return_value=m2n),
        patch.object(nccl_m2n.torch.cuda, "current_stream"),
        patch.object(nccl_m2n.dist, "barrier", side_effect=lambda group: events.append("handoff")) as barrier,
    ):
        updater._run_m2n_batch(manifest)
    expected = []
    previous = None
    for entry in manifest["entries"]:
        mesh = entry["source"]["mesh"]
        if previous is not None and previous != mesh:
            expected.append("handoff")
        expected.append(mesh)
        previous = mesh
    assert events == expected
    assert barrier.call_count > 0
    assert all(invocation.kwargs["group"] is updater._m2n_pg for invocation in barrier.call_args_list)


@pytest.mark.parametrize("mode", ["raw", "bridge"])
def test_nccl_m2n_refreshes_source_mapping_and_omits_atomic_residual_units(mode):
    from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase

    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.args = SimpleNamespace(megatron_to_hf_mode=mode)
    iterator = SimpleNamespace(model=[], quantization_config=None)
    native = "module.module.decoder.layers.2.mlp.linear_fc1.weight"
    local = "vp_stages.0.decoder.layers.0.mlp.linear_fc1.weight"
    with patch.object(
        nccl_m2n,
        "named_params_and_buffers",
        side_effect=lambda *a, **k: [(native if k.get("convert_to_global_name", True) else local, torch.zeros(1))],
    ):
        updater.configure_model(iterator)
    updater._m2n_local_tensors = {native: torch.zeros(1)}
    updater._update_bulk_weights = Mock()
    with patch.object(nccl_m2n, "_collect_errors", return_value=[]):
        for value in (1, 2):
            current = torch.full((1,), value)
            updater.before_base_weights({native if mode == "raw" else local: current})
            assert updater._m2n_local_tensors[native] is current
    assert updater._update_bulk_weights.call_count == 2

    iterator.excluded_hf_names = {"gate.weight", "up.weight"}
    routed = [("gate.weight", torch.zeros(1)), ("up.weight", torch.zeros(1))]
    residual = [("attention.weight", torch.zeros(1))]
    assert list(HfWeightIteratorBase._residual_units(iterator, [routed, residual])) == [residual]
    with pytest.raises(RuntimeError, match="atomic"):
        list(HfWeightIteratorBase._residual_units(iterator, [routed + residual]))


def test_nccl_m2n_async_api_payload_and_strict_teardown():
    import asyncio
    from unittest.mock import AsyncMock

    import httpx

    from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient

    client = SGLangApiClient("http://unused")
    request = AsyncMock(return_value={"success": True})

    async def exercise():
        with patch.object(SGLangApiClient, "_make_request", request):
            await client.init_weights_update_group("host", 123, 2, 4, "pp0", "nccl", m2n_manifest={"entries": []})
            assert request.call_args.args[1]["m2n_manifest"] == {"entries": []}
            await client.update_weights_from_distributed(
                ["w"],
                [torch.bfloat16],
                [[1]],
                "pp0",
                load_format="nccl_m2n",
                m2n_group_names=["pp0", "pp1"],
            )
            payload = request.call_args.args[1]
            assert payload["load_format"] == "nccl_m2n"
            assert payload["m2n_group_names"] == ["pp0", "pp1"]
            assert payload["dtypes"] == ["bfloat16"]
            request.side_effect = httpx.ConnectError("offline")
            with pytest.raises(httpx.ConnectError):
                await client.destroy_weights_update_group("pp0", strict=True)
            await client.destroy_weights_update_group("pp0")

    asyncio.run(exercise())


@pytest.mark.parametrize("rank", [0, 1])
def test_nccl_m2n_session_failure_is_collective_and_marks_connection_stale(rank):
    from miles.backends.training_utils.conn_status import ConnStatusManager
    from miles.backends.training_utils.weight_update import updater as updater_module

    protocol = object.__new__(UpdateWeightFromNcclM2N)
    operation = Mock(side_effect=RuntimeError("prepare failed"))
    updater = object.__new__(updater_module.WeightUpdater)
    updater.conn_status = ConnStatusManager()
    updater.conn_status.mark_reconnected({})
    updater._update_weights = lambda: protocol.run_engine_session(operation)
    with (
        patch.object(nccl_m2n.dist, "get_rank", return_value=rank),
        patch.object(nccl_m2n, "_collect_errors", return_value=["prepare failed"]) as collect,
    ):
        with pytest.raises(RuntimeError, match="prepare failed"):
            updater.update_weights()
    assert updater.conn_status.needs_reconnect({})
    assert operation.call_count == int(rank == 0)
    assert (collect.call_args.args[0] is None) is (rank != 0)

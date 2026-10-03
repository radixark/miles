"""Admission and ownership checks for the sole-codec GPU delta benchmarks."""

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest
import torch

_MODULE = Path(__file__).parents[2] / "manual" / "bench_gpu_delta.py"
_spec = importlib.util.spec_from_file_location("bench_gpu_delta", _MODULE)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_fixture_codec_refuses_missing_or_different_publication():
    good = {"protocol_version": 4, "codec": "snappy-zstd", "frame_bytes": 1 << 20}
    fixture = {"codec": "snappy-zstd", "rounds": [{"publications": {"snappy-zstd": good}}]}
    assert bench._validate_fixture_codec(fixture) == "snappy-zstd"
    for bad in ({}, good | {"protocol_version": 3}, good | {"codec": "zstd"}, good | {"frame_bytes": 1 << 21}):
        fixture["rounds"][0]["publications"]["snappy-zstd"] = bad
        with pytest.raises(ValueError, match="requires"):
            bench._validate_fixture_codec(fixture)


def test_pending_inventory_checks_exact_names_and_byte_counts(monkeypatch):
    spec = importlib.util.spec_from_file_location("bench_gpu_delta_producer", _MODULE.with_name("bench_gpu_delta_producer.py"))
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    monkeypatch.setattr(producer.dist, "get_rank", lambda: 0)
    protocol = Namespace(pending_baseline={"w": torch.zeros(8, dtype=torch.uint8)})
    plan = {"w": {"shape": [2, 2], "dtype": "BF16"}}
    assert producer._verify_pending_inventory(protocol, plan)["canonical_bytes"] == 8
    with pytest.raises(ValueError, match="ownership"):
        producer._verify_pending_inventory(protocol, {})
    with pytest.raises(ValueError, match="size"):
        producer._verify_pending_inventory(protocol, {"w": {"shape": [2, 3], "dtype": "BF16"}})


def test_producer_metadata_satisfies_current_negotiation_without_claiming_a_receiver(tmp_path):
    import asyncio

    from miles.backends.training_utils.weight_update.gpu_delta_session import negotiate_cohort

    spec = importlib.util.spec_from_file_location("gpu_delta_producer_metadata", _MODULE.with_name("bench_gpu_delta_producer.py"))
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    plan = [{"name": "w", "dtype": "U8", "shape": [2, 2], "encoding": "xor_bytes",
             "views": [{"id": "canonical", "slices": [[0, 2], [0, 2]]}]}]
    protocol = producer._make_protocol(Namespace(custom_update_weight_post_write_path=None), plan, tmp_path)
    description = asyncio.run(protocol._describe())
    cohort = negotiate_cohort(description)
    assert cohort.plan == plan
    assert cohort.engine_ids == ("producer-benchmark-no-receiver",)
    assert cohort.host_tensor_names == {"producer-benchmark-no-host-cache": ["w"]}

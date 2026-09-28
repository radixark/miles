import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "examples/multi_lora/run_train_sampler_sync_test.py"
_SPEC = importlib.util.spec_from_file_location("train_sampler_sync_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROBE
_SPEC.loader.exec_module(_PROBE)

_LAUNCHER_SCRIPT = Path(__file__).parents[3] / "examples/multi_lora/serve_qwen_dense_tinker.py"
_LAUNCHER_SPEC = importlib.util.spec_from_file_location("serve_qwen_dense_tinker", _LAUNCHER_SCRIPT)
assert _LAUNCHER_SPEC is not None and _LAUNCHER_SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(_LAUNCHER_SPEC)
sys.modules[_LAUNCHER_SPEC.name] = _LAUNCHER
_LAUNCHER_SPEC.loader.exec_module(_LAUNCHER)


def _stats(value: float):
    return _PROBE.DeltaStats(tokens=8, mean=value, p50=value, p90=value, p99=value, maximum=value)


def _thresholds():
    return argparse.Namespace(
        match_p90=0.1,
        minimum_step_change=0.02,
        maximum_stale_sampler_change=1e-4,
        minimum_mismatch_growth=0.02,
        maximum_resync_ratio=0.5,
    )


def test_delta_stats_masks_prompt_tokens():
    stats = _PROBE._delta_stats([100.0, 1.0, 2.0], [0.0, 1.1, 2.2], [0.0, 1.0, 1.0])

    assert stats.tokens == 2
    assert stats.mean == pytest.approx(0.15)
    assert stats.maximum == pytest.approx(0.2)


def test_launcher_uses_pipeline_safe_adapter_layout():
    assert _LAUNCHER.ScriptArgs().target_modules == "attn,mlp"


def test_five_phase_contract_passes():
    phases = {
        "pre_step_match": _stats(0.01),
        "trainer_change": _stats(0.2),
        "stale_sampler_change": _stats(0.0),
        "stale_mismatch": _stats(0.2),
        "resynced_match": _stats(0.01),
    }

    assert _PROBE._phase_failures(_thresholds(), phases) == []


def test_five_phase_contract_reports_each_boundary():
    phases = {
        "pre_step_match": _stats(0.2),
        "trainer_change": _stats(0.0),
        "stale_sampler_change": _stats(0.01),
        "stale_mismatch": _stats(0.2),
        "resynced_match": _stats(0.2),
    }

    assert len(_PROBE._phase_failures(_thresholds(), phases)) == 6

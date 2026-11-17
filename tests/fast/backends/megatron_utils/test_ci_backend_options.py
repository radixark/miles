import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import ci_utils
from miles.backends.training_utils import parallel


def test_hash_checkpoint_roundtrip_uses_backend_paths_and_checks_real_model_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend save/load paths carry actual model hashes and detect weight changes."""
    group = SimpleNamespace(rank=0)
    monkeypatch.setattr(parallel, "_parallel_state", SimpleNamespace(tp=group, pp=group, intra_dp_cp=group, cp=group))
    model = [torch.nn.Linear(2, 1)]
    args = SimpleNamespace(
        ci_test=True,
        ci_save_model_hash=True,
        ci_check_model_hash=True,
        backend=SimpleNamespace(save=tmp_path / "save", load=tmp_path / "save"),
    )
    hashes = ci_utils.compute_model_hashes_by_layer(model)

    ci_utils.save_model_hashes(args, model=model, iteration=7, hashes=hashes)
    saved = tmp_path / "save" / "iter_0000007" / "model_hash_tp0_pp0_dp0_cp0.json"
    assert json.loads(saved.read_text()) == hashes
    ci_utils.check_model_hashes(args, model=model, iteration=7)
    with torch.no_grad():
        model[0].weight.add_(1)
    with pytest.raises(AssertionError, match="Model hash mismatch"):
        ci_utils.check_model_hashes(args, model=model, iteration=7)
    args.backend.load = tmp_path / "missing"
    with pytest.raises(AssertionError, match="Hash file missing"):
        ci_utils.check_model_hashes(args, model=model, iteration=7)


@pytest.mark.parametrize("enabled,peak_gb", [(False, 22), (True, 19), (True, 22)])
def test_low_memory_resume_guard_uses_backend_switch(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, peak_gb: int
) -> None:
    """The backend switch controls the checkpoint peak-memory assertion."""
    calls = []
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: calls.append("peak") or peak_gb * 1024**3)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    args = SimpleNamespace(
        ci_test=True, backend=SimpleNamespace(low_memory_resume=enabled), hf_checkpoint="/models/Qwen3-4B"
    )

    if enabled and peak_gb >= 20:
        with pytest.raises(AssertionError, match="exceeds threshold"):
            ci_utils.check_peak_gpu_memory_after_load(args)
    else:
        ci_utils.check_peak_gpu_memory_after_load(args)

    assert calls == (["peak"] if enabled else [])

import json

import torch

from miles.backends.megatron_utils.router_replay_artifacts import (
    get_router_replay_artifact_paths,
    get_router_replay_sidecar_paths,
    load_router_replay_artifact_bundle,
)


def test_router_replay_artifact_paths_and_bundle_loading(tmp_path):
    artifact_paths = get_router_replay_artifact_paths(
        save_dir=str(tmp_path),
        step="training_7_mini3",
        tp_rank=1,
        pp_rank=2,
    )
    assert artifact_paths["step_dir"].endswith("/7")
    assert artifact_paths["main"].endswith("training_7_mini3_tp1_pp2.pt")

    tmp_path.joinpath("7").mkdir(exist_ok=True)
    torch.save(
        {
            "step": "training_7_mini3",
            "predictive_bias": {0: torch.randn(2, 4)},
        },
        artifact_paths["main"],
    )
    with open(artifact_paths["predictive_metrics"], "w", encoding="utf-8") as f:
        json.dump({"aggregates": {"predictive_loss": 1.25}, "debug": {"selected_total_tokens": 128}}, f)
    torch.save(
        {"layers": {"0": {"old_logits": torch.randn(2, 4)}}},
        artifact_paths["predictive_metric_tensors"],
    )

    bundle = load_router_replay_artifact_bundle(artifact_paths["main"])
    assert bundle["main"]["step"] == "training_7_mini3"
    assert bundle["predictive_metrics"]["aggregates"]["predictive_loss"] == 1.25
    assert bundle["predictive_metrics"]["debug"]["selected_total_tokens"] == 128
    assert "layers" in bundle["predictive_metric_tensors"]

    sidecar_paths = get_router_replay_sidecar_paths(artifact_paths["main"])
    assert sidecar_paths["predictive_metrics"] == artifact_paths["predictive_metrics"]
    assert sidecar_paths["predictive_metric_tensors"] == artifact_paths["predictive_metric_tensors"]

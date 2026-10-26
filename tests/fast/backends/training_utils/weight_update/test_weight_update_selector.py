"""The weight-update selector follows whether the trainer trains MTP layers, on every backend."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["weight-update"])

import pytest

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.backends.training_utils.weight_update.session import weight_update_selector
from miles.utils.args.runtime import TrainerConfig


@pytest.mark.parametrize(
    "speculative, enable_mtp_training, selector",
    [
        (None, False, "all"),
        (None, True, "all"),
        ("EAGLE", True, "all"),
        ("EAGLE", False, "target"),
        ("EAGLE3", False, "target"),
    ],
    ids=["no_draft", "no_draft_with_mtp", "trained_draft", "frozen_draft", "external_draft"],
)
def test_a_draft_is_synced_only_when_the_trainer_trains_mtp_layers(
    speculative: str | None, enable_mtp_training: bool, selector: str
) -> None:
    """FSDP and torchtitan reject --enable-mtp-training, so their drafts are always left out."""
    args = TrainerConfig.model_construct(
        sglang=SglangConfig(models=[], base_args={"speculative_algorithm": speculative}),
        enable_mtp_training=enable_mtp_training,
    )
    assert (
        weight_update_selector(
            speculative_algorithm=args.sglang.base_args["speculative_algorithm"],
            enable_mtp_training=args.enable_mtp_training,
        )
        == selector
    )

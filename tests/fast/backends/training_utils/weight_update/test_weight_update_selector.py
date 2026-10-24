"""The weight-update selector follows whether the trainer trains MTP layers, on every backend."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["weight-update"])

from argparse import Namespace

import pytest

from miles.backends.training_utils.weight_update.session import weight_update_selector


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
def test_a_draft_is_synced_only_when_the_trainer_trains_mtp_layers(speculative, enable_mtp_training, selector):
    """FSDP and torchtitan reject --enable-mtp-training, so their drafts are always left out."""
    args = Namespace(sglang_speculative_algorithm=speculative, enable_mtp_training=enable_mtp_training)
    assert weight_update_selector(args) == selector

"""What build_trainer_config refuses before torchtitan sees the arguments."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import json
from argparse import Namespace

import pytest

from miles.backends.torchtitan_utils.config import build_trainer_config


def test_a_tied_checkpoint_is_refused_under_pipeline_parallelism(tmp_path):
    """torchtitan gives a tied model its own lm_head and cannot tie it across
    stages, so under PP the trained lm_head would never reach the engine. Refusing
    up front beats torchtitan's NotImplementedError from deep inside Trainer.__init__."""
    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    args = Namespace(optimizer="adam", titan_pipeline_parallel_degree=2)
    with pytest.raises(ValueError, match="pipeline"):
        build_trainer_config(args, hf_assets_path=str(tmp_path), lr_total_steps=1, dump_subdir="x")

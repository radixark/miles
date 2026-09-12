"""Evaluation must not replace the persistent legacy training configuration."""

import importlib
import sys
from types import ModuleType, SimpleNamespace


def test_evaluation_first_preserves_training_opd_configuration(monkeypatch):
    states = []
    fake_rollout = ModuleType("miles.rollout.sglang_rollout")

    def initialize_state(args):
        if not states:
            states.append(args)
        return states[0]

    def generate(args, rollout_id, data_source, evaluation):
        assert initialize_state(args).use_opd
        assert initialize_state(args).opd_log_prob_top_k == 16
        assert not args.use_opd
        assert args.opd_log_prob_top_k == 0
        assert args.custom_rm_path == "examples.mopd_puzzles.tasks.reward_func"
        assert evaluation
        return "evaluated"

    fake_rollout.GenerateState = initialize_state
    fake_rollout.generate_rollout = generate
    monkeypatch.setitem(sys.modules, "miles.rollout.sglang_rollout", fake_rollout)
    monkeypatch.delitem(sys.modules, "examples.mopd_puzzles.evaluate", raising=False)
    module = importlib.import_module("examples.mopd_puzzles.evaluate")
    args = SimpleNamespace(use_opd=True, opd_log_prob_top_k=16, custom_rm_path="teacher")
    try:
        assert module.generate_rollout(args, 0, None, evaluation=True) == "evaluated"
        assert states[0] is args
        assert args.use_opd and args.opd_log_prob_top_k == 16
        assert args.custom_rm_path == "teacher"
    finally:
        sys.modules.pop("examples.mopd_puzzles.evaluate", None)

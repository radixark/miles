"""Probe bounded retries, terminal failures, and the real Sample archive format."""
import asyncio
import gzip
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import frozen_collect
from miles.rollout.base_types import RolloutFnTrainInput
from miles.utils.types import Sample


def main() -> None:
    os.environ.pop("FROZEN_RESUME_ROOT", None)
    os.environ.pop("FROZEN_RESUME_ROOTS", None)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        os.environ["PILOT_ROOT"] = directory
        trial = root / "timeout"
        trial.mkdir()
        (trial / "result.json").write_text(json.dumps({"exception_info": {"exception_type": "AgentTimeoutError"}}))
        calls = {}

        async def generate(state, sample, params):
            calls[sample.index] = calls.get(sample.index, 0) + 1
            sample.tokens = [1, 2]
            sample.response_length = 1
            sample.response = "answer"
            sample.loss_mask = [1]
            sample.rollout_log_probs = [-0.5]
            sample.rollout_topk_token_ids = np.array([[2]], dtype=np.int32)
            sample.rollout_topk_log_probs = np.array([[-0.5]], dtype=np.float32)
            sample.status = Sample.Status.COMPLETED
            sample.metadata.update({"exit_status": "Submitted", "eval_report": {"ok": True}, "reward": 0.0})
            if sample.index == 0:
                sample.metadata["exit_status"] = "SequenceLengthLimitExceeded"
                sample.metadata["reward"] = 1.0
                sample.status = Sample.Status.TRUNCATED
            elif sample.index == 1:
                sample.metadata.update({"exit_status": "TimeLimitExceeded", "trial_dir": str(trial)})
            elif sample.index == 2 and calls[sample.index] == 1:
                sample.status = Sample.Status.ABORTED
            elif sample.index == 8:
                sample.metadata["exit_status"] = "AgentError"
            return [sample]

        groups = [[Sample(index=g * 8 + n, group_index=g, metadata={}) for n in range(8)] for g in range(16)]
        collector = object.__new__(frozen_collect.RolloutFn)
        collector.state = SimpleNamespace(args=SimpleNamespace(debug_rollout_only=True, rollout_batch_size=16), sampling_params={})
        collector.data_source = SimpleNamespace(get_samples=lambda count: groups)
        with patch.object(frozen_collect, "generate_and_rm", generate):
            output = asyncio.run(collector._call_train(RolloutFnTrainInput(rollout_id=0)))
        assert len(output.samples) == 15  # one infra-failed group, all-zero groups retained
        assert calls[0] == calls[1] == 1
        assert calls[2] == calls[8] == 2 and sum(calls.values()) == 130
        assert output.samples[0][0].reward == output.samples[0][1].reward == 0.0
        with gzip.open(root / "slots/00000-0.pt.gz", "rb") as stream:
            archived = torch.load(stream, weights_only=False)
        restored = Sample.from_dict(archived["samples"][0])
        assert archived["valid"] and restored.status == Sample.Status.TRUNCATED and restored.reward == 0
        assert restored.rollout_topk_token_ids.tolist() == [[2]]
        assert len((root / "attempts.jsonl").read_text().splitlines()) == 130
        parent = root / "empty-parent"
        parent.mkdir()
        (parent / "attempts.jsonl").write_text("")
        os.environ["FROZEN_RESUME_ROOTS"] = str(parent) + ":" + directory
        with patch.object(frozen_collect, "generate_and_rm", side_effect=AssertionError("must reuse saved slots")):
            resumed = asyncio.run(collector._call_train(RolloutFnTrainInput(rollout_id=0)))
        assert len(resumed.samples) == 15
        assert len((root / "attempts.jsonl").read_text().splitlines()) == 130
        del os.environ["FROZEN_RESUME_ROOTS"]
        collector._slot = lambda sample, rollout_id: asyncio.sleep(0, result=None)
        empty = asyncio.run(collector._call_train(RolloutFnTrainInput(rollout_id=1)))
        assert empty.samples == []
        assert json.loads((root / "batch-01.json").read_text())["complete_groups"] == 0
        print("FULL_COLLECTOR_TESTS_PASSED: budget outcomes retained; bounded slot retries; complete groups; archive roundtrip")


if __name__ == "__main__":
    main()

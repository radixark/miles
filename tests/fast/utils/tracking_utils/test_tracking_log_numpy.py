"""tracking.log serializes numpy scalars for the event log.

log_multi_turn_data's round_number stats come from np.max / np.min over ints
(numpy.int64), and the event logger's pydantic dump refused them: a 6-node
GLM-5.3-Flash run died at its first rollout log with "Unable to serialize
unknown type: <class 'numpy.int64'>".
"""

import json

import numpy as np
import torch

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.tracking_utils import tracking


def test_numpy_and_torch_scalars_reach_the_event_log(tmp_path, monkeypatch):
    logger = EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main"))
    monkeypatch.setattr(tracking, "is_event_logger_initialized", lambda: True)
    monkeypatch.setattr(tracking, "get_event_logger", lambda: logger)
    monkeypatch.setattr(tracking._manager, "log", lambda *a, **k: None)

    metrics = {
        "multi_turn/round_number_max": np.max(np.array([3, 6])),
        "multi_turn/round_number_mean": np.mean(np.array([3, 6])),
        "train/loss": torch.tensor(0.5),
        "rollout/step": 0,
    }
    tracking.log(None, metrics, step_key="rollout/step")

    line = (tmp_path / "events.jsonl").read_text().splitlines()[-1]
    logged = json.loads(line)["metrics"]
    assert logged == {
        "multi_turn/round_number_max": 6,
        "multi_turn/round_number_mean": 4.5,
        "train/loss": 0.5,
        "rollout/step": 0,
    }

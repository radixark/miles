# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE-bench Environment Client."""

from __future__ import annotations

from typing import Any


# Support both in-repo and standalone imports
try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import SweAction, SweObservation, SweState
except ImportError:
    from models import SweAction, SweObservation, SweState

    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class SweEnv(EnvClient[SweAction, SweObservation, SweState]):
    """HTTP client for the SWE-bench environment."""

    def _step_payload(self, action: SweAction) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "command": action.command,
            "session_id": action.session_id,
            "block": action.block,
            "wait_seconds": action.wait_seconds,
            "file_path": action.file_path,
            "content": action.content,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SweObservation]:
        obs_data = payload.get("observation", {})
        observation = SweObservation(
            instruction=obs_data.get("instruction", ""),
            output=obs_data.get("output", ""),
            success=obs_data.get("success", True),
            error=obs_data.get("error", ""),
            task_id=obs_data.get("task_id", ""),
            task_path=obs_data.get("task_path", ""),
            session_id=obs_data.get("session_id"),
            action_type=obs_data.get("action_type", ""),
            info=obs_data.get("info", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> SweState:
        return SweState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_path=payload.get("task_path", ""),
            terminal_ready=payload.get("terminal_ready", False),
            last_action_type=payload.get("last_action_type", ""),
            last_command=payload.get("last_command", ""),
            last_output=payload.get("last_output", ""),
        )

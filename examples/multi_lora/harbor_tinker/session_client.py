"""HTTP client for the gateway's recorded OpenAI-compatible sessions, decoding turns into a tinker-cookbook Trajectory.

Skeleton: methods document what they will do; bodies land in follow-up commits.

Reused, not reimplemented: everything after ``Trajectory`` is ``tinker_cookbook.rl`` — ``data_processing.trajectory_to_data``
(one merged Datum when each ``ob`` extends the previous ``ob + ac``, else one per turn), ``compute_advantages``,
``assemble_training_data`` and ``train.py``'s ``_remove_mask``. The only shaping of ours is ``turns_to_trajectory``
(collector JSON → ``Transition`` / ``Trajectory``), because the cookbook's only ``Trajectory`` producer is its own
``run_rollout`` loop.
"""

from __future__ import annotations

from typing import Any

from tinker_cookbook.rl.types import Trajectory


def turns_to_trajectory(payload: dict[str, Any]) -> Trajectory:
    """GET /oai/sessions/{sid} JSON → Trajectory: each turn is Transition(ob=ModelInput.from_ints(input_ids), ac=TokensWithLogprobs(output_ids, logprobs, finish_reason), reward=0.0); final_ob = last input_ids + output_ids; stop_reason = last finish_reason."""
    raise NotImplementedError


class SessionClient:
    """Talks to the gateway's /oai/sessions/{sid} routes with the tenant's Tinker API key (httpx)."""

    def __init__(self, gateway_url: str, tinker_api_key: str) -> None:
        """Remember the gateway base URL (http://host:10613) and the bearer key."""
        self.gateway_url = gateway_url.rstrip("/")
        self.tinker_api_key = tinker_api_key

    async def bind(
        self, session_id: str, model_path: str | None = None, sampling_session_id: str | None = None
    ) -> None:
        """POST /oai/sessions/{sid} with {model} or {sampling_session_id}; session ids are uuid4().hex minted by the caller."""
        raise NotImplementedError

    def agent_base_url(self, session_id: str) -> str:
        """The base_url handed to harbor_agent_function.run: {gateway}/oai/sessions/{sid} (miles' resolve_session_url appends /v1)."""
        raise NotImplementedError

    async def trajectory(self, session_id: str) -> Trajectory:
        """GET /oai/sessions/{sid} and decode it with turns_to_trajectory."""
        raise NotImplementedError

    async def delete(self, session_id: str) -> None:
        """DELETE /oai/sessions/{sid} once the trajectory has been fetched."""
        raise NotImplementedError

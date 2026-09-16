"""HTTP client for the gateway's recorded OpenAI-compatible sessions, decoding turns into tinker-cookbook types.

Skeleton: methods document what they will do; bodies land in follow-up commits.

This is the only client-side data shaping of ours. A recorded turn is a cookbook ``Transition`` (``ob`` = the
prompt ids the engine saw, ``ac`` = the sampled ids and their logprobs); everything downstream — merging or
splitting turns into Datums (``trajectory_to_data``), group advantages (``compute_advantages``), batching and
the ``mask`` handling (``_remove_mask``) — is ``tinker_cookbook.rl`` unchanged.
"""

from __future__ import annotations

from typing import Any

from tinker_cookbook.rl.types import Trajectory, Transition


def turn_to_transition(turn: dict[str, Any], *, episode_done: bool) -> Transition:
    """One recorded turn → Transition(ob=ModelInput.from_ints(input_ids), ac=TokensWithLogprobs(output_ids, logprobs, finish_reason), reward=0.0, episode_done)."""
    raise NotImplementedError


def turns_to_trajectory(payload: dict[str, Any]) -> Trajectory:
    """GET /oai/sessions/{sid} JSON → Trajectory(transitions, final_ob = last input_ids + output_ids, stop_reason = last finish_reason)."""
    raise NotImplementedError


class SessionClient:
    """Talks to the gateway's /oai/sessions/{sid} routes with the tenant's Tinker API key."""

    def __init__(self, gateway_url: str, tinker_api_key: str) -> None:
        """Remember the gateway base URL (http://host:10613) and the bearer key."""
        self.gateway_url = gateway_url.rstrip("/")
        self.tinker_api_key = tinker_api_key

    def new_session_id(self) -> str:
        """Mint a client-side session id (uuid4 hex); the server auto-registers it on first use."""
        raise NotImplementedError

    async def bind(self, session_id: str, model_path: str) -> None:
        """POST /oai/sessions/{sid} {model: model_path} so the harness's dummy key can sample afterwards."""
        raise NotImplementedError

    def agent_base_url(self, session_id: str) -> str:
        """The base_url handed to harbor_agent_function.run: {gateway}/oai/sessions/{sid} (the agent function appends /v1)."""
        raise NotImplementedError

    async def trajectory(self, session_id: str) -> Trajectory:
        """GET /oai/sessions/{sid} and decode it with turns_to_trajectory."""
        raise NotImplementedError

    async def delete(self, session_id: str) -> None:
        """DELETE /oai/sessions/{sid} once the trajectory has been fetched."""
        raise NotImplementedError

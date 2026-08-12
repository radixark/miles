"""Multi-turn computer-use rollout for HUD environments.

This calls into ``examples.geo3k_vlm.multi_turn.rollout`` at runtime: its token
accumulation, per-turn loss masking and multimodal bookkeeping run on every turn
of every episode here (a runtime dependency on another example, not a reference
to one — see the FIXME below). What this file adds on top:

1. The env does real network I/O (Daytona sandbox + MCP), so ``reset`` /
   ``step`` / ``close`` run in ``asyncio.to_thread`` instead of blocking the
   event loop that drives every concurrent episode.
2. Per-turn generation is capped (``hud_max_tokens_per_turn``) so one rambling
   turn cannot eat the whole episode budget.
3. The episode reward is HUD's env-side ``evaluate`` (dense progress), graded
   in the ``finally`` block *before* the sandbox is deleted and stashed in
   ``sample.metadata`` for ``reward_func`` below.

Wire with:
  --custom-generate-function-path examples.experimental.hud.rollout.generate
  --custom-rm-path examples.experimental.hud.rollout.reward_func
  --custom-config-path examples/experimental/hud/hud2048_config.yaml
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

# FIXME(hud): private helpers of another example, imported across examples.
# They are the multi-turn VLM engine -- token accumulation, per-turn loss
# masking, multimodal merge, chat-template trimming for observations -- i.e. the
# part that silently corrupts training when it is subtly wrong. Sharing one
# implementation beats copying it, but note what sharing does *not* buy here:
# nothing under tests/ exercises this module, and the one VLM e2e test
# (tests/e2e/fsdp/test_qwen3_vl_4B_fsdp.py) runs single-turn, so these functions
# have no test coverage and no API contract. Order of repayment: unit-test them
# first, then promote to miles/rollout/generate_hub/multi_turn_vlm.py next to
# single_turn / multi_turn / agentic_tool_call and import from both examples.
from examples.geo3k_vlm.multi_turn.rollout import (
    _append_to_sample,
    _build_env,
    _encode_observation_for_generation,
    _finalize_sample,
    _load_env_module,
    _prepare_start_state,
    _run_inference_step,
    _should_stop_on_finish,
    _update_budget,
    _update_multimodal_state,
)
from miles.rollout.sglang_rollout import GenerateState
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_ENV_MODULE = "examples.experimental.hud.hud_task_env"


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    env_module = _load_env_module(getattr(args, "rollout_interaction_env_path", None) or DEFAULT_ENV_MODULE)
    max_turns = args.max_turns
    per_turn_cap = int(getattr(args, "hud_max_tokens_per_turn", 256))
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)

    sampling_params = sampling_params.copy()
    current_image_data, response_tokens, budget, multimodal_train_inputs_buffer = _prepare_start_state(
        sample, state, args, sampling_params
    )
    try:
        await asyncio.to_thread(env.reset)
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        infer_timeout = float(getattr(args, "hud_inference_timeout_s", 600))
        cur_sampling_params = sampling_params
        for turn_idx in range(max_turns):
            cur_sampling_params["max_new_tokens"] = min(per_turn_cap, budget) if budget is not None else per_turn_cap

            # Hard per-request deadline: a wedged engine otherwise hangs the
            # episode forever (requests have no client-side timeout) and, at
            # batch granularity, the whole rollout step (seen 2026-08-10).
            try:
                response_text, new_response_tokens, new_response_log_probs, finish_type = await asyncio.wait_for(
                    _run_inference_step(url, sample.tokens, cur_sampling_params, current_image_data, state.tokenizer),
                    timeout=infer_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("hud inference timed out after %.0fs; aborting episode", infer_timeout)
                sample.status = Sample.Status.ABORTED
                break
            _append_to_sample(sample, response_tokens, new_response_tokens, new_response_log_probs, loss_mask_val=1)
            budget = _update_budget(budget, len(new_response_tokens))

            # Per-turn cap hits are normal turn ends, not episode truncation.
            if finish_type == "length" and budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if finish_type != "length" and _should_stop_on_finish(sample, finish_type):
                break

            observation, done, _step_info = await asyncio.to_thread(env.step, response_text)
            if done:
                sample.status = Sample.Status.COMPLETED
                break

            next_user_message = env.format_observation(observation)
            obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = (
                _encode_observation_for_generation(
                    state.tokenizer,
                    state.processor,
                    next_user_message,
                    sample.metadata,
                    args.apply_chat_template,
                    args.apply_chat_template_kwargs,
                )
            )
            bos_id = state.tokenizer.bos_token_id
            if bos_id is not None and len(obs_prompt_ids) and obs_prompt_ids[0] == bos_id:
                obs_prompt_ids = obs_prompt_ids[1:]

            obs_prompt_ids = list(obs_prompt_ids)
            _append_to_sample(sample, response_tokens, obs_prompt_ids, [0.0] * len(obs_prompt_ids), loss_mask_val=0)
            budget = _update_budget(budget, len(obs_prompt_ids))

            current_image_data = _update_multimodal_state(
                sample,
                current_image_data,
                obs_image_data,
                obs_multimodal_inputs,
                obs_multimodal_train_inputs,
                multimodal_train_inputs_buffer,
            )

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if turn_idx + 1 >= max_turns:
                sample.status = Sample.Status.COMPLETED
                break

        return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)
    finally:
        # Grade before teardown: evaluate lives inside the sandbox. Truncated and
        # aborted episodes are graded too -- partial progress is still progress,
        # and a row whose grade is all-or-nothing simply returns 0.
        try:
            verdict = await asyncio.to_thread(env.compute_final_reward)
            sample.metadata.update(verdict)
        except Exception:  # noqa: BLE001
            logger.exception("hud grading failed; reward defaults to 0.0")
            sample.metadata.setdefault("reward", 0.0)
        try:
            await asyncio.to_thread(env.close)
        except Exception:  # noqa: BLE001
            pass


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Pass through the env-computed dense reward stashed by ``generate``."""
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)

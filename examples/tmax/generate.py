"""TMax reward function, the generate function of OPD with an SGLang teacher, and pure OPD's reward post-process.

The reward is computed in the sandbox by the task's tests (tmax_agent_function.run)
and rides back in ``sample.metadata["reward"]``. ``None`` there means the sandbox
failed for a non-policy reason; returning it lets the missing-reward filter drop
the group instead of training on a false 0.
"""

import asyncio
import copy
import functools
import json
import logging
import os
from collections import Counter

from miles.rollout import on_policy_distillation
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub import agentic_tool_call
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# A teacher request that errors or times out is retried; after this many attempts it fails the rollout, since a teacher
# that stopped answering would otherwise leave every step without its OPD signal.
_TEACHER_ATTEMPTS = 3


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | None | list[float | None]:
    if isinstance(samples, list):
        return [s.metadata.get("reward") for s in samples]
    return samples.metadata.get("reward")


async def generate_with_teacher(input: GenerateFnInput) -> GenerateFnOutput:
    """``--custom-generate-function-path`` for OPD with an SGLang teacher (``--opd-type sglang``): Miles' agentic
    generate, then the teacher's log-prob of every response token of each collected sample in
    ``sample.teacher_log_probs``, which Miles trains on as the Megatron teacher's.

    Scoring happens here, not in a reward function: session server v2 returns the samples with the agent's reward set,
    so Miles never calls ``--custom-rm-path`` for them. The task reward stays the reward, so the missing-reward filter,
    the metrics, additive OPD and ``opd_pure_rewards`` work as with the Megatron teacher (Miles' own SGLang-mode reward
    function and post-process replace it with 0). The teacher request is Miles', sent to the least busy of the servers
    listed in ``TMAX_OPD_TEACHER_URLS_FILE`` (the launcher's), else to ``--rm-url``. Aborted samples, samples without
    a task reward (their group is dropped) and evaluation samples are not scored.
    """
    output = await agentic_tool_call.generate(input)
    if not input.evaluation:
        samples = output.samples if isinstance(output.samples, list) else [output.samples]
        await asyncio.gather(
            *(
                _score_with_teacher(input.args, s)
                for s in samples
                if s.status != Sample.Status.ABORTED and s.metadata.get("reward") is not None
            )
        )
    return output


# Miles registers a generate function's own flags through its add_arguments (--custom-agent-function-path,
# --max-seq-len, ...): the wrapped function's.
generate_with_teacher.add_arguments = agentic_tool_call.generate.add_arguments


async def _score_with_teacher(args, sample: Sample) -> None:
    timeout_s = float(os.environ.get("TMAX_OPD_TEACHER_TIMEOUT_S", "1800"))
    for attempt in range(1, _TEACHER_ATTEMPTS + 1):
        url = _pick_teacher(args)
        _teacher_requests_in_flight[url] += 1
        try:
            teacher_args = copy.copy(args)
            teacher_args.rm_url = url
            response = await asyncio.wait_for(
                on_policy_distillation.reward_func(teacher_args, sample), timeout=timeout_s
            )
            break
        except Exception as e:  # an HTTP error, a dropped connection, the timeout
            if attempt == _TEACHER_ATTEMPTS:
                raise RuntimeError(f"the OPD teacher failed {attempt} times on sample {sample.index}") from e
            logger.warning(f"OPD teacher request for sample {sample.index} to {url} failed (attempt {attempt}): {e!r}")
            await asyncio.sleep(10 * attempt)
        finally:
            _teacher_requests_in_flight[url] -= 1
    sample.teacher_log_probs = _response_log_probs(response, sample)


# Teacher requests in flight per server, in this rollout process: the next request goes to the least busy server.
_teacher_requests_in_flight: Counter[str] = Counter()


def _pick_teacher(args) -> str:
    """The least busy of the teacher servers the launcher started (its opd_teacher_urls_file), else --rm-url."""
    path = os.environ.get("TMAX_OPD_TEACHER_URLS_FILE")
    urls = _read_teacher_urls(path) if path else (args.rm_url,)
    return min(urls, key=lambda url: _teacher_requests_in_flight[url])


@functools.cache
def _read_teacher_urls(path: str) -> tuple[str, ...]:
    with open(path) as f:
        urls = tuple(json.load(f))
    assert urls, f"{path} lists no OPD teacher server"
    return urls


def _response_log_probs(response: dict, sample: Sample) -> list[float]:
    """The teacher's log-prob of each response token. SGLang scores every input token after the first
    (``logprob_start_len=0``), as Miles' ``post_process_rewards`` reads it; the scored ids must be the sample's."""
    n = sample.response_length
    scored = response["meta_info"]["input_token_logprobs"][1:][-n:] if n else []
    if [entry[1] for entry in scored] != (sample.tokens[-n:] if n else []):
        raise ValueError(f"the OPD teacher scored other tokens than sample {sample.index}'s response")
    return [float(entry[0]) for entry in scored]


def opd_pure_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """``--custom-reward-post-process-path`` for pure OPD, open-instruct's --opd_pure.

    Returns (raw rewards, trained rewards). Every trained reward is 0, so GRPO's advantage is 0 and the advantage
    is Miles' OPD term alone, -opd_kl_coef * (log p_rollout - log p_teacher) per token. The raw rewards are the task
    rewards, which Miles logs as rollout/raw_reward. The missing-reward filter has already dropped groups without one.
    """
    return [s.get_reward_value(args) for s in samples], [0.0] * len(samples)

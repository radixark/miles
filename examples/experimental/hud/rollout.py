"""Multi-turn computer-use rollout on HUD v6 environments.

The division of labour, and why this file is small:

- **HUD** owns the episode: `task.run()` boots the environment (one Daytona
  sandbox per episode, deleted on exit), runs its agent loop against our
  policy, grades with the task template's second yield, and hands back a
  ``Trace`` whose ``AgentStep``s carry token-level ``Sample``s -- the exact
  prompt/output token ids and sampling logprobs of every turn, straight from
  the inference server (see sglang_compat.py for how stock sglang serves
  those).
- **This file** turns one graded HUD run into one Miles training sample:
  stitch the per-turn token ids into a single sequence with a loss mask, and
  rebuild the vision inputs (pixel_values) from the very screenshots the
  policy saw.

Stitching leans on a property verified per episode rather than assumed: the
server re-renders the chat template every turn, so turn k+1's prompt ids must
start with turn k's prompt+output ids. Where that stops holding, the episode is
kept only up to that turn -- training on a mis-stitched sequence would corrupt
silently, and the turns before the break are still exact.

One scope constraint: every sample this file emits carries at least one image
(episodes whose kept turns predate the first screenshot are written off, and
even write-offs carry a dummy one -- see ``_failed`` for why Miles' FSDP
trainer requires it). Text-only HUD tasks are therefore out of scope here.

Wire with:
  --custom-generate-function-path examples.experimental.hud.rollout.generate
  --custom-rm-path examples.experimental.hud.rollout.reward_func
  --custom-config-path examples/experimental/hud/hud2048_config.yaml
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_runtime = None
_runtime_lock = threading.Lock()
_gate: asyncio.Semaphore | None = None


def _load_daytona_key(args: Any) -> None:
    """Put the Daytona key in this process's environment, from a file if need be.

    Reading it here rather than exporting it before launch keeps it out of Ray's
    ``runtime_env``, which is recorded -- and echoed into job logs -- verbatim.
    """
    if os.environ.get("DAYTONA_API_KEY"):
        return
    path = getattr(args, "hud_daytona_key_file", None) or os.environ.get(
        "DAYTONA_API_KEY_FILE", "~/.config/daytona/api_key"
    )
    path = Path(path).expanduser()
    if not path.exists():
        raise ValueError(f"no Daytona credentials: set DAYTONA_API_KEY or put a key at {path}")
    os.environ["DAYTONA_API_KEY"] = path.read_text().strip()


def _shared_runtime(args: Any):
    """One DaytonaRuntime for the whole process.

    Sharing matters: the runtime resolves (and, when stale, rebuilds) the env
    snapshot once per process; per-episode instances would race on that.
    """
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _load_daytona_key(args)
            from daytona import Image  # noqa: PLC0415 - not part of the training image
            from hud.eval import DaytonaRuntime  # noqa: PLC0415

            dockerfile = Path(args.hud_env_dir).expanduser() / "Dockerfile.hud"
            _runtime = DaytonaRuntime(
                args.hud_snapshot_name,
                image=Image.from_dockerfile(dockerfile) if dockerfile.exists() else None,
            )
        return _runtime


def _sandbox_gate(args: Any) -> asyncio.Semaphore:
    """Cap concurrent sandboxes process-wide, independent of batch geometry."""
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(int(getattr(args, "hud_max_sandboxes", 32)))
    return _gate


def _build_agent(args: Any, sampling_params: dict):
    from examples.experimental.hud.agent import SYSTEM_PROMPT, ComputerChatAgent  # noqa: PLC0415
    from examples.experimental.hud.sglang_compat import sglang_client  # noqa: PLC0415
    from hud.agents.types import OpenAIChatConfig  # noqa: PLC0415

    max_steps = int(getattr(args, "hud_max_steps", 24))
    system_prompt = getattr(args, "hud_system_prompt", None) or SYSTEM_PROMPT
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"
    return ComputerChatAgent.for_shot_width(int(getattr(args, "hud_shot_width", 640)))(
        OpenAIChatConfig(
            model_client=sglang_client(base_url),
            model=args.hf_checkpoint,
            system_prompt=system_prompt.format(max_steps=max_steps),
            max_steps=max_steps,
            completion_kwargs={
                "max_tokens": int(getattr(args, "hud_max_tokens_per_turn", 512)),
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
                "extra_body": {"return_token_ids": True},
            },
        )
    )


def _mint_task(meta: dict):
    from hud import Task  # noqa: PLC0415

    return Task(env=meta["env"], id=meta["id"], args=meta.get("args") or {})


def _collect_turns(run) -> list:
    """The turns that produced tokens, in order."""
    from hud.agents.types import AgentStep  # noqa: PLC0415

    samples = run.trace.collect(lambda s: s.sample if isinstance(s, AgentStep) else None)
    return [s for s in samples if s is not None and s.output_token_ids]


def _collect_images(run) -> list[PILImage.Image]:
    """Every screenshot the policy was shown, in message order.

    The computer tool returns exactly one image per successful call, and the
    harness forwards the last image of each tool result to the model, so the
    images the model saw are the tool results' images, in step order.
    """
    import mcp.types as mcp_types  # noqa: PLC0415
    from hud.agents.types import ToolStep  # noqa: PLC0415

    images: list[PILImage.Image] = []
    for step in run.trace.steps:
        if not isinstance(step, ToolStep) or step.result is None:
            continue
        last = None
        for block in step.result.content:
            if isinstance(block, mcp_types.ImageContent):
                last = block
        if last is not None:
            images.append(PILImage.open(io.BytesIO(base64.b64decode(last.data))).convert("RGB"))
    return images


def _resync_whitespace(
    tokens: list[int],
    loss_mask: list[int],
    logprobs: list[float],
    prompt: list[int],
    response_start: int,
    tokenizer,
) -> tuple[list[int], list[int], list[float]] | None:
    """Realign an accumulated sequence with a re-rendered prompt that differs
    only in whitespace, or return None.

    The one benign divergence: the model emits a blank line before its tool
    call (``text\\n\\n<tool_call>``) and the tool parser + chat template
    re-render it with a single newline. The tokenizer merges adjacent
    punctuation into those tokens (``".\\n\\n"`` vs ``".\\n"`` are single,
    different tokens), so the check is on decoded *text*: at each divergence
    a small window on both sides must become equal once whitespace is
    stripped, and the sequences must realign exactly after it -- a real
    history rewrite cannot pass that. The re-render's tokens are adopted
    (masked 0, logprob 0): later turns were conditioned on the canonical
    form, so this keeps the training sequence identical to the context the
    policy actually saw, at the cost of a token or two of loss per
    occurrence. The returned tokens are a verbatim prefix of ``prompt``.
    """

    def sans_ws(ids: list[int]) -> str:
        return "".join(tokenizer.decode(ids).split())

    WINDOW = 4
    out_t: list[int] = []
    out_m: list[int] = []
    out_l: list[float] = []
    i = j = 0
    while i < len(tokens):
        if j < len(prompt) and tokens[i] == prompt[j]:
            out_t.append(tokens[i])
            if i >= response_start:
                out_m.append(loss_mask[i - response_start])
                out_l.append(logprobs[i - response_start])
            i += 1
            j += 1
            continue
        if i < response_start:
            return None  # the fixed first prompt was rewritten: a real break
        # Smallest window (wi, wj) -- allowing an empty side for pure
        # whitespace insertion/deletion -- whose texts match sans whitespace
        # and after which the token streams realign.
        found = None
        for wi in range(i, min(i + WINDOW, len(tokens)) + 1):
            for wj in range(j, min(j + WINDOW, len(prompt)) + 1):
                if wi == i and wj == j:
                    continue
                if sans_ws(tokens[i:wi]) != sans_ws(prompt[j:wj]):
                    continue
                if wi == len(tokens) or (wj < len(prompt) and tokens[wi] == prompt[wj]):
                    found = (wi, wj)
                    break
            if found:
                break
        if found is None:
            return None
        wi, wj = found
        out_t += prompt[j:wj]
        out_m += [0] * (wj - j)
        out_l += [0.0] * (wj - j)
        i, j = wi, wj
    return out_t, out_m, out_l


def _stitch(turns, tokenizer=None) -> tuple[list[int], list[int], list[float], int] | None:
    """Per-turn (prompt_ids, output_ids) -> one training sequence.

    Returns (tokens, loss_mask, rollout_log_probs, response_start), where the
    mask and logprobs cover the response region (everything after the first
    turn's prompt): 1/logprob on tokens the policy emitted, 0/0.0 on the chat
    scaffolding and screenshots re-rendered into later prompts.

    A turn whose prompt stops reproducing the accumulated sequence ends the
    stitch: the turns before it are still exactly what the policy emitted, so
    the episode is kept up to that point rather than thrown away. (The harness
    rewrites a turn's history when it cannot parse the tool call it produced --
    a turn truncated mid-JSON, say -- which is one way this happens.) One
    divergence is tolerated because it is canonicalization, not rewriting:
    whitespace-only differences resync via ``_resync_whitespace``.
    """
    tokens: list[int] = []
    loss_mask: list[int] = []
    logprobs: list[float] = []
    response_start = 0
    for k, s in enumerate(turns):
        prompt = list(s.prompt_token_ids)
        output = list(s.output_token_ids)
        if k == 0:
            tokens = prompt
            response_start = len(prompt)
        else:
            if prompt[: len(tokens)] != tokens:
                resynced = (
                    _resync_whitespace(tokens, loss_mask, logprobs, prompt, response_start, tokenizer)
                    if tokenizer is not None
                    else None
                )
                if resynced is None:
                    pairs = zip(tokens, prompt, strict=False)
                    d = next((i for i, (a, b) in enumerate(pairs) if a != b), len(tokens))
                    logger.warning(
                        "hud stitch ends early: turn %d diverges at token %d/%d; keeping %d turn(s)",
                        k,
                        d,
                        len(tokens),
                        k,
                    )
                    if tokenizer is not None:
                        lo = max(0, d - 24)
                        logger.warning(
                            "hud stitch divergence detail: emitted %r != re-rendered %r",
                            tokenizer.decode(tokens[lo : d + 24]),
                            tokenizer.decode(prompt[lo : d + 24]),
                        )
                    break
                tokens, loss_mask, logprobs = resynced
                logger.info("hud stitch resynced whitespace at turn %d", k)
            delta = prompt[len(tokens) :]
            tokens += delta
            loss_mask += [0] * len(delta)
            logprobs += [0.0] * len(delta)
        out_lp = list(s.output_logprobs)
        if len(out_lp) != len(output):
            out_lp = [0.0] * len(output)
        tokens += output
        loss_mask += [1] * len(output)
        logprobs += out_lp
    if not loss_mask:
        return None  # not one usable turn
    return tokens, loss_mask, logprobs, response_start


def _vision_inputs(state: Any, images: list[PILImage.Image], tokens: list[int]) -> dict | None:
    """pixel_values/image_grid_thw for the stitched sequence, verified against it.

    The trace usually holds one more screenshot than the ids: the final tool
    call's image lands after the loop's last model call and never enters a
    prompt. So the match is by *prefix* -- the longest run of screenshots whose
    cumulative image-token count equals the count in the ids. Anything but an
    exact prefix match is a processing disagreement and fails loudly.
    """
    if not images:
        return None
    import torch  # noqa: PLC0415

    image_token_id = state.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    actual = sum(1 for t in tokens if t == image_token_id)
    if actual == 0:
        return None

    processed = state.processor.image_processor(images=images, return_tensors="pt")
    grid = processed["image_grid_thw"]
    merge = getattr(state.processor.image_processor, "merge_size", 2)
    per_image = (torch.prod(grid, dim=-1) // (merge * merge)).tolist()

    total, keep = 0, 0
    for count in per_image:
        if total == actual:
            break
        total += int(count)
        keep += 1
    if total != actual:
        raise ValueError(
            f"image token mismatch: ids have {actual}, screenshots produce "
            f"{int(sum(per_image))} (no prefix matches)"
        )
    if keep == len(images):
        return {"pixel_values": processed["pixel_values"], "image_grid_thw": grid}
    trimmed = state.processor.image_processor(images=images[:keep], return_tensors="pt")
    return {"pixel_values": trimmed["pixel_values"], "image_grid_thw": trimmed["image_grid_thw"]}


def _failed(sample: Sample, reason: str, state: Any) -> Sample:
    """A structurally valid write-off that trains on nothing.

    Downstream expects every field list-shaped and consistent even for
    write-offs (``remove_sample`` zeroes the loss but the sample still travels
    with the batch), so a failure must not leave ``None``s behind.

    The write-off carries one minimal image, not just a pad token: under FSDP
    every rank must invoke the vision tower the same number of times per
    microbatch, because its layers' weight all-gathers are collectives. A
    text-only sample skips them, and the ranks that did issue them block until
    another phase happens to post a matching call -- a many-minute stall per
    step, or a watchdog kill under the default NCCL timeout (miles#2406). One
    dummy screenshot's worth of image tokens keeps every sample on the same
    collective schedule.
    """
    logger.warning(reason)
    import torch  # noqa: PLC0415

    tok = state.processor.tokenizer
    processed = state.processor.image_processor(images=[PILImage.new("RGB", (64, 64))], return_tensors="pt")
    merge = getattr(state.processor.image_processor, "merge_size", 2)
    n_image_tokens = int(torch.prod(processed["image_grid_thw"], dim=-1).sum().item()) // (merge * merge)

    sample.status = Sample.Status.FAILED
    sample.remove_sample = True
    sample.tokens = (
        [tok.convert_tokens_to_ids("<|vision_start|>")]
        + [tok.convert_tokens_to_ids("<|image_pad|>")] * n_image_tokens
        + [tok.convert_tokens_to_ids("<|vision_end|>")]
    )
    sample.multimodal_train_inputs = {
        "pixel_values": processed["pixel_values"],
        "image_grid_thw": processed["image_grid_thw"],
    }
    sample.response_length = 0
    sample.response = ""
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.metadata.setdefault("reward", 0.0)
    return sample


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for HUD rollouts."

    # In-function on purpose: sglang_rollout pulls in sglang_router, which only
    # exists on a training node; keeping it out of module scope is what lets
    # the offline tests import everything above.
    from miles.rollout.sglang_rollout import GenerateState  # noqa: PLC0415

    state = GenerateState(args)
    meta = (sample.metadata or {}).get("hud_task")
    if not meta:
        raise ValueError("no HUD task row: expected sample.metadata['hud_task'] = {env, id, args}")

    task = _mint_task(meta)
    agent = _build_agent(args, dict(sampling_params))
    runtime = _shared_runtime(args)
    timeout = float(getattr(args, "hud_episode_timeout_s", 1800))

    async with _sandbox_gate(args):
        try:
            job = await task.run(agent, runtime=runtime, rollout_timeout=timeout)
        except Exception as e:  # noqa: BLE001
            # One episode runs one cloud sandbox, and a long run starts
            # hundreds: a provider hiccup is a routine event, not a reason to
            # take the rollout down. Written off with the traceback kept, so a
            # systematic failure still shows up as a rising drop count rather
            # than as silence.
            logger.warning("hud episode failed to run: %s", e, exc_info=True)
            return _failed(sample, f"hud episode errored: {type(e).__name__}", state)

    run = (getattr(job, "runs", None) or [None])[0]
    if run is None:
        return _failed(sample, "hud episode produced no run", state)

    sample.metadata["reward"] = float(run.reward or 0.0)
    sample.metadata["hud_trace_status"] = str(run.trace.status)

    turns = _collect_turns(run)
    stitched = _stitch(turns, state.tokenizer) if turns else None
    if stitched is None:
        # No token data, or the prefix property broke. Either way there is
        # nothing safe to train on in this episode.
        return _failed(sample, f"hud episode unusable (turns={len(turns)}, stitch broke)", state)

    tokens, loss_mask, logprobs, response_start = stitched
    try:
        vision = _vision_inputs(state, _collect_images(run), tokens)
    except ValueError as e:
        return _failed(sample, f"hud episode dropped: {e}", state)
    if vision is None:
        # The stitch ended before the first screenshot came back, so the kept
        # turns are text-only. Written off for the same vision-tower collective
        # parity that shapes _failed -- and there is next to nothing to learn
        # from a turn the harness could not even parse a tool call out of.
        return _failed(sample, "hud episode kept no screenshot", state)
    sample.multimodal_train_inputs = vision

    sample.tokens = tokens
    sample.response_length = len(tokens) - response_start
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = logprobs
    sample.response = state.tokenizer.decode(tokens[response_start:], skip_special_tokens=False)
    sample.status = Sample.Status.COMPLETED
    logger.info(
        "[hud %s] reward=%.3f turns=%d tokens=%d (response=%d)",
        meta.get("slug", meta["id"]),
        sample.metadata["reward"],
        len(turns),
        len(tokens),
        sample.response_length,
    )
    return sample


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Pass through the reward HUD's task template computed inside the env."""
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)

"""Multi-turn text-to-SQL episode and reward on the River text-to-SQL recipe's tools and grader.

The prompt, the read-only SQLite tool, the grader and the reward terms come from the `text_to_sql_recipe` package;
only its pure modules are imported, never the River client. Each turn either calls `execute_sql_query` once or
answers; after the last tool turn the next turn is graded as the answer whatever it contains. A query that tries to
write or escape the sandbox ends the rollout as unsafe.

Observations are appended to the sampled ids as the chat template frames a new user message, never re-rendered, so
the trainer sees exactly the tokens the engine sampled.
"""

import argparse
import asyncio
from copy import deepcopy

from text_to_sql_recipe import grading, parsing, prompts, reward, sqlite_exec
from text_to_sql_recipe.config import RewardConfig

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import append_observation_tokens
from miles.utils.http_utils import post
from miles.utils.types import Sample

_TRAIN_REWARD = RewardConfig(verifier="none", redraft_penalty=0.0)
# evaluation scores the outcome only, as the recipe's evaluation does
_EVAL_REWARD = RewardConfig(verifier="none", process_penalty_weight=0.0, redraft_penalty=0.0)

# SQLite runs in worker threads; this bounds the tool calls and grading queries in flight together
_SQL_SLOTS = asyncio.Semaphore(16)
# gold SQL is slow on the largest databases, so each task's result is computed once per process
_GOLD_RESULTS: dict[str, asyncio.Task] = {}


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    task = sample.metadata["task"]
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    max_tool_turns = args.text_to_sql_max_tool_turns

    episode = {"turns": [], "final_text": "", "unsafe": False, "tool_calls": 0, "tool_protocol_events": []}
    sample.metadata["text_to_sql"] = episode
    sample.metadata["evaluation"] = input.evaluation
    sample.tokens = compute_prompt_ids_from_sample(input.state, sample)

    for turn in range(1, max_tool_turns + 2):
        payload, halt_status = compute_request_payload(
            args,
            sample.tokens,
            input.sampling_params,
            evaluation=input.evaluation,
            kv_cache_namespace=sample.kv_cache_namespace,
        )
        if payload is None:
            sample.status = halt_status
            break
        output = await post(url, payload, headers=compute_routing_headers(args, sample))
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason == "abort":
            break

        text = output["text"]
        episode["turns"].append(text)
        message = {"role": "assistant", "content": _visible_text(text)}
        tools_allowed = turn <= max_tool_turns
        reward.observe_tool_calls(episode["tool_protocol_events"], message, turn=turn, tools_allowed=tools_allowed)
        if finish_reason == "length" or not tools_allowed:
            episode["final_text"] = message["content"].strip()
            break

        call = next((c for c in parsing.parse_tool_calls(message) if c.name == prompts.SQL_TOOL_NAME), None)
        if call is not None:
            episode["tool_calls"] += 1
            observation = await _sql_observation(task, call.arguments, turns_left=max_tool_turns - turn)
            if observation is None:
                episode["unsafe"] = True
                break
        elif parsing.has_tool_call_intent(message):
            observation = parsing.MALFORMED_ACTION_PROMPT
        else:
            episode["final_text"] = message["content"].strip()
            break

        messages = [{"role": "user", "content": observation}]
        if turn == max_tool_turns:
            messages.append({"role": "user", "content": prompts.SQL_LIMIT_PROMPT})
        observation_ids = _observation_ids(tokenizer, messages, last_token_id=sample.tokens[-1])
        if args.rollout_max_context_len and len(sample.tokens) + len(observation_ids) >= args.rollout_max_context_len:
            sample.status = Sample.Status.TRUNCATED
            break
        append_observation_tokens(sample, observation_ids, tokenizer=tokenizer)

    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--text-to-sql-max-tool-turns", type=int, default=5)


generate.add_arguments = _add_arguments


async def reward_func(args, sample: Sample, **kwargs) -> dict:
    task, episode = sample.metadata["task"], sample.metadata["text_to_sql"]
    if episode["unsafe"]:
        grade = {"grade": "INCORRECT", "error_category": "unsafe_sql", "format_compliant": False}
    else:
        graded_task, expected = await _gold_result(task)
        async with _SQL_SLOTS:
            grade = await asyncio.to_thread(grading.grade_with_expected, episode["final_text"], graded_task, expected)

    scored = reward.rollout_reward(
        grade,
        _EVAL_REWARD if sample.metadata["evaluation"] else _TRAIN_REWARD,
        task=task,
        messages=[{"role": "assistant", "content": text} for text in episode["turns"]],
        prompt_message_count=0,
        unsafe=episode["unsafe"],
        refuted=False,
        redrafts=0.0,
        loop_stopped=False,
        tool_protocol_events=episode["tool_protocol_events"],
    )
    return {
        "score": scored.reward,
        "correct": float(grade["grade"] == "CORRECT"),
        "format_compliant": float(bool(grade.get("format_compliant"))),
        "tool_calls": float(episode["tool_calls"]),
        "process_penalty": scored.metrics["process_penalty"],
        "invalid_tool_calls": scored.metrics["tool_protocol/invalid_calls"],
    }


def _visible_text(text: str) -> str:
    """The turn without its reasoning, which is what the recipe parses tool calls and the answer from.

    The generation prompt opens a `<think>` block, so a turn that never closes it is all reasoning.
    """
    _, closed, tail = text.partition("</think>")
    return tail if closed else ""


def _observation_ids(tokenizer, messages: list[dict], *, last_token_id: int) -> list[int]:
    """The ids that close the sampled turn and open the next one, as the recipe's incremental framing does."""
    stop = "" if last_token_id == tokenizer.eos_token_id else tokenizer.eos_token
    framing = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(stop + "\n" + framing, add_special_tokens=False)


async def _sql_observation(task: dict, arguments: dict, *, turns_left: int) -> str | None:
    """The user message reporting one SQL call, or None when the query was unsafe."""
    try:
        async with _SQL_SLOTS:
            results = await asyncio.to_thread(
                sqlite_exec.execute_sql_tool, task, sqlite_exec.tool_executor(task), arguments
            )
    except sqlite_exec.SQLiteInfrastructureError:
        raise
    except sqlite_exec.SQL_TOOL_TERMINAL_ERRORS:
        return None
    except Exception as exc:  # as in the recipe's SqlEnv: any other error is shown to the model
        error = (
            str(exc).encode("utf-8", "ignore")[: prompts.SQL_TOOL_CONTEXT_ERROR_MAX_BYTES].decode("utf-8", "ignore")
        )
        return (
            f"Results for {prompts.SQL_TOOL_NAME}:\nError: {error} {prompts.SQL_TOOL_CONTEXT_ERROR_GUIDANCE}"
            f"\n\n{prompts.tool_result_prompt()}"
        )
    body = "\n\n".join(result.format() for result in results) or "(no output)"
    return (
        f"Results for {prompts.SQL_TOOL_NAME}:\n{body}\n\n{prompts.tool_result_prompt()}"
        f"\n\nYou have {turns_left} turns left to complete the task."
    )


async def _gold_result(task: dict):
    """The task as graded and its gold result."""
    task_id = str(task["id"])
    if task_id not in _GOLD_RESULTS:
        _GOLD_RESULTS[task_id] = asyncio.create_task(asyncio.to_thread(_execute_gold, dict(task)))
    return await _GOLD_RESULTS[task_id]


def _execute_gold(task: dict):
    grading.prepare_streaming_grading_storage([task])
    failures = {}
    results = grading.gold_results([task], concurrency=1, failures=failures)
    if not results:
        raise RuntimeError(f"gold SQL failed for task {task['id']}: {failures}")
    return task, results[str(task["id"])]

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
import dataclasses
import functools
from copy import deepcopy

from text_to_sql_recipe import config, grading, parsing, prompts, reward, sqlite_exec
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

# SQLite runs in worker threads; this bounds tool calls, grading and gold queries together, and lets a grading
# retry drain the other work, as the recipe's grading runtime does
_SQL_WORK = sqlite_exec.SqlWork(16)
# gold SQL is slow on the largest databases, so each task's result is computed once per process
_GOLD_RESULTS: dict[str, asyncio.Task] = {}


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    task = sample.metadata["task"]
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    max_tool_turns = args.text_to_sql_max_tool_turns
    turn_token_cap = input.sampling_params.get("max_new_tokens", args.rollout_max_response_len)

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
        if finish_reason == "length" and payload["sampling_params"]["max_new_tokens"] < turn_token_cap:
            # the context budget cut this turn; the recipe never grades such a turn, so it scores as no answer
            break
        if not tools_allowed:
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
        if args.max_seq_len and len(sample.tokens) + len(observation_ids) >= args.max_seq_len:
            sample.status = Sample.Status.TRUNCATED
            break
        append_observation_tokens(sample, observation_ids, tokenizer=tokenizer)

    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--text-to-sql-max-tool-turns", type=int, default=5)
    parser.add_argument("--text-to-sql-recipe", type=str, default="qwen3.6-35b-a3b")


generate.add_arguments = _add_arguments


async def reward_func(args, sample: Sample, **kwargs) -> dict:
    task, episode = sample.metadata["task"], sample.metadata["text_to_sql"]
    if episode["unsafe"]:
        grade = {"grade": "INCORRECT", "error_category": "unsafe_sql", "format_compliant": False}
    else:
        graded_task, expected = await _gold_result(task)
        grade = await _grade(episode["final_text"], graded_task, expected)

    scored = reward.rollout_reward(
        grade,
        _reward_config(args.text_to_sql_recipe, evaluation=sample.metadata["evaluation"]),
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


@functools.cache
def _reward_config(recipe: str, *, evaluation: bool) -> RewardConfig:
    """The recipe preset's reward terms, without the VeriEQL verifier and re-draft guard this example does not port."""
    train = dataclasses.replace(config.load(recipe).reward, verifier="none", redraft_penalty=0.0)
    if not evaluation:
        return train
    # evaluation scores the outcome only, as the recipe's evaluation does
    return dataclasses.replace(train, process_penalty_weight=0.0, tool_format_penalty=0.0)


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


async def _grade(final_text: str, task: dict, expected) -> dict:
    """Grade the answer, retrying a timeout with the other SQL work drained, as the recipe's `GradingRuntime` does."""
    for attempt in range(grading.GRADING_TIMEOUT_RETRIES + 1):
        last_attempt = attempt == grading.GRADING_TIMEOUT_RETRIES
        try:
            grade = await _SQL_WORK.execute(
                grading.grade_with_expected, final_text, task, expected, exclusive=attempt > 0
            )
        except sqlite_exec.SQLiteInfrastructureError as error:
            if last_attempt or not grading.numeric_comparison_timed_out(error):
                raise
            continue
        if last_attempt or grade.get("error_category") != "timeout":
            return grade


def _run_sql_tool(task: dict, arguments: dict):
    # building the executor hashes and may snapshot the whole database, so it must stay off the event loop
    return sqlite_exec.execute_sql_tool(task, sqlite_exec.tool_executor(task), arguments)


async def _sql_observation(task: dict, arguments: dict, *, turns_left: int) -> str | None:
    """The user message reporting one SQL call, or None when the query was unsafe."""
    try:
        results = await _SQL_WORK.execute(_run_sql_tool, task, arguments)
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
        pending = asyncio.create_task(_SQL_WORK.execute(_execute_gold, dict(task)))
        _GOLD_RESULTS[task_id] = pending
        pending.add_done_callback(functools.partial(_forget_failed_gold, task_id))
    return await asyncio.shield(_GOLD_RESULTS[task_id])


def _forget_failed_gold(task_id: str, pending: asyncio.Task) -> None:
    """Drop a failed gold result so the next sample of the task runs the gold query again."""
    if (pending.cancelled() or pending.exception() is not None) and _GOLD_RESULTS.get(task_id) is pending:
        del _GOLD_RESULTS[task_id]


def _execute_gold(task: dict):
    grading.prepare_streaming_grading_storage([task])
    failures = {}
    results = grading.gold_results([task], concurrency=1, failures=failures)
    if not results:
        raise RuntimeError(f"gold SQL failed for task {task['id']}: {failures}")
    return task, results[str(task["id"])]

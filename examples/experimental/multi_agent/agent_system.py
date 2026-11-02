import asyncio
import re
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from examples.experimental.multi_agent.prompts import (
    SOLVER_PROMPT_TEMPLATE,
    generate_rewriter_template,
    generate_select_template,
)
from transformers import PreTrainedTokenizerBase

from miles.rollout.rm_hub import batched_async_rm
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.http_utils import post
from miles.utils.types import Sample


@dataclass
class AgentState:
    sample: Sample
    tokenizer: PreTrainedTokenizerBase
    sampling_params: dict[str, Any]
    results: dict[str, list[Sample]] = field(default_factory=lambda: {"solver": [], "rewriter": [], "selector": []})


async def generate_response(args: ImmutableNamespace, prompt: str, key: str, *, state: AgentState) -> str | None:
    try:
        sampling_params = state.sampling_params
        tokenizer = state.tokenizer
        max_context_length = args.rollout_max_context_len
        sample = deepcopy(state.sample)

        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        sample.prompt = prompt
        prompt_token_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_token_ids
        prompt_length = len(prompt_token_ids)
        current_sampling_params = deepcopy(sampling_params)
        current_sampling_params["max_new_tokens"] = min(
            sampling_params["max_new_tokens"], max_context_length - prompt_length
        )

        if current_sampling_params["max_new_tokens"] <= 0:
            return None

        payload = {"input_ids": prompt_token_ids, "sampling_params": current_sampling_params, "return_logprob": True}

        output = await post(url, payload)

        # Extract new response tokens
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            # abort
            new_response_tokens = []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response = output["text"]

        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            # case "abort":
            #     sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

        state.results[key].append(sample)

        final = output["text"].replace("<|user|>", "")
        if "</think>" in final:
            contents = final.split("</think>")
            if len(contents) == 2 and contents[1] != "":
                reason_content = contents[0].strip()
                response_content = contents[1].strip()
                sample.reason_content = reason_content
                sample.response_content = response_content
                return response_content
        sample.reason_content = None
        sample.response_content = None
        return None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


class Agent:
    """A base class for our AI agents."""

    def __init__(self):
        pass

    async def run(
        self, args: ImmutableNamespace, prompt: str, *, state: AgentState, key: str, max_retries: int = 1
    ) -> str | None:
        """Runs the agent by sending a prompt to the LLM."""
        for _i in range(max_retries):
            try:
                response = await generate_response(args, prompt, key=key, state=state)
                return response
            except Exception as e:
                print(f"Error querying LLM: {e}")
                time.sleep(1)
        print(f"Failed to query LLM after {max_retries} retries")
        return None


class SolverAgent(Agent):
    """The agent responsible for generating and improving solutions."""

    def __init__(self):
        super().__init__()

    async def generate_initial_solution(
        self, args: ImmutableNamespace, problem_statement: str, *, state: AgentState
    ) -> str | None:
        """Generates the first solution attempt."""
        prompt = SOLVER_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, prompt, state=state, max_retries=3, key="solver")


class RewriterAgent(Agent):
    """The agent responsible for rewriting solutions."""

    def __init__(self):
        super().__init__()

    async def rewrite(
        self, args: ImmutableNamespace, problem_statement: str, previous_solutions: list[str], *, state: AgentState
    ) -> str | None:
        """Generates the rewrited solution."""

        # 动态生成模板
        template = generate_rewriter_template(len(previous_solutions))

        # 构建参数字典
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(previous_solutions):
            format_params[f"solution{i+1}"] = solution

        prompt = template.format(**format_params)
        return await self.run(args, prompt, state=state, max_retries=1, key="rewriter")


class SelectorAgent(Agent):
    """The agent responsible for selecting solutions."""

    def __init__(self):
        super().__init__()

    async def select(
        self, args: ImmutableNamespace, problem_statement: str, candidate_solutions: list[str], *, state: AgentState
    ) -> str | None:
        """Generates the rewrited solution."""

        # 动态生成模板
        template = generate_select_template(len(candidate_solutions))

        # 构建参数字典
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(candidate_solutions):
            format_params[f"solution{i+1}"] = solution

        prompt = template.format(**format_params)
        return await self.run(args, prompt, state=state, max_retries=10, key="selector")

    def extract_selected_solution_idx(self, response: str, candidate_solutions: list[str]) -> int:
        """Extracts the selected solution ID from the response."""
        PATTERN = re.compile(r"Judgment:\s*(\d+)")
        matched = PATTERN.findall(response)
        try:
            selected_id = int(matched[0]) - 1
            if selected_id < len(candidate_solutions) and selected_id >= 0:
                return selected_id
            else:
                return None
        except Exception as e:
            print(f"extract_selected_solution_idx error: {e}")
            return None


async def rewrite_worker(
    args: ImmutableNamespace,
    previous_solutions: list[str],
    problem_statement: str,
    worker_id: int,
    *,
    state: AgentState,
) -> str | None:
    rewriter = RewriterAgent()
    new_solution = await rewriter.rewrite(args, problem_statement, previous_solutions, state=state)
    return new_solution


async def solver_worker(
    args: ImmutableNamespace, problem_statement: str, worker_id: int, *, state: AgentState
) -> str | None:
    """
    单组 solver 流程。
    """

    try:
        solver = SolverAgent()
        current_solution = await solver.generate_initial_solution(args, problem_statement, state=state)
        return current_solution

    except Exception as e:
        print(f"[Worker-{worker_id}] exception: {e}")
        print(f"[Worker-{worker_id}] traceback: {traceback.format_exc()}")
        return None


async def run_agent_system(args: ImmutableNamespace, *, state: AgentState) -> list[Sample]:
    """
    并发运行 num_parallel 组 pipeline。
    """

    # Each rollout creates its own runtime state in rollout_with_multi_agents.
    problem_statement = state.sample.prompt
    tasks = [solver_worker(args, problem_statement, worker_id, state=state) for worker_id in range(args.num_parallel)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rewards = await batched_async_rm(args=args, samples=state.results["solver"])
    for sample, reward in zip(state.results["solver"], rewards, strict=False):
        sample.reward = reward

    previous_solutions = [item for item in results if isinstance(item, str)]

    def reward_adjustment(samples: list[Sample], reward_weight: float) -> list[Sample]:
        for sample in samples:
            sample.reward = sample.reward * reward_weight
        return samples

    if len(previous_solutions) == 0:
        reward_adjustment(state.results["solver"], args.incorrect_reward_weight)
        return state.results["solver"]

    # Rewriting
    tasks = [
        rewrite_worker(args, previous_solutions, problem_statement, worker_id, state=state)
        for worker_id in range(args.num_parallel)
    ]
    rewrited_solutions_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理异常结果
    rewrited_solutions = []
    for _i, result in enumerate(rewrited_solutions_raw):
        if isinstance(result, str):
            rewrited_solutions.append(result)

    rewards = await batched_async_rm(args=args, samples=state.results["rewriter"])
    for sample, reward in zip(state.results["rewriter"], rewards, strict=False):
        sample.reward = reward

    if len(rewrited_solutions) == 0:
        reward_adjustment(state.results["solver"], args.incorrect_reward_weight)
        reward_adjustment(state.results["rewriter"], args.incorrect_reward_weight)
        return state.results["solver"] + state.results["rewriter"]

    # Selection
    selector = SelectorAgent()
    response = await selector.select(args, problem_statement, rewrited_solutions, state=state)
    if len(state.results["selector"]) == 0:
        reward_adjustment(state.results["solver"], args.incorrect_reward_weight)
        reward_adjustment(state.results["rewriter"], args.incorrect_reward_weight)
        return state.results["solver"] + state.results["rewriter"]

    assert (
        len(state.results["selector"]) == 1
    ), f"selector should only return one solution, but got {len(state.results['selector'])}"
    if response is None:
        state.results["selector"][0].reward = 0
    else:
        selected_solution_idx = selector.extract_selected_solution_idx(response, rewrited_solutions)
        if selected_solution_idx is None:
            state.results["selector"][0].reward = 0
        else:
            selected_solution = rewrited_solutions[selected_solution_idx]
            for sample in state.results["rewriter"]:
                if sample.response_content is not None and selected_solution in sample.response_content:
                    state.results["selector"][0].reward = sample.reward
                    break

    ## 如果最终答案正确，对所有reward添加正向的奖励；如果最终答案不正确，对所有reward添加负向的惩罚。
    if state.results["selector"][0].reward == 1:
        reward_adjustment(state.results["solver"], args.correct_reward_weight)
        reward_adjustment(state.results["rewriter"], args.correct_reward_weight)
        reward_adjustment(state.results["selector"], args.correct_reward_weight)
    else:
        reward_adjustment(state.results["solver"], args.incorrect_reward_weight)
        reward_adjustment(state.results["rewriter"], args.incorrect_reward_weight)
        reward_adjustment(state.results["selector"], args.incorrect_reward_weight)

    return state.results["solver"] + state.results["rewriter"] + state.results["selector"]

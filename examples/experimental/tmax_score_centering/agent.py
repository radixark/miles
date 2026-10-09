"""TMax's training Vanillux protocol, using Harbor environments and Miles TITO.

Adapted from hamishivi/tmax at 6d3d606 (Vanillux2Agent/agent.py and
training/open-instruct/open_instruct/environments/swerl_vanillux_sandbox.py).
The released training messages win over the evaluation agent's system prompt.
Tests remain Harbor-owned and are uploaded only after the agent finishes.
"""

import asyncio
import json
import os
import re
import shlex
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from examples.experimental.harbor.harbor_agent_function import (
    _ensure_provider_key,
    build_trial_config,
    trial_result_to_metadata,
)
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig, VerifierConfig
from harbor.trial.trial import Trial
from openai import AsyncOpenAI

from miles.rollout.agentic.session import openai_session_url

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_PROMPTS = yaml.safe_load(Path(__file__).with_name("vanillux_prompts.yaml").read_text())
_STATE = "/tmp/.tmax_vanillux"
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in a persistent shell. Working directory and environment variables are preserved between calls.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                "required": ["command"],
            },
        },
    }
]

_CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length of (?P<limit>\d+) tokens.*?"
    r"(?P<input>\d+) tokens from the input messages and (?P<completion>\d+) tokens",
    re.DOTALL,
)
_INPUT_CONTEXT_LIMIT_RE = re.compile(
    r"The input \(\d+ tokens\) is longer than the model's context length \(\d+ tokens\)"
)


def _observation(stdout: str, stderr: str, return_code: int) -> str:
    output = stdout + ("\n" if stdout and stderr else "") + stderr
    config = _PROMPTS["observation"]
    head, tail = config["head_chars"], config["tail_chars"]
    if len(output) > config["max_chars"]:
        output = f"{config['too_long_hint']}\n\n---- HEAD ({head} chars) ----\n{output[:head]}\n---- {len(output) - head - tail} chars elided ----\n---- TAIL ({tail} chars) ----\n{output[-tail:]}"
    return f"{output or '(no output)'}\n\n(exit_code={return_code})"


def _wrap_command(command: str) -> str:
    script = f'set -a\nsource {_STATE}/env 2>/dev/null || true\nset +a\ncd "$(cat {_STATE}/cwd)" 2>/dev/null || cd /workspace || exit 1\neval {shlex.quote(command)}\n_tmax_ec=$?\nexport -p > {_STATE}/env\npwd > {_STATE}/cwd\nexit $_tmax_ec'
    return "bash -c " + shlex.quote(script)


class TMaxAgent(BaseAgent):
    """One bash tool, persistent cwd/exported env, and submit-time verification."""

    @staticmethod
    def name() -> str:
        return "tmax-vanillux"

    def version(self) -> str:
        return "6d3d606-training"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str,
        *,
        messages: list[dict],
        api_base: str,
        request_kwargs: dict,
        max_steps: int = 64,
        command_timeout: int = 120,
        preserve_workdir: bool = False,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.messages = deepcopy(messages)
        self.api_base = api_base
        self.request_kwargs = dict(request_kwargs)
        self.max_steps = max_steps
        self.command_timeout = command_timeout
        self.preserve_workdir = preserve_workdir

    async def setup(self, environment: BaseEnvironment) -> None:
        command = f"mkdir -p /workspace /output /logs/verifier {_STATE} && ([ -d /app ] || ln -s /workspace /app) && printf '/app\\n' > {_STATE}/cwd && : > {_STATE}/env"
        if self.preserve_workdir:
            command = f"mkdir -p {_STATE} && pwd > {_STATE}/cwd && export -p > {_STATE}/env"
        result = await environment.exec(command=command, timeout_sec=30)
        if result.return_code:
            raise RuntimeError(f"TMax shell setup failed: {result.stderr}")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        messages = self.messages
        submitted = False
        timings = []
        tool_time = 0.0
        context.n_input_tokens = context.n_output_tokens = 0
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        request = dict(self.request_kwargs)
        standard = {key: request.pop(key) for key in ("max_tokens", "temperature", "top_p", "stop") if key in request}
        try:
            async with AsyncOpenAI(base_url=self.api_base, api_key="dummy", timeout=600, max_retries=0) as client:
                for step in range(self.max_steps):
                    started = time.monotonic()
                    call_standard = dict(standard)
                    try:
                        response = await client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            tools=_TOOLS,
                            **call_standard,
                            extra_body=request,
                        )
                    except Exception as exc:
                        # SGLang rejects a request before generation when the
                        # requested completion plus the already accumulated
                        # conversation exceeds its context window.  Retry the
                        # same turn with the largest completion that fits; a
                        # real transport/model error still propagates.
                        if _INPUT_CONTEXT_LIMIT_RE.search(str(exc)):
                            break
                        match = _CONTEXT_LIMIT_RE.search(str(exc))
                        if not match:
                            raise
                        limit = int(match.group("limit"))
                        input_tokens = int(match.group("input"))
                        available = limit - input_tokens - 64
                        if available < 64:
                            break
                        base_max_tokens = int(call_standard.get("max_tokens", available))
                        call_standard["max_tokens"] = min(base_max_tokens, available)
                        response = await client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            tools=_TOOLS,
                            **call_standard,
                            extra_body=request,
                        )
                    choice = response.choices[0]
                    message = choice.message.model_dump(exclude_none=True)
                    messages.append(message)
                    if response.usage:
                        context.n_input_tokens += response.usage.prompt_tokens
                        context.n_output_tokens += response.usage.completion_tokens
                    if choice.finish_reason == "length":
                        break
                    calls = message.get("tool_calls", [])
                    command = None
                    if len(calls) == 1 and calls[0]["function"]["name"] == "bash":
                        try:
                            command = json.loads(calls[0]["function"]["arguments"]).get("command")
                        except (ValueError, AttributeError):
                            pass
                    if not isinstance(command, str) or not command.strip():
                        error = _PROMPTS["format_error_template"].replace(
                            "{{error}}", "Your last response did not include a valid `bash` tool call."
                        )
                        if calls:
                            messages.extend(
                                {"role": "tool", "tool_call_id": call["id"], "content": error} for call in calls
                            )
                        else:
                            messages.append({"role": "user", "content": error})
                        continue
                    tool_started = time.monotonic()
                    result = await environment.exec(command=_wrap_command(command), timeout_sec=self.command_timeout)
                    elapsed = time.monotonic() - tool_started
                    tool_time += elapsed
                    output = _observation(result.stdout or "", result.stderr or "", result.return_code)
                    messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": output})
                    timings.append({"step": step, "model_s": tool_started - started, "tool_s": elapsed})
                    # Training submits on observed output, not merely a marker in the command text.
                    submitted = SUBMIT_MARKER in (result.stdout or "") + (result.stderr or "")
                    if submitted:
                        break
        finally:
            (self.logs_dir / "trajectory.json").write_text(json.dumps(messages, indent=2))
            (self.logs_dir / "tmax.json").write_text(
                json.dumps(
                    {
                        "submitted": submitted,
                        "turns": len(timings),
                        "total_tool_time": tool_time,
                        "timings": timings,
                    },
                    indent=2,
                )
            )


async def run(base_url: str, prompt: list[dict], request_kwargs: dict, metadata: dict, **kwargs) -> dict:
    """Run a real Harbor trial and return its binary verifier result to Miles."""
    if not isinstance(prompt, list) or [m["role"] for m in prompt] != ["system", "user"]:
        raise ValueError("TMax requires the released messages; do not use --apply-chat-template")
    _ensure_provider_key()
    session_url = openai_session_url(base_url)
    config = build_trial_config(metadata, session_url, request_kwargs)
    if os.environ.get("HARBOR_ENV_TYPE") == "e2b":
        config.environment = config.environment.model_copy(
            update={"import_path": "examples.experimental.tmax_score_centering.environment:TMaxE2BEnvironment"}
        )
    evaluation = metadata.get("split") == "eval"
    if evaluation:
        # Benchmark task resources and stage timeouts win over the training
        # smoke overrides. Do not mutate worker-wide environment variables.
        config.environment = config.environment.model_copy(
            update={"override_cpus": None, "override_memory_mb": None, "override_storage_mb": None}
        )
        config.verifier = VerifierConfig()
        config.timeout_multiplier = 1.0
    config.agent = AgentConfig(
        import_path="examples.experimental.tmax_score_centering.agent:TMaxAgent",
        model_name=os.environ.get("AGENT_MODEL_NAME", "model"),
        override_timeout_sec=None if evaluation else float(os.environ.get("AGENT_TIMEOUT", "900")),
        kwargs={
            "messages": prompt,
            "api_base": session_url,
            "request_kwargs": request_kwargs,
            "max_steps": int(os.environ.get("TMAX_EVAL_MAX_STEPS" if evaluation else "TMAX_MAX_STEPS", "64")),
            "preserve_workdir": evaluation,
        },
    )
    trial = await Trial.create(config)
    # Evaluation relies on Harbor's task-defined stage timeouts; some TB2
    # tasks permit several hours, exceeding the training smoke's outer cap.
    timeout = None if evaluation else float(os.environ.get("AGENT_TRIAL_TIMEOUT", "1500"))
    result = await asyncio.wait_for(trial.run(), timeout=timeout)
    verdict = trial_result_to_metadata(result)
    verdict["trial_dir"] = str(trial.paths.trial_dir)
    has_eval_verdict = evaluation and bool(getattr(result.verifier_result, "rewards", None))
    if verdict["exit_status"] != "Submitted" and not has_eval_verdict:
        raise RuntimeError(f"TMax trial failed: {verdict}")
    agent_result = json.loads((trial.paths.trial_dir / "agent" / "tmax.json").read_text())
    verdict["agent_metrics"].update(agent_result)
    if not evaluation and not agent_result["submitted"]:
        verdict.update(reward=0.0, exit_status="StepOrTokenLimitExceeded")
    return verdict


async def reward_func(args, samples, **kwargs):
    if isinstance(samples, list):
        return [sample.metadata["reward"] for sample in samples]
    return samples.metadata["reward"]

import asyncio
import json
import os
import tempfile
from pathlib import Path

from miles.rollout.agentic.agent_function import InfraAbort

_CONFIG = """model = "policy"
model_provider = "miles"

[model_providers.miles]
name = "miles"
base_url = "{base_url}/v1"
env_key = "MILES_API_KEY"
wire_api = "responses"
request_max_retries = 1
stream_max_retries = 1

[analytics]
enabled = false

[feedback]
enabled = false

[features]
plugins = false
"""


async def run(base_url: str, prompt: list[dict] | str, request_kwargs: dict, metadata: dict, **kwargs) -> dict:
    task = prompt if isinstance(prompt, str) else "\n\n".join(message["content"] for message in prompt)
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as workdir:
        Path(home, "config.toml").write_text(_CONFIG.format(base_url=base_url))
        events_path, stderr_path = Path(home, "events.jsonl"), Path(home, "stderr.log")
        with events_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = await asyncio.create_subprocess_exec(
                *("codex", "exec", "--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"),
                *("--cd", workdir, "--", task),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env={**os.environ, "CODEX_HOME": home, "MILES_API_KEY": "sk-dummy"},
            )
        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), metadata.get("timeout_s", 600))
        except TimeoutError:
            timed_out = True
        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        if timed_out and events and events[-1]["type"] == "error":
            raise InfraAbort("ModelUnreachable", events[-1]["message"])
        if timed_out:
            return {"reward": 0.0, "exit_status": "TimeLimitExceeded"}
        failed = [event for event in events if event["type"] == "turn.failed"]
        if failed:
            raise InfraAbort("CodexTurnFailed", failed[-1]["error"]["message"])
        if proc.returncode != 0 or not any(event["type"] == "turn.completed" for event in events):
            raise InfraAbort(f"CodexExit{proc.returncode}", stderr_path.read_text(errors="replace")[-2000:])
        answer = Path(workdir, "answer.txt")
        correct = answer.is_file() and answer.read_text(errors="replace").strip() == str(metadata["answer"]).strip()
        return {"reward": float(correct), "exit_status": "Submitted"}

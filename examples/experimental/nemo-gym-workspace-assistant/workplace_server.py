"""Thin HTTP transport for NeMo Gym's unmodified Workplace tools and verifier.

Gold actions stay in the server's task catalog; policy requests contain task IDs.
Run with a pinned Gym checkout on PYTHONPATH and this example's requirements.
"""

import json
import logging
import math
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from resources_servers.workplace_assistant.utils import execute_actions_and_reset_state, get_tools, is_correct
from tap import Tap

logger = logging.getLogger(__name__)
DOMAINS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
TABLES = {
    "calendar": "_calendar_events",
    "email": "_emails",
    "analytics": "_plots_data",
    "project_management": "_project_tasks",
    "customer_relationship_manager": "_crm_data",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def tables(env: Mapping[str, Any]) -> dict:
    return {
        domain: pl.from_pandas(getattr(env["containers"][domain], attr), nan_to_null=True).to_dicts()
        for domain, attr in TABLES.items()
    }


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: str


@dataclass
class Episode:
    task: dict
    env: dict = field(default_factory=lambda: get_tools(DOMAINS))
    actions: list = field(default_factory=list)
    updated: float = field(default_factory=time.monotonic)
    lock: Any = field(default_factory=threading.Lock)


class Workplace:
    def __init__(self, dataset: Path, ttl: float = 10800) -> None:
        rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
        self.tasks = {row["id"]: row for row in rows}
        if len(self.tasks) != len(rows) or not rows:
            raise ValueError("Task IDs must be unique and dataset nonempty")
        self.sessions: dict[str, Episode] = {}
        self.lock = threading.Lock()
        self.ttl = ttl
        self.graded = 0
        self.solved = 0

    def create(self, task_id: int) -> str:
        if task_id not in self.tasks:
            raise HTTPException(404, "Unknown task ID")
        episode = Episode(task=self.tasks[task_id])
        session_id = uuid.uuid4().hex
        with self.lock:
            cutoff = time.monotonic() - self.ttl
            self.sessions = {k: v for k, v in self.sessions.items() if v.updated > cutoff}
            self.sessions[session_id] = episode
        return session_id

    def get(self, session_id: str) -> Episode:
        with self.lock:
            episode = self.sessions.get(session_id)
        if episode is None:
            raise HTTPException(404, "Unknown or expired episode")
        return episode

    def delete(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)

    def tool(self, session_id: str, call: ToolCall) -> dict:
        episode = self.get(session_id)
        with episode.lock:
            episode.updated = time.monotonic()
            allowed = {tool["name"] for tool in episode.task["responses_create_params"]["tools"]}
            if call.name not in allowed:
                return {"output": "Error executing tool: tool is not declared for this task"}
            episode.actions.append(call.model_dump())
            try:
                args = json.loads(call.arguments)
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                # Exactly the native resource-server rule: omit JSON null kwargs.
                args = {key: value for key, value in args.items() if value is not None}
                result = episode.env["functions"][call.name](**args)
                return {"output": json_safe(result)}
            except Exception:
                logger.exception("Workplace tool execution failed: %s", call.name)
                return {"output": "Error executing tool: invalid arguments or operation failed"}

    def verify(self, session_id: str) -> dict:
        episode = self.get(session_id)
        try:
            with episode.lock:
                # Also check the live tool state against native replay. A mismatch is
                # an infrastructure error, never a zero-reward policy sample.
                replay = execute_actions_and_reset_state(episode.actions)
                if tables(replay) != tables(episode.env):
                    raise RuntimeError("Live tools and native verifier replay disagree")
                reward = float(is_correct(episode.actions, episode.task["ground_truth"], None))
            with self.lock:
                self.graded += 1
                self.solved += int(reward)
            return {
                "reward": reward,
                "valid": True,
                "state_replay_consistent": True,
                "task_id": episode.task["id"],
                "tool_calls": len(episode.actions),
            }
        finally:
            self.delete(session_id)


def create_app(dataset: Path) -> FastAPI:
    service = Workplace(dataset)
    app = FastAPI()
    app.state.workplace = service

    @app.get("/health")
    def health() -> dict:
        with service.lock:
            return {
                "tasks": len(service.tasks),
                "active": len(service.sessions),
                "graded": service.graded,
                "solved": service.solved,
            }

    @app.post("/sessions/{task_id}")
    def create(task_id: int) -> dict:
        return {"session_id": service.create(task_id)}

    @app.post("/sessions/{session_id}/tool")
    def tool(session_id: str, body: ToolCall) -> dict:
        return service.tool(session_id, body)

    @app.post("/sessions/{session_id}/verify")
    def verify(session_id: str) -> dict:
        return service.verify(session_id)

    @app.delete("/sessions/{session_id}")
    def delete(session_id: str) -> dict:
        service.delete(session_id)
        return {"deleted": True}

    return app


class Args(Tap):
    dataset: Path
    host: str = "0.0.0.0"
    port: int = 8211


def main() -> None:
    args = Args().parse_args()
    uvicorn.run(create_app(args.dataset), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

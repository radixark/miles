"""Native-tool integration checks and policy-loop failure handling."""

import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import workplace_agent
from fastapi.testclient import TestClient
from prepare_workplace import convert
from run_nemotron35_workplace import ScriptArgs, execute
from workplace_server import DOMAINS, create_app, get_tools


class NativeWorkplaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.source = Path(cls.directory.name) / "native.jsonl"
        env = get_tools(DOMAINS)
        event = env["containers"]["calendar"]._calendar_events.iloc[0]
        task = env["containers"]["project_management"]._project_tasks.iloc[0]
        plans = [
            [
                {
                    "name": "calendar_update_event",
                    "arguments": json.dumps(
                        {"event_id": event["event_id"], "field": "event_name", "new_value": "Example review"}
                    ),
                },
                {
                    "name": "calendar_update_event",
                    "arguments": json.dumps(
                        {
                            "event_id": event["event_id"],
                            "field": "duration",
                            "new_value": str(int(event["duration"]) + 5),
                        }
                    ),
                },
            ],
            [
                {
                    "name": "project_management_update_task",
                    "arguments": json.dumps(
                        {"task_id": task["task_id"], "field": "task_name", "new_value": "Example handoff"}
                    ),
                },
                {
                    "name": "project_management_update_task",
                    "arguments": json.dumps(
                        {
                            "task_id": task["task_id"],
                            "field": "list_name",
                            "new_value": "Backlog" if task["list_name"] != "Backlog" else "In Progress",
                        }
                    ),
                },
            ],
        ]
        cls.rows = [
            {
                "id": i,
                "category": "test_" + str(i),
                "ground_truth": plan,
                "responses_create_params": {
                    "input": [{"role": "user", "content": "Perform the requested workplace changes."}],
                    "tools": env["schemas"],
                    "parallel_tool_calls": False,
                    "temperature": 1.0,
                },
            }
            for i, plan in enumerate(plans)
        ]
        cls.source.write_text("".join(json.dumps(row) + "\n" for row in cls.rows))

    def test_gold_noop_isolation_and_omitted_action(self) -> None:
        with TestClient(create_app(self.source)) as client:
            for row in self.rows:
                first = client.post(f"/sessions/{row['id']}").json()["session_id"]
                second = client.post(f"/sessions/{row['id']}").json()["session_id"]
                for call in row["ground_truth"]:
                    result = client.post(f"/sessions/{first}/tool", json=call)
                    self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(client.post(f"/sessions/{first}/verify").json()["reward"], 1)
                self.assertEqual(client.post(f"/sessions/{second}/verify").json()["reward"], 0)
                third = client.post(f"/sessions/{row['id']}").json()["session_id"]
                for call in row["ground_truth"][:-1]:
                    client.post(f"/sessions/{third}/tool", json=call).raise_for_status()
                self.assertEqual(client.post(f"/sessions/{third}/verify").json()["reward"], 0)
            self.assertEqual(client.get("/health").json()["active"], 0)

    def test_bad_arguments_visible_and_verifier_failure_not_reward(self) -> None:
        with TestClient(create_app(self.source), raise_server_exceptions=False) as client:
            sid = client.post(f"/sessions/{self.rows[0]['id']}").json()["session_id"]
            body = {"name": self.rows[0]["ground_truth"][0]["name"], "arguments": "[]"}
            self.assertIn("Error executing tool", client.post(f"/sessions/{sid}/tool", json=body).json()["output"])
            with patch("workplace_server.is_correct", side_effect=RuntimeError("verifier unavailable")):
                response = client.post(f"/sessions/{sid}/verify")
            self.assertEqual(response.status_code, 500)
            self.assertEqual(client.get("/health").json()["active"], 0)
            self.assertEqual(client.post("/sessions/-1").status_code, 404)

    def test_export_has_no_gold_or_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "train.jsonl"
            receipt = convert(self.source, target)
            exported = [json.loads(line) for line in target.read_text().splitlines()]
            native = [json.loads(line) for line in self.source.read_text().splitlines()]
            self.assertEqual(receipt["tasks"], len(native))
            for row, original in zip(exported, native, strict=True):
                self.assertEqual(row["prompt"], original["responses_create_params"]["input"])
                self.assertEqual(row["metadata"]["workplace_policy"], original["responses_create_params"])
                self.assertNotIn("ground_truth", row["metadata"])
                self.assertNotIn("provenance", row["metadata"])

    def test_undeclared_tool_is_not_replayed(self) -> None:
        with TestClient(create_app(self.source)) as client:
            service = client.app.state.workplace
            session_id = service.create(self.rows[0]["id"])
            episode = service.get(session_id)
            episode.task = {**episode.task, "responses_create_params": {"tools": []}}
            response = client.post(f"/sessions/{session_id}/tool", json=self.rows[0]["ground_truth"][0])
            self.assertIn("not declared", response.json()["output"])
            self.assertEqual(episode.actions, [])
            self.assertEqual(client.post(f"/sessions/{session_id}/verify").json()["reward"], 0)

    def test_invalid_export_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "native.jsonl"
            target = Path(directory) / "miles.jsonl"
            target.write_text("existing dataset")
            rows = (
                json.loads(self.source.read_text().splitlines()[0]),
                json.loads(self.source.read_text().splitlines()[1]),
            )
            rows[1]["responses_create_params"]["unhandled_field"] = "unexpected"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Unreviewed policy field"):
                convert(source, target)
            self.assertEqual(target.read_text(), "existing dataset")
            with self.assertRaisesRegex(ValueError, "distinct"):
                convert(source, source)


class PolicyLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_turns_preserve_reasoning_and_native_reward(self) -> None:
        requests = []
        count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal count
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path.endswith("/chat/completions"):
                count += 1
                message = {"content": "done", "reasoning_content": "reasoning retained"}
                reason = "stop"
                if count == 1:
                    reason = "tool_calls"
                    message["tool_calls"] = [
                        {"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}
                    ]
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"finish_reason": reason, "message": message}],
                        "usage": {"completion_tokens": 5, "prompt_tokens": 2},
                    },
                )
            if request.url.path.endswith("/tool"):
                return httpx.Response(200, json={"output": "tool observation"})
            return httpx.Response(
                200, json={"valid": True, "state_replay_consistent": True, "reward": 1.0, "tool_calls": 1}
            )

        params = {
            "input": [{"role": "user", "content": "task"}],
            "tools": [{"type": "function", "name": "test_tool", "parameters": {}}],
        }
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await workplace_agent.policy_loop(
                client,
                "http://policy/sessions/x",
                "http://resource/sessions/y",
                params,
                {"max_tokens": 1000},
                81920,
                24,
            )
        self.assertTrue(result["workplace_episode_valid"])
        self.assertEqual(result["workplace_reward"], 1.0)
        self.assertEqual(result["workplace_turns"], 2)
        policy = [body for path, body in requests if path.endswith("/chat/completions")]
        self.assertEqual(policy[1]["messages"][1]["reasoning_content"], "reasoning retained")
        self.assertEqual(policy[1]["max_tokens"], 995)
        self.assertEqual(json.loads(policy[1]["messages"][2]["content"]), {"output": "tool observation"})

    async def test_context_retry_uses_exact_server_count(self) -> None:
        budgets = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            budgets.append(body["max_tokens"])
            if len(budgets) == 1:
                return httpx.Response(400, json={"message": "30000 tokens from the input messages"})
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await workplace_agent.completion(client, "http://policy", {"max_tokens": 40000}, 50000)
        self.assertTrue(response["ok"])
        self.assertEqual(budgets, [40000, 19992])

    async def test_abort_and_http_failure_never_become_zero_reward(self) -> None:
        params = {"input": [{"role": "user", "content": "task"}], "tools": []}
        for code in (200, 503):

            def handler(request: httpx.Request, status_code: int = code) -> httpx.Response:
                return httpx.Response(status_code, json={"choices": [{"finish_reason": "abort"}]})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                call = workplace_agent.policy_loop(
                    client, "http://policy", "http://resource", params, {"max_tokens": 1000}, 81920, 24
                )
                if code == 503:
                    with self.assertRaises(httpx.HTTPStatusError):
                        await call
                else:
                    self.assertFalse((await call)["workplace_episode_valid"])

    async def test_run_cleans_up_after_abort_and_http_failure(self) -> None:
        real_client = httpx.AsyncClient
        for code in (200, 503):
            methods = []

            def handler(request: httpx.Request, methods: list = methods, code: int = code) -> httpx.Response:
                methods.append((request.method, request.url.path))
                if request.url.path == "/sessions/1":
                    return httpx.Response(200, json={"session_id": "created"})
                if request.method == "DELETE":
                    return httpx.Response(200, json={"deleted": True})
                return httpx.Response(code, json={"choices": [{"finish_reason": "abort"}]})

            metadata = {
                "workplace_task_id": 1,
                "max_seq_len": 81920,
                "workplace_policy": {"input": [{"role": "user", "content": "task"}], "tools": []},
            }
            with patch.dict("os.environ", {"WORKPLACE_RESOURCE_URL": "http://resource"}):
                with patch(
                    "workplace_agent.httpx.AsyncClient",
                    side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
                ):
                    call = workplace_agent.run(
                        "http://policy/v1",
                        metadata["workplace_policy"]["input"],
                        {"max_tokens": 1000},
                        metadata,
                    )
                    if code == 503:
                        with self.assertRaises(httpx.HTTPStatusError):
                            await call
                    else:
                        self.assertFalse((await call)["workplace_episode_valid"])
            self.assertEqual(methods[-1], ("DELETE", "/sessions/created"))


class LauncherTests(unittest.TestCase):
    def test_external_ray_keeps_cluster_and_builds_async_job(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "MILES_SCRIPT_EXTERNAL_RAY": "1",
                    "MILES_SCRIPT_ENABLE_RAY_SUBMIT": "1",
                    "WANDB_API_KEY": "",
                    "MASTER_ADDR": "192.0.2.1",
                },
            ),
            patch("miles.utils.external_utils.command_utils.exec_command_cpu") as cpu,
            patch("miles.utils.external_utils.command_utils.exec_command_gpu", return_value="0"),
        ):
            execute(ScriptArgs(output_dir="/data/workplace run", verifier_url="http://192.0.2.1:8211"))
        commands = [call.args[0] for call in cpu.call_args_list]
        self.assertFalse(any("ray start" in command or "ray stop" in command for command in commands))
        jobs = [command for command in commands if "ray job submit" in command]
        self.assertEqual(len(jobs), 1)
        argv = shlex.split(jobs[0])
        self.assertTrue(any(token.endswith("/train_async.py") for token in argv))
        self.assertEqual(argv[argv.index("--save") + 1], "/data/workplace run/checkpoints")
        self.assertNotIn("--load", argv)
        self.assertNotIn("--colocate", argv)
        runtime = json.loads(next(token.split("=", 1)[1] for token in argv if token.startswith("--runtime-env-json=")))
        self.assertEqual(runtime["env_vars"]["WORKPLACE_RESOURCE_URL"], "http://192.0.2.1:8211")
        self.assertEqual(runtime["env_vars"]["MILES_NEMOTRONH_KEEP_MTP"], "")

    def test_unsupported_topology_and_abort_are_rejected(self) -> None:
        for kwargs in (
            {"num_nodes": 1},
            {"pause_generation_mode": "abort"},
            {"global_batch_size": 64},
            {"context_length": 1024},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ScriptArgs(**kwargs)


if __name__ == "__main__":
    unittest.main()

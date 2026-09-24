"""Offline Workplace policy-loop and async launcher checks."""

import json
import shlex
import unittest
from unittest.mock import patch

import httpx
import workplace_agent
from run_nemotron35_workplace import ScriptArgs, execute


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

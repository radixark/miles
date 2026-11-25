import asyncio
import gzip
import importlib.util
import json
import os
import threading
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from miles.rollout.polar_config import _render_template_value, resolve_polar_slime_config
from miles.rollout.polar_reward import custom_rm, reward_func
from miles.rollout.polar_rollout import (
    AsyncPolarRolloutWorker,
    PolarRolloutSchedulerError,
    _completed_trainable_session_count,
    _dump_all_trajectories,
    _resolve_max_tokens,
)


class PolarIntegrationTests(unittest.TestCase):
    def test_custom_rm_supports_single_and_batched_calls(self):
        args = SimpleNamespace(polar_reward_key="score")
        samples = [
            SimpleNamespace(reward={"score": 1.0}),
            SimpleNamespace(reward={"score": 2.0}),
        ]

        self.assertEqual(asyncio.run(custom_rm(args, samples[0])), 1.0)
        self.assertEqual(asyncio.run(custom_rm(args, samples)), [1.0, 2.0])
        self.assertEqual(
            asyncio.run(custom_rm(args, samples, request_id="test")),
            [1.0, 2.0],
        )
        self.assertEqual(
            asyncio.run(reward_func(args, samples)),
            [{"score": 1.0}, {"score": 2.0}],
        )

    def test_template_rendering_preserves_structured_values(self):
        context = {
            "sample": SimpleNamespace(
                label="reference answer",
                metadata=SimpleNamespace(task_id="task-7"),
            )
        }
        rendered = _render_template_value(
            {
                "reference_answer": "{sample.label}",
                "task": "id={sample.metadata.task_id}",
            },
            context,
        )
        self.assertEqual(rendered["reference_answer"], "reference answer")
        self.assertEqual(rendered["task"], "id=task-7")

    def test_context_parallel_token_admission_limit(self):
        args = SimpleNamespace(max_tokens_per_gpu=32768, context_parallel_size=4)
        self.assertEqual(_resolve_max_tokens(args), 131072)
        self.assertIsNone(_resolve_max_tokens(SimpleNamespace(max_tokens_per_gpu=None)))

    def test_loopback_callback_host_is_allowed_for_shared_namespace(self):
        args = SimpleNamespace(
            polar_rollout_url="http://polar.example:8080",
            polar_callback_host="127.0.0.1",
            polar_task_template={"agent": {}},
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            update_weights_interval=1,
        )
        config = resolve_polar_slime_config(args)
        self.assertEqual(config.callback_host, "127.0.0.1")

    def test_local_rollout_accepts_loopback_callback_host(self):
        args = SimpleNamespace(
            polar_rollout_url="http://127.0.0.1:8080",
            polar_callback_host="127.0.0.1",
            polar_task_template={"agent": {}},
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            update_weights_interval=1,
        )
        config = resolve_polar_slime_config(args)
        self.assertEqual(config.callback_host, "127.0.0.1")

    def test_task_id_template_is_unique_per_run(self):
        base_args = dict(
            polar_rollout_url="http://127.0.0.1:8080",
            polar_callback_host="127.0.0.1",
            polar_task_template={"agent": {}},
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            update_weights_interval=1,
        )
        with mock.patch("miles.rollout.polar_config._RUN_TASK_SALT", "run-a"):
            first = resolve_polar_slime_config(SimpleNamespace(**base_args))
        with mock.patch("miles.rollout.polar_config._RUN_TASK_SALT", "run-b"):
            second = resolve_polar_slime_config(SimpleNamespace(**base_args))

        self.assertTrue(first.task_id_template.endswith("-run-a"))
        self.assertTrue(second.task_id_template.endswith("-run-b"))

    def test_callback_endpoint_accepts_json_task_result(self):
        class FakeTaskResult:
            @classmethod
            def model_validate(cls, payload):
                return payload

        async def exercise():
            worker = object.__new__(AsyncPolarRolloutWorker)
            worker.config = SimpleNamespace(callback_host="127.0.0.1")
            worker._task_events = {}
            worker._task_results = {}

            server, server_task = await worker._start_callback_listener()
            try:
                event = asyncio.Event()
                worker._task_events["task-callback-1"] = event
                with mock.patch(
                    "miles.rollout.polar_rollout._load_task_result_type",
                    return_value=FakeTaskResult,
                ):
                    import httpx

                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            worker._callback_url,
                            json={"task_id": "task-callback-1", "status": "completed"},
                        )
                return response, event
            finally:
                server.should_exit = True
                await asyncio.wait_for(server_task, timeout=5.0)

        response, event = asyncio.run(exercise())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertTrue(event.is_set())

    def test_transient_polar_task_miss_is_retried(self):
        worker = object.__new__(AsyncPolarRolloutWorker)
        worker._running = True
        worker._metrics = {}
        worker._state_lock = threading.RLock()
        attempts = []
        emitted = []

        async def submit_attempt(client, pending):
            attempts.append(pending.group_id)
            if len(attempts) == 1:
                raise PolarRolloutSchedulerError(
                    "Task task-1 missing on Polar (404); retriable after restart/race"
                )
            return "completed"

        async def emit_completed(completed):
            emitted.append(completed)

        worker._submit_attempt = submit_attempt
        worker._emit_completed = emit_completed
        pending = SimpleNamespace(group_id=7)

        asyncio.run(worker._submit_and_collect(None, pending))
        self.assertEqual(attempts, [7, 7])
        self.assertEqual(emitted, ["completed"])

    def test_callback_race_waits_for_terminal_task_status(self):
        class FakeTaskStatus:
            @classmethod
            def model_validate(cls, payload):
                return SimpleNamespace(**payload)

        class FakeTaskResult:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeClient:
            def __init__(self):
                self.responses = [
                    {
                        "task_id": "task-race-1",
                        "status": "running",
                        "results": [],
                        "result_paths": [],
                    },
                    {
                        "task_id": "task-race-1",
                        "status": "completed",
                        "results": [],
                        "result_paths": [],
                    },
                ]

            async def get(self, url):
                del url
                return SimpleNamespace(
                    status_code=200,
                    raise_for_status=lambda: None,
                    json=lambda: self.responses.pop(0),
                )

        worker = object.__new__(AsyncPolarRolloutWorker)
        worker.config = SimpleNamespace(rollout_server_url="http://polar.example")
        worker._task_results = {}
        event = asyncio.Event()
        event.set()

        async def exercise():
            with mock.patch(
                "miles.rollout.polar_rollout._load_task_status_type",
                return_value=FakeTaskStatus,
            ), mock.patch(
                "miles.rollout.polar_rollout._load_task_result_type",
                return_value=FakeTaskResult,
            ):
                return await worker._await_task_result(
                    FakeClient(), "task-race-1", event, task_timeout=1.0
                )

        result = asyncio.run(exercise())
        self.assertEqual(result.status, "completed")

    def test_partial_group_admission_uses_trainable_sessions(self):
        task_result = SimpleNamespace(
            results=[
                SimpleNamespace(session_id="session-1", status="FAILED"),
                SimpleNamespace(session_id="session-2", status="FAILED"),
            ]
        )
        samples = [
            SimpleNamespace(
                metadata={"polar": {"session_id": "session-1"}},
                loss_mask=[1],
            ),
            SimpleNamespace(
                metadata={"polar": {"session_id": "session-2"}},
                loss_mask=[0, 1],
            ),
        ]
        self.assertEqual(_completed_trainable_session_count(task_result, samples), 2)

        worker = object.__new__(AsyncPolarRolloutWorker)
        self.assertIsNone(
            AsyncPolarRolloutWorker._task_rejection_reason(
                worker,
                SimpleNamespace(status="failed", results=task_result.results),
                ["sample-1", "sample-2"],
            )
        )

    def test_all_trajectories_are_persisted_as_compressed_jsonl(self):
        sample = SimpleNamespace(
            group_index=7,
            index=11,
            prompt=[{"role": "user", "content": "question"}],
            tokens=[1, 2, 3, 4],
            response_length=2,
            reward={"score": 0.75},
            loss_mask=[1, 0],
            rollout_log_probs=[-0.2, 0.0],
            status=SimpleNamespace(value="completed"),
            metadata={
                "polar": {
                    "session_id": "session-1",
                    "task_id": "task-1",
                    "node_id": "node-1",
                    "trace_index": 0,
                    "trace_debug": {
                        "finish_reason": "stop",
                        "response_messages": [{"role": "assistant", "content": "answer"}],
                    },
                }
            },
        )

        with tempfile.TemporaryDirectory() as dump_dir, mock.patch.dict(
            os.environ, {"MILES_POLAR_TRAJECTORY_DUMP_DIR": dump_dir}
        ):
            output = _dump_all_trajectories(3, [[sample]])
            self.assertIsNotNone(output)
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["session_id"], "session-1")
        self.assertEqual(records[0]["tokens"], [1, 2, 3, 4])
        self.assertEqual(records[0]["loss_mask"], [1, 0])
        self.assertEqual(records[0]["trainable_token_count"], 1)
        self.assertEqual(records[0]["response_messages"][0]["content"], "answer")

    @unittest.skipUnless(
        importlib.util.find_spec("sglang") is not None,
        "Miles data-source integration requires the optional sglang dependency",
    )
    def test_ceil_to_batch_size(self):
        from miles.rollout.polar_data_source import ceil_to_batch_size

        self.assertEqual(ceil_to_batch_size(0, 4), 0)
        self.assertEqual(ceil_to_batch_size(1, 4), 4)
        self.assertEqual(ceil_to_batch_size(8, 4), 8)
        with self.assertRaises(ValueError):
            ceil_to_batch_size(1, 0)


if __name__ == "__main__":
    unittest.main()

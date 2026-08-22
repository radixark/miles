import asyncio
import gzip
import importlib.util
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from miles.rollout.polar_config import _render_template_value, resolve_polar_slime_config
from miles.rollout.polar_reward import custom_rm, reward_func
from miles.rollout.polar_rollout import _dump_all_trajectories, _resolve_max_tokens


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

    def test_remote_rollout_rejects_loopback_callback_host(self):
        args = SimpleNamespace(
            polar_rollout_url="http://192.168.111.7:8080",
            polar_callback_host="127.0.0.1",
            polar_task_template={"agent": {}},
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            update_weights_interval=1,
        )
        with self.assertRaisesRegex(ValueError, "reachable from the Polar rollout server"):
            resolve_polar_slime_config(args)

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

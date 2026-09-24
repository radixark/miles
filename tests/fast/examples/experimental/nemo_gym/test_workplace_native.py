"""Native Gym integration tests; reported as skipped when Gym is not installed."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from prepare_workplace import convert

pytest.importorskip(
    "resources_servers.workplace_assistant.utils",
    reason="Put the pinned NeMo Gym checkout on PYTHONPATH (see README-workplace.md)",
)
pytest.importorskip("polars")
pytest.importorskip("pyarrow")
server = pytest.importorskip("workplace_server")
DOMAINS, create_app, get_tools = server.DOMAINS, server.create_app, server.get_tools


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

    def test_tool_exception_details_stay_in_server_logs(self) -> None:
        diagnostic = "private diagnostic sentinel"
        with TestClient(create_app(self.source)) as client:
            service = client.app.state.workplace
            sid = service.create(self.rows[0]["id"])
            call = self.rows[0]["ground_truth"][0]
            failing = Mock(side_effect=RuntimeError(diagnostic))
            with (
                patch.dict(service.get(sid).env["functions"], {call["name"]: failing}),
                self.assertLogs("workplace_server", level="ERROR") as logs,
            ):
                response = client.post(f"/sessions/{sid}/tool", json=call)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(), {"output": "Error executing tool: invalid arguments or operation failed"}
            )
            self.assertNotIn(diagnostic, response.text)
            self.assertNotIn("RuntimeError", response.text)
            self.assertIn(diagnostic, " ".join(logs.output))
            client.delete(f"/sessions/{sid}")

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

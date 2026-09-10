"""Training failures remain visible even when metrics cannot be uploaded."""

import asyncio
import json

import pytest

from examples.multi_lora.pressure_client import _record_failure
from examples.multi_lora.pressure_telemetry import JournalRun


@pytest.mark.asyncio
@pytest.mark.parametrize("reporting_fails", [False, True])
async def test_original_failure_releases_waiting_clients(tmp_path, monkeypatch, reporting_fails):
    barrier = asyncio.Barrier(2)
    peer = asyncio.create_task(barrier.wait())
    await asyncio.sleep(0)
    if reporting_fails:

        def full_disk(*args, **kwargs):
            raise OSError("disk full while reporting")

        monkeypatch.setattr("examples.multi_lora.pressure_client._write_json", full_disk)

    async def fail():
        try:
            raise ValueError("original sampling failure")
        except BaseException:
            await _record_failure(tmp_path, "lora_000", 0, barrier)
            raise

    with pytest.raises(ValueError, match="original sampling failure"):
        await asyncio.wait_for(fail(), timeout=1)
    with pytest.raises(asyncio.BrokenBarrierError):
        await asyncio.wait_for(peer, timeout=1)
    if not reporting_fails:
        record = json.loads((tmp_path / "lora_000-error.json").read_text())
        assert "ValueError: original sampling failure" in record["error"]
        assert record["completed_steps"] == 0


def test_metrics_survive_without_any_uploader(tmp_path):
    run = JournalRun(tmp_path, "lora_000", config={"clients": 121})
    run.update_summary({"model_id": "a"})
    for step in range(1, 4):
        run.log({"train/loss": 0.25}, step=step, histogram={"time/step_seconds": [2.0, 3.0]})
    run.finish()
    events = [json.loads(line) for line in (tmp_path / "lora_000-telemetry.jsonl").read_text().splitlines()]
    assert [event["step"] for event in events if event["kind"] == "log"] == [1, 2, 3]
    assert events[-1] == {"kind": "finish", "exit_code": 0}
    assert events[2]["histogram"] == {"time/step_seconds": [2.0, 3.0]}
    with pytest.raises(FileExistsError):
        JournalRun(tmp_path, "lora_000", config={})

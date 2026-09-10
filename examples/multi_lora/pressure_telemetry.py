"""Journal metrics locally; a separate process uploads each LoRA's W&B run.

Network retries, SDK initialization and shutdown never run in a training client.
To retry an interrupted upload, run this module with the same ``--root``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


class JournalRun:
    """One writer owns each append-only journal; no W&B dependency in the writer."""

    def __init__(self, directory, tag, **settings):
        self.path = Path(directory) / f"{tag}-telemetry.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x") as stream:
            stream.write(json.dumps({"kind": "init", "id": uuid.uuid4().hex[:8], **settings}, default=str) + "\n")

    def _append(self, kind, **values):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"kind": kind, **values}, allow_nan=False, default=str) + "\n")

    def update_summary(self, values):
        self._append("summary", values=values)

    def log(self, values, *, step=None, histogram=None, table=None):
        self._append("log", values=values, step=step, histogram=histogram, table=table)

    def finish(self, exit_code=0):
        self._append("finish", exit_code=exit_code)


def start_uploader(root):
    environment = dict(os.environ)
    # A client's SDK must not share a service inherited from another run.
    environment.pop("WANDB_SERVICE", None)
    with (Path(root) / "telemetry-upload.log").open("a") as stream:
        process = subprocess.Popen(
            [sys.executable, "-m", "examples.multi_lora.pressure_telemetry", "--root", str(root)],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (Path(root) / "telemetry-upload.pid").write_text(str(process.pid))
    return process


def _upload_event(wandb, run, event):
    if event["kind"] == "summary":
        run.summary.update(event["values"])
    elif event["kind"] == "log":
        values = dict(event["values"])
        if histogram := event.get("histogram"):
            values.update({key: wandb.Histogram(samples) for key, samples in histogram.items()})
        if table := event.get("table"):
            data = wandb.Table(columns=table["columns"], data=table["data"])
            values["timing/per_lora"] = data
            values["timing/per_lora_mean"] = wandb.plot.bar(
                data, "adapter", "mean_seconds", title="Mean step time by LoRA"
            )
        run.log(values, step=event.get("step"))
    elif event["kind"] == "finish":
        run.finish(exit_code=event["exit_code"])
    else:
        raise ValueError(f"unknown telemetry event: {event['kind']}")


def upload(root):
    # Keep the optional SDK, its threads and its service out of training processes.
    import wandb

    streams, runs, finished = {}, {}, set()
    while True:
        paths = sorted([*root.glob("*-telemetry.jsonl"), *(root / "clients").glob("*/*-telemetry.jsonl")])
        for path in paths:
            if path in finished:
                continue
            if path not in streams:
                stream = path.open()
                line = stream.readline()
                if not line.endswith("\n"):
                    stream.close()
                    continue
                metadata = json.loads(line)
                run = wandb.init(
                    id=metadata["id"],
                    entity=metadata["entity"],
                    project=metadata["project"],
                    group=metadata["group"],
                    name=metadata["name"],
                    config=metadata["config"],
                    dir=str(path.parent),
                    resume="allow",
                    reinit="create_new",
                    save_code=False,
                    settings=wandb.Settings(console="off", x_disable_stats=True),
                )
                receipt = path.with_name(
                    "wandb.json"
                    if path.name == "capacity-telemetry.jsonl"
                    else path.name.replace("-telemetry.jsonl", "-wandb.json")
                )
                receipt.write_text(json.dumps({"url": run.url, "id": run.id}) + "\n")
                streams[path], runs[path] = stream, run
            stream, run = streams[path], runs[path]
            offset = stream.tell()
            line = stream.readline()
            if not line.endswith("\n"):
                stream.seek(offset)
                continue
            event = json.loads(line)
            _upload_event(wandb, run, event)
            if event["kind"] == "finish":
                stream.close()
                finished.add(path)
                del streams[path], runs[path]
        if paths and finished == set(paths):
            return
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    upload(parser.parse_args().root)

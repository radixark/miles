"""Search for the largest passing DAPO client count, then train continuously.

Run on the gateway submission host. A JSON argv file configures
``run_pressure.py``; it must use a dedicated, existing Ray cluster.
Only jobs whose command includes this experiment's exact trial directory
are stopped. GPU allocations and the Ray cluster remain held.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from examples.multi_lora.pressure_cleanup import reap_trial_processes
from examples.multi_lora.pressure_client import _write_json
from examples.multi_lora.pressure_telemetry import JournalRun, start_uploader
from ray.job_submission import JobSubmissionClient


def _replace_flag(command, flag, value):
    result = list(command)
    if flag in result:
        result[result.index(flag) + 1] = str(value)
    else:
        result.extend([flag, str(value)])
    return result


def _is_oom(text):
    lower = text.lower()
    return any(marker in lower for marker in ("cuda out of memory", "cuda error: out of memory"))


class PressureTest:
    def __init__(self, args):
        self.args = args
        self.root = args.experiment_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = JobSubmissionClient(args.dashboard_address)
        self.command = json.loads(args.launcher_command_file.read_text())
        self.command = _replace_flag(self.command, "--output-dir", self.root / "runs")
        self.run = JournalRun(
            self.root,
            "capacity",
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.run_id,
            name=f"{args.run_id}-capacity",
            config=vars(args),
        )
        self.uploader = start_uploader(self.root) if args.wandb else None
        self.state = {"status": "searching", "maximum_confirmed": False, "trials": []}
        self._record()

    def _record(self):
        _write_json(self.root / "state.json", self.state)
        self.run.update_summary(self.state)
        print(json.dumps(self.state), flush=True)

    def _job(self, trial):
        marker = str(self.root / "runs" / trial)
        matches = [job for job in self.jobs.list_jobs() if job.submission_id and marker in shlex.split(job.entrypoint)]
        receipt = self.root / f"{trial}-launcher.json"
        if receipt.exists():
            started = json.loads(receipt.read_text()).get("started_at_ms", 0)
            matches = [job for job in matches if job.start_time >= started]
        live = [job for job in matches if not job.status.is_terminal()]
        assert len(live) <= 1, f"multiple live Ray jobs match trial {trial}"
        return live[0] if live else max(matches, key=lambda job: job.start_time, default=None)

    def _stop_server(self, trial):
        job = self._job(trial)
        if job is None:
            return
        if not job.status.is_terminal():
            self.jobs.stop_job(job.submission_id)
        deadline = time.monotonic() + 180
        while not self.jobs.get_job_status(job.submission_id).is_terminal():
            if time.monotonic() > deadline:
                raise TimeoutError("trial job did not terminate; refusing to overlap another gateway")
            time.sleep(2)
        if job.job_id:
            cleanup = reap_trial_processes(job.job_id, ray_address=self.args.ray_address)
            _write_json(self.root / f"{trial}-cleanup.json", {"job_id": job.job_id, "nodes": cleanup})

    def _launch(self, trial, count):
        command = _replace_flag(self.command, "--run-id", trial)
        command = _replace_flag(command, "--n-adapters", count)
        environment = dict(os.environ)
        environment.pop("WANDB_API_KEY", None)
        started = int(time.time() * 1000)
        with (self.root / f"{trial}.log").open("w") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        _write_json(
            self.root / f"{trial}-launcher.json", {"pid": process.pid, "argv": command, "started_at_ms": started}
        )

    def _server_log(self, trial):
        path = self.root / f"{trial}.log"
        return path.read_text(errors="replace") if path.exists() else ""

    def _wait_ready(self, trial):
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            job = self._job(trial)
            if job and job.status.is_terminal():
                text = self._server_log(trial)
                if _is_oom(text):
                    return False
                raise RuntimeError(f"gateway failed without confirmed CUDA OOM: {trial}; see its log")
            try:
                with urllib.request.urlopen(self.args.base_url + "/api/v1/healthz", timeout=5) as response:
                    if response.status == 200 and job and not job.status.is_terminal():
                        return True
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(5)
        raise TimeoutError(f"gateway readiness timed out: {trial}; this is not an OOM measurement")

    def _clients(self, trial, count, *, continuous=False):
        output = self.root / "clients" / trial
        output.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(Path(__file__).with_name("pressure_client.py")),
            "--base-url",
            self.args.base_url,
            "--model",
            self.args.model,
            "--dataset",
            str(self.args.dataset),
            "--clients",
            str(count),
            "--steps",
            "0" if continuous else str(self.args.trial_steps),
            "--output-dir",
            str(output),
            "--run-id",
            f"{self.args.run_id}-{trial}",
            "--checkpoint-root",
            str(self.root / "runs" / trial),
            "--wandb-project",
            self.args.wandb_project,
            "--phase",
            "continuous" if continuous else "search",
            "--checkpoint-every",
            "100" if continuous else "0",
            "--telemetry-managed",
        ]
        if self.args.wandb_entity:
            command.extend(["--wandb-entity", self.args.wandb_entity])
        with (output / "client.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        _write_json(output / "process.json", {"pid": process.pid, "clients": count})
        deadline = float("inf") if continuous else time.monotonic() + self.args.trial_timeout
        timing_steps = None
        while process.poll() is None:
            if continuous:
                summaries = [
                    json.loads(path.read_text()) for path in sorted(output.glob("lora_*-timing-summary.json"))
                ]
                observed_steps = [(entry["adapter"], entry["steps"]) for entry in summaries]
                if observed_steps and observed_steps != timing_steps:
                    self._log_timings(output, count, summaries)
                    timing_steps = observed_steps
            job = self._job(trial)
            if (job and job.status.is_terminal()) or time.monotonic() > deadline:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(5)
        result_path = output / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if process.returncode == 0 and result.get("status") == "passed":
            assert len(result["clients"]) == count
            assert all(client["steps"] >= self.args.trial_steps for client in result["clients"])
            return True
        text = (output / "client.log").read_text(errors="replace") + self._server_log(trial)
        if _is_oom(text):
            return False
        raise RuntimeError(f"clients failed without confirmed CUDA OOM: {trial}; see {output}")

    def _log_timings(self, output, count, summaries):
        assert all(entry["phase"] == "continuous" and entry["clients"] == count for entry in summaries)
        columns = [
            "adapter",
            "steps",
            "mean_seconds",
            "p50_seconds",
            "p90_seconds",
            "p95_seconds",
            "min_seconds",
            "max_seconds",
            "std_seconds",
        ]
        _write_json(output / "timing-summary.json", {"clients": count, "adapters": summaries})
        self.run.log({}, table={"columns": columns, "data": [[entry[key] for key in columns] for entry in summaries]})

    def search(self):
        trial, requested = "probe-auto", "auto"
        if not self.args.resume_auto:
            self._launch(trial, requested)
        ready = self._wait_ready(trial)
        report_path = self.root / "runs" / trial / "slot-capacity.json"
        if not report_path.exists():
            raise RuntimeError("one-slot GPU probe failed before estimating capacity")
        report = json.loads(report_path.read_text())
        self.state["probe"] = report
        self._record()
        count = report["n_slots"]
        passing, failing = set(), set()
        while True:
            success = ready and self._clients(trial, count)
            (passing if success else failing).add(count)
            self.state["trials"].append({"name": trial, "clients": count, "status": "passed" if success else "oom"})
            self.run.log({"capacity/clients": count, "capacity/passed": int(success)})
            self._record()
            self._stop_server(trial)
            if passing and max(passing) + 1 in failing:
                return max(passing)
            count = max(passing) + 1 if passing else min(failing) - 1
            if count < 1:
                raise RuntimeError("no client count passed on this topology")
            trial = f"search-n{count}"
            self._launch(trial, count)
            ready = self._wait_ready(trial)

    def run_forever(self):
        try:
            maximum = self.search()
            while maximum >= 1:
                self.state.update(
                    status="continuous_starting",
                    maximum_confirmed=True,
                    maximum_clients=maximum,
                    qualification=f"all clients passed {self.args.trial_steps} DAPO steps; N+1 OOM",
                )
                self._record()
                trial = f"continuous-n{maximum}"
                self._launch(trial, maximum)
                ready = self._wait_ready(trial)
                if ready:
                    self.state["status"] = "continuous_training"
                    self._record()
                    if self._clients(trial, maximum, continuous=True):
                        raise RuntimeError("unlimited training exited unexpectedly")
                self._stop_server(trial)
                self.state.update(maximum_confirmed=False, late_oom_clients=maximum)
                self._record()
                maximum = self._qualify_below(maximum)
        except BaseException as error:
            self.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            self._record()
            self.run.finish(exit_code=1)
            raise

    def _qualify_below(self, upper):
        for count in range(upper - 1, 0, -1):
            trial = f"late-oom-search-n{count}"
            self._launch(trial, count)
            success = self._wait_ready(trial) and self._clients(trial, count)
            self.state["trials"].append({"name": trial, "clients": count, "status": "passed" if success else "oom"})
            self.run.log({"capacity/clients": count, "capacity/passed": int(success)})
            self._record()
            self._stop_server(trial)
            if success:
                return count
        raise RuntimeError("continuous workload did not fit even one client")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--launcher-command-file", type=Path, required=True)
    parser.add_argument("--dashboard-address", required=True)
    parser.add_argument("--ray-address", required=True, help="GCS address of the existing dedicated Ray cluster")
    parser.add_argument("--base-url", default="http://127.0.0.1:10639")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-auto", action="store_true")
    parser.add_argument("--trial-steps", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=7200)
    parser.add_argument("--trial-timeout", type=float, default=21600)
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb", action="store_true", help="Upload local metrics to W&B (disabled by default)")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "miles-pressure-test"))
    PressureTest(parser.parse_args()).run_forever()

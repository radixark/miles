import os
import subprocess
import sys
import uuid

from examples.multi_lora.pressure_cleanup import _reap_on_node


def test_reaps_orphan_without_touching_another_job_or_checkout(tmp_path):
    job_id = uuid.uuid4().hex
    source = str(tmp_path / "source")
    processes = []
    try:
        for job, path in [(job_id, source), ("another-job", source), (job_id, source + "-other")]:
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    env={**os.environ, "RAY_JOB_ID": job, "PYTHONPATH": path},
                )
            )
        result = _reap_on_node(job_id, source)
        assert result["terminated_pids"] == [processes[0].pid]
        assert processes[0].poll() is not None
        assert processes[1].poll() is None
        assert processes[2].poll() is None
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()

"""Harbor trials on real cloud sandboxes, through the full rollout path.

What the sandbox smoke (``test_sandbox_golden.py``, next to this file) cannot
see, this covers: the launcher wiring delivering the Harbor environment to
rollout workers, the session server + TITO recording a real model's turns
under the strict gate, terminus-2 driving the sandbox from the trainer host,
and the reward flowing back through generate.reward_func. Rollout only
(``--debug-rollout-only``): everything Harbor-specific runs before the
optimizer step, and skipping that step lets the recipe's own model,
GLM-4.7-Flash, fit on 2 GPUs.

``HARBOR_ENV_TYPE`` picks the sandbox backend, exactly as the recipe does.
Nothing else here is backend-specific: the credential and SDK preflight is
the launcher's own ``harbor_env_vars``, so adding a backend to
``PROVIDER_CREDENTIALS`` is all it takes to run this against it. Which
combinations have actually been run is the table in
``docs/user-guide/sandbox-providers.md``.

Registered ``disabled`` because CI runners carry no sandbox credential. Run
it manually on a GPU devbox that has one:

    # on the devbox, from the repo root (2 GPUs)
    # uv, not pip: the branch carries a uv-workspace dependency pip cannot resolve
    uv pip install "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
    export HARBOR_ENV_TYPE=e2b
    export E2B_API_URL=http://<your-e2b-service> E2B_SANDBOX_URL=$E2B_API_URL
    # key at ~/.config/e2b/api_key
    # what an interactive devbox shell sets and a plain ssh session does not:
    # the Ray head's address (the launcher falls back to 127.0.0.1) and a file
    # descriptor limit Ray's worker sockets fit in (1024 kills the raylet)
    export MASTER_ADDR=$(hostname -i | awk '{print $1}')
    ulimit -n $(ulimit -Hn)
    PYTHONPATH=. python tests/e2e/agentic/test_harbor_rollout.py

Swap the extra and the backend name for another provider (``harbor[modal]``
with ``HARBOR_ENV_TYPE=modal``, ...); each provider's own credential and
endpoint variables are documented in ``miles/rollout/agentic/credentials.py``.

terminus-2 is a host-process agent: the sandboxes never call back into the
trainer, so the only network requirement is this machine -> the provider.
"""

import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U
from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS

register_cuda_ci(
    est_time=1200,
    suite="stage-c-2-gpu-h200",
    hardware=["hopper"],
    labels=["agentic"],
    disabled="CI runners hold no sandbox credential; run it manually on a GPU devbox that has one",
)

REPO = Path(__file__).resolve().parents[3]
HARBOR_EXAMPLE_DIR = REPO / "examples" / "experimental" / "harbor"
sys.path.insert(0, str(HARBOR_EXAMPLE_DIR))
from launch_common import agentic_pythonpath_dirs, agentic_train_args, harbor_env_vars  # noqa: E402

TB2_REPO = "https://github.com/laude-institute/terminal-bench-2.git"
TASKS_DIR = "/root/datasets/terminal-bench-2"  # native Harbor task dirs; cloned in prepare()
SMOKE_TASK = "fix-git"

MODEL_REPO = "zai-org/GLM-4.7-Flash"  # the recipe's model (examples/experimental/harbor/run.py)
MODEL_DIR = "/root/models/GLM-4.7-Flash"
NUM_GPUS = 2
PROMPT_DATA = "/root/datasets/harbor_tb2_smoke.jsonl"
TRIALS_DIR = "/tmp/harbor_trials_e2e"


def harbor_worker_env() -> dict[str, str]:
    """The rollout workers' Harbor environment, assembled by the launcher's own code.

    Building it is also the credential and SDK preflight: the launcher raises
    with the provider's provision hint when either is missing.
    """
    # bound each trial so the smoke stays a smoke: a looping agent would
    # otherwise run to the engine's context limit. fix-git takes terminus
    # well over 12 of its one-command turns, so the cap leaves room to solve.
    os.environ.setdefault("AGENT_TRIAL_TIMEOUT", "1200")
    os.environ.setdefault("HARBOR_AGENT_MAX_ITERATIONS", "30")
    args = SimpleNamespace(
        harbor_env_type=os.environ.get("HARBOR_ENV_TYPE", ""),
        harbor_env_kwargs=os.environ.get("HARBOR_ENV_KWARGS", ""),
        harbor_tasks_dir=TASKS_DIR,
        harbor_trials_dir=TRIALS_DIR,
        agent_model_name="model",
        agent_timeout=600,
        router_external_host="",  # terminus-2 runs on this host; no sandbox callback
        # every registered provider's key-file argument, so a new backend needs no change here
        **{spec["arg_attr"]: os.environ.get(spec["file_env_var"], "") for spec in PROVIDER_CREDENTIALS.values()},
    )
    return harbor_env_vars(args)


def probe_e2b_endpoint() -> None:
    """Fail before the model download when a configured endpoint is unreachable.

    Only a self-hosted E2B endpoint gets one: its address comes from
    configuration, so it can be wrong, and an unauthenticated request is
    enough to prove it answers.
    """
    api_url = os.environ.get("E2B_API_URL", "").strip()
    if os.environ.get("HARBOR_ENV_TYPE", "").strip().lower() != "e2b" or not api_url:
        return
    try:
        request = urllib.request.Request(f"{api_url}/nodes", headers={"X-API-Key": "probe"})
        urllib.request.urlopen(request, timeout=10).read()
    except urllib.error.HTTPError:
        pass  # a 401 still proves the control plane answers
    except OSError as e:
        sys.exit(f"E2B endpoint unreachable at {api_url} ({e}); does this machine have a route to it?")


def prepare():
    # a stale trial dir from a prior manual run must not vouch for this one
    shutil.rmtree(TRIALS_DIR, ignore_errors=True)
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    if not (Path(MODEL_DIR) / "config.json").is_file():
        U.exec_command_cpu(f"hf download {MODEL_REPO} --local-dir {MODEL_DIR}")
    if not (Path(TASKS_DIR) / SMOKE_TASK).is_dir():
        # clear any partial clone (an interrupted one leaves a non-empty dir git refuses)
        shutil.rmtree(TASKS_DIR, ignore_errors=True)
        U.exec_command_cpu(f"git clone --depth 1 {TB2_REPO} {TASKS_DIR}")
    # One prompt, run as a GRPO group of 2: the instruction text is unused by the
    # Harbor path (the task directory carries it) but must be non-empty.
    row = {
        "prompt": [{"role": "user", "content": "Recover the lost git commits (see the task directory)."}],
        "metadata": {"instance_id": SMOKE_TASK, "agent_name": "terminus-2"},
    }
    Path(PROMPT_DATA).parent.mkdir(parents=True, exist_ok=True)
    Path(PROMPT_DATA).write_text(json.dumps(row) + "\n")


def execute(worker_env: dict[str, str]):
    ckpt_args = f"--hf-checkpoint {MODEL_DIR} "
    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--num-rollout 1 "
        "--rollout-batch-size 1 "
        "--n-samples-per-prompt 2 "
        "--over-sampling-batch-size 1 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--max-seq-len 65536 "
        "--global-batch-size 2 "
    )
    # the recipe's own wiring and engine settings (run.py), so the flags tested
    # here are the flags shipped; only the scale differs
    agent_args = agentic_train_args(tito_model="glm47", session_server_workers=4)
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-decode-log-interval 1000 "
        "--sglang-reasoning-parser glm45 --sglang-tool-call-parser glm47 "
    )
    # rollout only; fsdp + megatron_model_type=None is the pair execute_train
    # accepts for skipping megatron init
    misc_args = (
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
        "--train-backend fsdp --debug-rollout-only --ci-test "
    )
    train_args = (
        f"{ckpt_args} {rollout_args} {agent_args} {sglang_args} {U.get_default_wandb_args(__file__)} {misc_args}"
    )

    extra_env_vars = {
        "PYTHONPATH": ":".join([*agentic_pythonpath_dirs(), str(REPO)]),
        **worker_env,
    }
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


def check_trials():
    """The rollout finishing is not enough: at least one Harbor trial must
    have reached its verifier (a reward, no exception)."""
    trial_dirs = sorted(Path(TRIALS_DIR).glob(f"{SMOKE_TASK}__*"))
    assert trial_dirs, f"no Harbor trial directories under {TRIALS_DIR}"
    clean = [d for d in trial_dirs if not (d / "exception.txt").exists()]
    print(f"harbor trials: {len(trial_dirs)} total, {len(clean)} without exception")
    assert clean, f"every trial under {TRIALS_DIR} ended in an exception; see the newest exception.txt"


if __name__ == "__main__":
    worker_env = harbor_worker_env()
    probe_e2b_endpoint()
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(worker_env)
    check_trials()

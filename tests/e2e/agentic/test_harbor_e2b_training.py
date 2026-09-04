"""One GRPO step with Harbor trials on real e2b sandboxes: the full training path.

What the sandbox smoke (scripts/sandbox_smoke) cannot see, this covers: the
launcher wiring delivering the Harbor environment to rollout workers, the
session server + TITO recording a real model's turns, terminus-2 driving the
sandbox from the trainer host, the reward flowing back through
generate.reward_func, and one optimizer step. Deliberately fixed to
harbor x e2b x terminus-2 x TB2 fix-git -- one combination, the one we run.

Registered ``disabled`` because it needs what CI runners do not have yet: a
network route to an E2B-compatible sandbox service and the platform key on
the machine. Until then, run it manually on a GPU devbox that has both:

    # on the devbox, from the repo root (2 GPUs)
    # uv, not pip: the branch carries a uv-workspace dependency pip cannot resolve
    uv pip install "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
    export E2B_API_URL=http://<your-e2b-service>
    export E2B_SANDBOX_URL=$E2B_API_URL
    # key at ~/.config/e2b/api_key
    python tests/e2e/agentic/test_harbor_e2b_training.py

terminus-2 is a host-process agent: the sandboxes never call back into the
trainer, so the only network requirement is this machine -> control plane.
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

register_cuda_ci(
    est_time=900,
    suite="stage-c-2-gpu-h200",
    labels=["agentic"],
    disabled="needs a network route to the sandbox service and its key on the runner; run manually on a GPU devbox that has both",
)

REPO = Path(__file__).resolve().parents[3]
HARBOR_EXAMPLE_DIR = REPO / "examples" / "experimental" / "harbor"
sys.path.insert(0, str(HARBOR_EXAMPLE_DIR))
from launch_common import agentic_pythonpath_dirs, agentic_train_args, harbor_env_vars  # noqa: E402

TB2_REPO = "https://github.com/laude-institute/terminal-bench-2.git"
TASKS_DIR = "/root/datasets/terminal-bench-2"  # native Harbor task dirs; cloned in prepare()
SMOKE_TASK = "fix-git"

MODEL_NAME = "Qwen3-0.6B"
NUM_GPUS = 2
PROMPT_DATA = "/root/datasets/harbor_tb2_smoke.jsonl"
TRIALS_DIR = "/tmp/harbor_trials_e2e"


def preflight():
    """Fail fast with instructions instead of failing every trial later."""
    api_url = os.environ.get("E2B_API_URL", "").strip()
    if not api_url:
        sys.exit("set E2B_API_URL (and E2B_SANDBOX_URL) to your E2B-compatible service; see the module docstring")
    key_file = Path(os.environ.get("E2B_API_KEY_FILE", "~/.config/e2b/api_key")).expanduser()
    # non-empty, mirroring the real check (credentials.sandbox_key_supply): an
    # empty placeholder file must fail here, not deep inside training
    file_has_key = key_file.is_file() and bool(key_file.read_text().strip())
    if not os.environ.get("E2B_API_KEY", "").strip() and not file_has_key:
        sys.exit(f"no e2b credential: set E2B_API_KEY or put a non-empty key at {key_file}")
    try:
        request = urllib.request.Request(f"{api_url}/nodes", headers={"X-API-Key": "preflight"})
        urllib.request.urlopen(request, timeout=10).read()
    except urllib.error.HTTPError:
        pass  # a 401 still proves the control plane answers
    except OSError as e:
        sys.exit(f"sandbox service unreachable at {api_url} ({e}); does this machine have a route to it?")
    try:
        import harbor  # noqa: F401
    except ImportError:
        sys.exit("harbor is not importable; see the module docstring for the install line")


def prepare():
    # a stale trial dir from a prior manual run must not vouch for this one
    shutil.rmtree(TRIALS_DIR, ignore_errors=True)
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
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


def harbor_worker_env() -> dict[str, str]:
    """The rollout workers' Harbor environment, assembled by the launcher's own code."""
    if os.environ.get("E2B_API_KEY", "").strip():
        os.environ.setdefault("AGENT_TRIAL_TIMEOUT", "1200")
    args = SimpleNamespace(
        harbor_env_type="e2b",
        harbor_env_kwargs="",
        harbor_tasks_dir=TASKS_DIR,
        harbor_trials_dir=TRIALS_DIR,
        agent_model_name="model",
        agent_timeout=600,
        router_external_host="",  # terminus-2 runs on this host; no sandbox callback
        daytona_api_key_file="",
        e2b_api_key_file=os.environ.get("E2B_API_KEY_FILE", ""),
        modal_config_file="",
    )
    return harbor_env_vars(args)


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "
    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--num-rollout 1 "
        "--rollout-batch-size 1 "
        "--n-samples-per-prompt 2 "
        "--over-sampling-batch-size 1 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--max-seq-len 8192 "
        "--global-batch-size 2 "
    )
    # the recipes' own wiring, so the flags tested here are the flags shipped
    agent_args = agentic_train_args(tito_model="qwen3", session_server_workers=4)
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    sglang_args = "--rollout-num-gpus-per-engine 2 --sglang-decode-log-interval 1000 "
    perf_args = "--use-dynamic-batch-size --max-tokens-per-gpu 32768 "
    # tito strict check off: qwen3's template re-renders history with an empty
    # think skeleton (known false positive; engine-recorded tokens train losslessly)
    ci_args = "--ci-test --ci-disable-kl-checker --ci-disable-tito-strict-checker "
    misc_args = f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate --train-backend fsdp "

    train_args = (
        f"{ckpt_args} {rollout_args} {agent_args} {optimizer_args} {grpo_args} "
        f"{sglang_args} {U.get_default_wandb_args(__file__)} {perf_args} {ci_args} {misc_args}"
    )

    extra_env_vars = {
        "PYTHONPATH": ":".join([*agentic_pythonpath_dirs(), str(REPO)]),
        **harbor_worker_env(),
    }
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


def check_trials():
    """The training job finishing is not enough: at least one Harbor trial must
    have reached its verifier (a reward, no exception)."""
    trial_dirs = sorted(Path(TRIALS_DIR).glob(f"{SMOKE_TASK}__*"))
    assert trial_dirs, f"no Harbor trial directories under {TRIALS_DIR}"
    clean = [d for d in trial_dirs if not (d / "exception.txt").exists()]
    print(f"harbor trials: {len(trial_dirs)} total, {len(clean)} without exception")
    assert clean, f"every trial under {TRIALS_DIR} ended in an exception; see the newest exception.txt"


if __name__ == "__main__":
    preflight()
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
    check_trials()

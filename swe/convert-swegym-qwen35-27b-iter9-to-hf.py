#!/usr/bin/env python3
"""Submit or run Qwen3.5-27B torch_dist -> HuggingFace conversion.

Typical use from the debug machine:
    python swe/convert-swegym-qwen35-27b-iter9-to-hf.py --submit

To run directly inside an existing miles_ycy container:
    python swe/convert-swegym-qwen35-27b-iter9-to-hf.py --run-conversion
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from textwrap import dedent, indent


SCRIPT_PATH = Path(__file__).resolve()
MILES_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_INPUT_DIR = Path(
    "/fs/nlp-intern/yangchengyi/ckpts/"
    "swegym_qwen3.5-27b-fp8_reward1_submitted_first_Qwen3.5-27B_agentic_async_debug_session_over_sample/"
    "iter_0000009"
)
DEFAULT_OUTPUT_DIR = Path(str(DEFAULT_INPUT_DIR) + "_hf")
DEFAULT_ORIGIN_HF_DIR = Path("/fs/open_plms/Qwen/Qwen3.5-27B")
DEFAULT_MEGATRON_PATH = Path("/fs/nlp-intern/yangchengyi/Megatron-LM")
DEFAULT_IMAGE = "harbor.unisound.ai/unisound/miles_ycy:latest"
DEFAULT_JOB_NAME = "ycy-qwen35-27b-iter9-to-hf"
DEFAULT_NAMESPACE = "nlp-train"
DEFAULT_MODEL_NAME = "qwen3_5"
DEFAULT_VOCAB_SIZE = 248320


def build_pythonpath(megatron_path: Path) -> str:
    entries = [str(MILES_ROOT), str(megatron_path), str(SCRIPT_PATH.parent)]
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    if old_pythonpath:
        entries.append(old_pythonpath)
    return os.pathsep.join(entries)


def validate_conversion_inputs(input_dir: Path, origin_hf_dir: Path, megatron_path: Path) -> None:
    required_files = [
        input_dir / ".metadata",
        input_dir / "common.pt",
        input_dir / "metadata.json",
        origin_hf_dir / "config.json",
        megatron_path / "megatron",
        MILES_ROOT / "tools" / "convert_torch_dist_to_hf.py",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required conversion inputs:\n" + "\n".join(missing))


def run_conversion(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    origin_hf_dir = args.origin_hf_dir.resolve()
    megatron_path = args.megatron_path.resolve()
    validate_conversion_inputs(input_dir, origin_hf_dir, megatron_path)

    sentinel = output_dir / "model.safetensors.index.json"
    if sentinel.exists() and not args.force:
        print(f"Skip conversion because {sentinel} already exists. Use --force to rewrite tensors.")
        return

    env = os.environ.copy()
    env["MEGATRON_PATH"] = str(megatron_path)
    env["PYTHONPATH"] = build_pythonpath(megatron_path)

    cmd = [
        sys.executable,
        str(MILES_ROOT / "tools" / "convert_torch_dist_to_hf.py"),
        "--model-name",
        args.model_name,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--origin-hf-dir",
        str(origin_hf_dir),
        "--vocab-size",
        str(args.vocab_size),
        "--chunk-size",
        str(args.chunk_size_gib * 1024**3),
    ]
    if args.force:
        cmd.append("--force")

    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(MILES_ROOT))


def build_job_manifest(args: argparse.Namespace) -> str:
    validate_conversion_inputs(args.input_dir.resolve(), args.origin_hf_dir.resolve(), args.megatron_path.resolve())

    inner_cmd = [
        "python3",
        str(SCRIPT_PATH),
        "--run-conversion",
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
        "--origin-hf-dir",
        str(args.origin_hf_dir),
        "--megatron-path",
        str(args.megatron_path),
        "--model-name",
        args.model_name,
        "--vocab-size",
        str(args.vocab_size),
        "--chunk-size-gib",
        str(args.chunk_size_gib),
    ]
    if args.force:
        inner_cmd.append("--force")

    pythonpath_prefix = f"{MILES_ROOT}:{args.megatron_path}"
    shell_script = dedent(
        f"""\
        set -euo pipefail
        cd {shlex.quote(str(MILES_ROOT))}
        conda deactivate 2>/dev/null || true
        export HOME=/home/yangchengyi
        export MEGATRON_PATH={shlex.quote(str(args.megatron_path))}
        export PYTHONPATH={shlex.quote(pythonpath_prefix)}:${{PYTHONPATH:-}}
        {shlex.join(inner_cmd)}
        """
    )

    indented_script = indent(shell_script.rstrip(), "                    ")
    return f"""apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: {args.job_name}
  namespace: {args.namespace}
spec:
  runPolicy:
    cleanPodPolicy: Running
  sshAuthMountPath: /home/yangchengyi/.ssh
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        metadata:
          labels:
            app: {args.job_name}
        spec:
          restartPolicy: Never
          containers:
          - name: mpi-launcher
            image: "{args.image}"
            imagePullPolicy: IfNotPresent
            lifecycle:
              postStart:
                exec:
                  command:
                  - /bin/bash
                  - -c
                  - /etc/init.d/ssh start; /usr/local/bin/init-home-symlinks.sh
            securityContext:
              privileged: false
            command: ["/bin/bash", "-lc"]
            args:
            - |
{indented_script}
            env:
            - name: http_proxy
              value: ""
            - name: https_proxy
              value: ""
            - name: HTTP_PROXY
              value: ""
            - name: HTTPS_PROXY
              value: ""
            - name: no_proxy
              value: localhost,127.0.0.1,::1
            - name: NO_PROXY
              value: localhost,127.0.0.1,::1
            - name: DATA_LINK_SRC
              value: "/fs/nlp-intern/yangchengyi"
            resources:
              requests:
                cpu: "{args.cpu_request}"
                memory: "{args.memory_request}"
              limits:
                cpu: "{args.cpu_limit}"
                memory: "{args.memory_limit}"
            volumeMounts:
            - mountPath: /fs
              name: jfs
            - mountPath: /home/yangchengyi
              name: home
            - mountPath: /dev/shm
              name: dshm
          volumes:
          - name: jfs
            hostPath:
              path: /fs
          - name: home
            hostPath:
              path: /home/yangchengyi
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 64Gi
"""


def submit_job(manifest: str) -> None:
    subprocess.run(["kubectl", "create", "-f", "-"], input=manifest, text=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="Submit the Kubernetes MPIJob with kubectl create.")
    mode.add_argument("--print-yaml", action="store_true", help="Print the Kubernetes Job manifest and exit.")
    mode.add_argument(
        "--run-conversion",
        action="store_true",
        help="Run the conversion directly in the current Python environment.",
    )

    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--origin-hf-dir", type=Path, default=DEFAULT_ORIGIN_HF_DIR)
    parser.add_argument("--megatron-path", type=Path, default=DEFAULT_MEGATRON_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--chunk-size-gib", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Pass --force to the converter if output exists.")

    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--cpu-request", default="16")
    parser.add_argument("--cpu-limit", default="64")
    parser.add_argument("--memory-request", default="400Gi")
    parser.add_argument("--memory-limit", default="900Gi")
    parser.add_argument("--ttl-seconds-after-finished", type=int, default=7 * 24 * 3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_conversion:
        run_conversion(args)
        return

    manifest = build_job_manifest(args)
    if args.submit:
        submit_job(manifest)
    else:
        print(manifest)


if __name__ == "__main__":
    main()

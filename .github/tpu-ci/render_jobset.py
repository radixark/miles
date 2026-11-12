#!/usr/bin/env python3
"""Print the JobSet (JSON, which kubectl accepts) that runs one TPU launcher's recipe on a v7x 2x2x1 CI host.

The shape follows sgl-project/sglang-torchtpu's .github/tpu-ci/jobset-tpu.yaml, whose GKE cluster, `tpu-ci`
namespace and Kueue queues this CI shares: the JobSet is created suspended with a queue label, and Kueue admits it
when a whole CI host is free. The pod runs .github/tpu-ci/smoke.sh.
"""

import argparse
import json
import re
from pathlib import Path

NAMESPACE = "tpu-ci"
SMOKE_SCRIPT = Path(__file__).with_name("smoke.sh")


def render(*, image: str, sha: str, launcher: str, recipe: str, queue: str, deadline_s: int, run_label: str) -> dict:
    pod_spec = {
        "restartPolicy": "Never",
        "nodeSelector": {
            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
            "cloud.google.com/gke-tpu-topology": "2x2x1",
        },
        # Kueue's ResourceFlavor pins the exact CI pool on admission; this keeps the pod off the developer pools
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [{"matchExpressions": [{"key": "ci-tpu-pool", "operator": "Exists"}]}]
                }
            }
        },
        "tolerations": [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}],
        "volumes": [
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}},
            # an unmounted /tmp is the node's ephemeral storage; Ray's session logs alone can fill it
            {"name": "tmp", "emptyDir": {"sizeLimit": "64Gi"}},
            # checkpoints (Qwen3-30B-A3B and GLM-4.7-Flash are ~60 GB each), datasets and the HF cache, in host RAM:
            # the node's disk is smaller than the checkpoints plus the container's own writes
            {"name": "data", "emptyDir": {"medium": "Memory", "sizeLimit": "160Gi"}},
        ],
        "containers": [
            {
                "name": "test",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-c"],
                "args": [SMOKE_SCRIPT.read_text()],
                "env": [
                    {"name": "MILES_SHA", "value": sha},
                    {"name": "LAUNCHER", "value": launcher},
                    {"name": "RECIPE", "value": recipe},
                    # gated checkpoints; optional so the pod still starts without the secret
                    {
                        "name": "HF_TOKEN",
                        "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "token", "optional": True}},
                    },
                ],
                # No ephemeral-storage and no memory limit: the CI ClusterQueues cover only these resources (Kueue
                # never admits a request for an uncovered one), and the MoE trainers keep their optimizer in host RAM.
                "resources": {
                    "requests": {"google.com/tpu": "4", "cpu": "4", "memory": "32Gi"},
                    "limits": {"google.com/tpu": "4"},
                },
                "volumeMounts": [
                    {"name": "dshm", "mountPath": "/dev/shm"},
                    {"name": "tmp", "mountPath": "/tmp"},
                    {"name": "data", "mountPath": "/data"},
                ],
                "securityContext": {"privileged": True},
            }
        ],
    }
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            # pod names add ~20 characters to this and must stay within 63
            "generateName": f"miles-{re.sub('[^a-z0-9]+', '-', launcher.lower())[:24]}-",
            "namespace": NAMESPACE,
            "labels": {
                # created suspended; Kueue releases it onto a free CI host
                "kueue.x-k8s.io/queue-name": queue,
                # lets the workflow's cleanup step find the JobSets of a cancelled run
                "miles-ci-run": run_label,
            },
        },
        "spec": {
            "ttlSecondsAfterFinished": 7200,
            "failurePolicy": {"maxRestarts": 0},
            "replicatedJobs": [
                {
                    "name": "slice",
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "backoffLimit": 0,
                            "completions": 1,
                            "parallelism": 1,
                            "activeDeadlineSeconds": deadline_s,
                            "template": {"metadata": {"labels": {"app": "miles-tpu-ci"}}, "spec": pod_spec},
                        }
                    },
                }
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--sha", required=True, help="miles commit the pod checks out")
    parser.add_argument("--launcher", required=True, help="scripts/tpu/run_<launcher>_torchtitan.py")
    parser.add_argument("--recipe", default="smoke")
    parser.add_argument("--queue", default="tpu-prgate-1h", help="Kueue LocalQueue in the tpu-ci namespace")
    parser.add_argument("--deadline-s", type=int, required=True, help="the JobSet's activeDeadlineSeconds")
    parser.add_argument("--run-label", required=True, help="value of the miles-ci-run label")
    args = parser.parse_args()
    if not (Path(__file__).parents[2] / f"scripts/tpu/run_{args.launcher}_torchtitan.py").is_file():
        parser.error(f"no launcher scripts/tpu/run_{args.launcher}_torchtitan.py")
    jobset = render(
        image=args.image,
        sha=args.sha,
        launcher=args.launcher,
        recipe=args.recipe,
        queue=args.queue,
        deadline_s=args.deadline_s,
        run_label=args.run_label,
    )
    print(json.dumps(jobset, indent=2))


if __name__ == "__main__":
    main()

# TPU CI

`pr-test-tpu.yml` runs the `scripts/tpu` launchers' `smoke` recipe on TPU v7x. It runs on sglang-torchtpu's TPU CI
cluster (`sgl-project/sglang-torchtpu`, `.github/tpu-ci/README.md`): the GKE cluster `tpu-cluster` (project
`sglang-torchtpu`, `us-central1-c`), its `tpu-ci` namespace, its Kueue PR-gate queue and its `tpu-ci@` service
account. miles adds no node pool or queue of its own.

```
pr-test-tpu.yml: plan ─▶ image (_build-tpu-ci-image.yml) ─▶ smoke × launcher (_run-ci-tpu.yml) ─▶ finish
                                                               render_jobset.py → JobSet (suspended)
                                                               Kueue admits it onto a free 2x2x1 CI host
                                                               the pod runs smoke.sh; run_jobset.sh streams its log
```

| When | Runs |
|---|---|
| PR labelled `run-ci-tpu` (each push while labelled) | `qwen3_4b` |
| Nightly, 12:00 UTC, `main` | `qwen3_4b`, `qwen3_30b_a3b`, `glm4_7_flash_4layer` |
| `gh workflow run pr-test-tpu.yml --ref <branch> -f launchers=qwen3_30b_a3b` | the listed launchers |

## One run

1. **Image.** The image is `docker/build.py`'s `tpu` variant, the one `docker-build.yml` publishes as
   `radixark/miles:dev-tpu`, built for the commit under test so that a PR changing it runs on its own.
   `_build-tpu-ci-image.yml` reads the variant's Dockerfile and build args from `build.py` and hashes them with the
   files `docker-build.yml` rebuilds `dev-tpu` for (`docker/Dockerfile.tpu`, `docker/tpu_patch/*`,
   `docker/install-kube-tools.sh`, `requirements.txt`) and `cloudbuild.yaml` into the tag `ci-<hash>` of
   `us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/miles-tpu`, next to the TPU nodes. A tag that
   exists is reused; otherwise Cloud Build builds it (`run_cloudbuild.py`, copied from sglang-torchtpu, streams the
   build log into the job). A change to miles' code alone never rebuilds the image. The registry deletes tags after
   21 days, which also bounds how stale a moving input the hash cannot see (the x86 wheels release) gets.
2. **JobSet.** `render_jobset.py` prints one single-host JobSet labelled with the queue `tpu-prgate-1h`, the queue
   sglang-torchtpu's PR jobs use: miles and sglang PRs take turns on the same hosts. That pool (`ci-tpu7x-2x2x1`)
   autoscales between 2 and 8 hosts, and its quota covers the maximum, so a job admitted while the pool is short
   makes the cluster autoscaler add a host. That takes longer than Kueue's 5-minute `waitForPodsReady`, so such a
   job is evicted and admitted once more before it starts; the runner allows for it. Checkpoints, datasets and the
   HF cache live in a RAM-backed `/data` (160 GiB; the node's disk is 100 GB), caches in `/dev/shm` and `/tmp`,
   never the container overlay.
3. **Pod.** `smoke.sh` checks out the commit under test over the image's `/root/miles`, installs it without
   dependencies, runs the launcher's `prepare`, starts `sglang` in the background, then `train`. The pod's exit code
   is `train`'s, or 1 if the engine dies first.
4. **Runner.** `run_jobset.sh` creates the JobSet, fails if Kueue made no Workload for it, waits for admission with
   whatever the 6-hour runner budget leaves after the run, re-mints the one-hour cluster token every 40 minutes,
   streams the pod log and reports the JobSet's `Completed` / `Failed` condition. The JobSet is deleted when the
   step exits, and by label when the run is cancelled.

Deadlines per launcher are in `pr-test-tpu.yml`'s `plan` step (download, both cold compiles and three steps).

## One-time setup (cluster admins)

- **Workload Identity Federation.** The provider
  `projects/180183988733/locations/global/workloadIdentityPools/github/providers/github-oidc-v2` must accept tokens
  from `radixark/miles` (its attribute condition), and `tpu-ci@sglang-torchtpu.iam.gserviceaccount.com` needs
  `roles/iam.workloadIdentityUser` for `principalSet://iam.googleapis.com/projects/180183988733/locations/global/workloadIdentityPools/github/attribute.repository/radixark/miles`.
  `tpu-ci@` already holds what sglang-torchtpu's CI uses: Cloud Build, the `sglang-torchtpu_cloudbuild` bucket,
  the `sglang-torchtpu-ci` registry and the `tpu-ci` namespace.
- **GitHub.** Create the `run-ci-tpu` label in `radixark/miles` (`tests/ci/labels.py` lists it).
- **Hugging Face.** Pods read the optional `hf-token` secret in `tpu-ci`, as sglang-torchtpu's do.

## Changing it

| To | Do |
|---|---|
| Run another launcher | Add `scripts/tpu/run_<name>_torchtitan.py` with `prepare` / `sglang` / `train` commands, and its deadline to `plan` in `pr-test-tpu.yml`. |
| Run it on PRs by default | Add it to the `pull_request` case of `plan`. |
| Rebuild the image | Change any hashed input; a Dockerfile, `tpu_patch` or `tpu` variant edit already does. |

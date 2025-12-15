## Quick Miles Smoke Test (8×GPU)

Use this when you want to sanity-check local changes while occupying an entire Hopper node (8 GPUs) without running the full production recipe.

### 1. Allocate a Hopper node

```bash
salloc --gres=gpu:8 --partition=hopper-prod --qos=high \
       --nodes=1 --time=01:00:00 --job-name=miles-smoke
```

### 2. Prepare secrets, paths, and mounts on the login node

```bash
export WANDB_API_KEY="xxx"                      # wandb API token used by run-qwen3-4B-fsdp.sh
export CONTAINER_IMAGE="docker://radixark/miles:latest"
export MILES_REPO="$(pwd)"                      # path to your local miles checkout

# Host locations for checkpoints/datasets/logs that the FSDP script expects under /root
export HF_QWEN3_4B_DIR="${MILES_REPO}/data/models/Qwen3-4B"
export DAPO_MATH_DIR="${MILES_REPO}/data/datasets/dapo-math-17k"
export AIME24_DIR="${MILES_REPO}/data/datasets/aime-2024"
export SHARED_DATA_DIR="${MILES_REPO}/data/shared_data"

export CONTAINER_MOUNTS="/fsx:/fsx,/scratch:/scratch,${MILES_REPO}:/root/miles,${HF_QWEN3_4B_DIR}:/root/Qwen3-4B,${DAPO_MATH_DIR}:/root/dapo-math-17k,${AIME24_DIR}:/root/aime-2024,${SHARED_DATA_DIR}:/root/shared_data"

# One-time setup: download checkpoints/datasets into the folders above
hf auth login
mkdir -p "${HF_QWEN3_4B_DIR}" "${DAPO_MATH_DIR}" "${AIME24_DIR}" "${SHARED_DATA_DIR}"
hf download Qwen/Qwen3-4B --local-dir "${HF_QWEN3_4B_DIR}"
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir "${DAPO_MATH_DIR}"
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir "${AIME24_DIR}"
```


### 3. Run the 8×GPU FSDP recipe inside the container

```bash
srun --gres=gpu:8 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --no-container-mount-home \
     bash -lc '
         set -euo pipefail
         cd /root/miles
         bash scripts/run-qwen3-4B-fsdp.sh
     '
```

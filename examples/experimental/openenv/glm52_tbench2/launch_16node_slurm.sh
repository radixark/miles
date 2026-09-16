#!/bin/bash
# Site adapter for run_glm5_2_744b_a40b_daytona.py: container + Ray bring-up,
# nothing else. Every experiment setting lives in the recipe; extra arguments
# pass straight through to it:
#
#   sbatch --export=ALL launch_16node_slurm.sh [--num-rollout 10 ...]
#
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=48:00:00
set -uo pipefail

: "${MILES_ROOT:?path to this miles checkout}"
: "${CONTAINER_IMAGE:?squashfs image with the miles runtime}"
: "${CONTAINER_MOUNTS:?must expose the checkouts, model/data dirs and node-local scratch}"
: "${DAYTONA_ENV_FILE:?file exporting DAYTONA_API_KEY (chmod 600, never in git)}"
: "${FABRIC_PREFIX:?leading octets of the compute-fabric IP, e.g. 10.4.}"

RECIPE=$MILES_ROOT/examples/experimental/openenv/glm52_tbench2/run_glm5_2_744b_a40b_daytona.py
RECIPE_ARGS=$(printf '%q ' "$@")
C="--container-image=$CONTAINER_IMAGE --container-mounts=$CONTAINER_MOUNTS --container-name=ray"
COMMON="export PYTHONNOUSERSITE=1 RAY_memory_monitor_refresh_ms=0 PYTHONPATH=$MILES_ROOT"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
# Compute-fabric IP: `hostname -I` ordering varies and the management subnet
# is not routable between nodes; a wrong prefix hangs silently at Ray startup.
head_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes[0]}" hostname -I | tr ' ' '\n' | grep -E "^$FABRIC_PREFIX" | head -1)
: "${head_ip:?no address matching $FABRIC_PREFIX on ${nodes[0]}}"
ngpu_total=$(( ${#nodes[@]} * 4 ))
echo "head=${nodes[0]} ($head_ip)  nodes=${#nodes[@]}  gpus=$ngpu_total"

srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=4 -w "${nodes[0]}" $C bash -c "
  $COMMON
  ray start --head --node-ip-address=$head_ip --port=6379 --num-gpus=4 --disable-usage-stats
  for t in \$(seq 1 90); do
    ray status 2>/dev/null | grep -q '/$ngpu_total\.0 GPU' && break
    echo \"waiting for ray nodes... (\$t/90)\"; sleep 10
  done
  source $DAYTONA_ENV_FILE
  export MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=$head_ip OPENENV_RUN_ID=\${OPENENV_RUN_ID:-$SLURM_JOB_ID}
  cd $MILES_ROOT
  python3 $RECIPE train --num-nodes $SLURM_JOB_NUM_NODES $RECIPE_ARGS
" &
HEAD_PID=$!

# Workers retry the join: head container extraction time varies, and a single
# `ray start` attempt can time out before the head GCS is listening.
sleep 30
for ((i=1; i<${#nodes[@]}; i++)); do
  srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=4 -w "${nodes[$i]}" $C bash -c "
    $COMMON
    until ray start --address=$head_ip:6379 --num-gpus=4 --disable-usage-stats --block; do
      echo 'worker retry ray join...'; sleep 10
    done" &
done

wait $HEAD_PID
echo "=== driver exited; stopping job ==="
scancel "$SLURM_JOB_ID"

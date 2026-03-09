#!/bin/bash
#SBATCH -D ./
#SBATCH --job-name=rocket-16nodes-rl
#SBATCH --output=output.%j.out
#SBATCH --error=error.%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=hpc-high
#SBATCH --exclusive
#SBATCH --cpus-per-gpu=16

# =============================================================================
# Standalone sbatch script – Qwen3-235B  (16 nodes, 128 GPUs)
# =============================================================================

EXEC_DATE=$(date +%Y-%m-%d_%H-%M)
EXP="${EXP:-qwen235b-standalone}"

# --------------- container / image config ---------------
IMAGE_PATH="${IMAGE_PATH:-/mnt/vast/checkpoints/jiadongguo/docker_images/miles-20260305.sqsh}"
container_mounts="/mnt/vast/checkpoints/jiadongguo/rdma:/data"

# --------------- NCCL / UCX env vars ---------------
export NCCL_DEBUG=WARN \
    UCX_IB_PCI_RELAXED_ORDERING=on \
    UCX_TLS=tcp \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NCCL_IB_PCI_RELAXED_ORDERING=1 \
    NCCL_SOCKET_IFNAME=eth0 \
    UCX_NET_DEVICES=eth0

export RUST_BACKTRACE=full
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export OMPI_MCA_COLL_HCOLL_ENABLE=0
export AZCOPY_AUTO_LOGIN_TYPE=SPN

# --------------- bucket size (env-overridable) ---------------
BUCKET_SIZE="${BUCKET_SIZE:-1}"

# --------------- srun container args ---------------
SRUN_CONTAINER_ARGS="\
    --container-image ${IMAGE_PATH} \
    --container-mounts ${container_mounts} \
    --container-writable \
    --no-container-mount-home \
"

# --------------- shared shell functions (injected into containers) ---------------
SETUP_FUNCTION='
setup_and_run() {
    local node_rank="$1"
    local nnodes="$2"
    local head_ip="$3"

    export NODE_RANK=${node_rank}
    export NNODES=${nnodes}
    export HEAD_NODE_IP=${head_ip}
    export MASTER_ADDR=${head_ip}

    echo "=== Node ${NODE_RANK}/${NNODES} started (container) ==="
    echo "Hostname : $(hostname)"
    echo "Head IP  : ${HEAD_NODE_IP}"
    echo "Timestamp: $(date)"
    echo ""

    # ---- fetch latest code ----
    cd /sgl-workspace/sglang
    git remote add jd https://github.com/JD-ETH/sglang.git 2>/dev/null || true
    git config user.name JD-ETH && git config user.email jaedon.guo@gmail.com
    git add -A && git stash
    git fetch jd --quiet
    git reset --hard jd/remote-instance-loader-miles-integration

    cd /root/miles
    git remote add lt https://github.com/Risc-lt/miles.git 2>/dev/null || true
    git config user.name JD-ETH && git config user.email jaedon.guo@gmail.com
    git add -A && git stash
    git fetch lt --quiet
    git reset --hard lt/jd/rdma-sharable-cpu-replica

    # ---- symlinks ----
    rm -rf /root/models /root/datasets /root/multinode
    ln -sf /data/models /root/models
    ln -sf /data/datasets /root/datasets
    ln -sf /data/multinode /root/multinode

    # ---- log dir (on vast mount) ----
    LOG_DIR="/data/logs/qwen235b/'"${EXEC_DATE}-${EXP}"'"
    mkdir -p "${LOG_DIR}"
    export MILES_LOG_DIR="${LOG_DIR}"

    # ---- clean stale profiler traces ----
    rm -rf /root/rdma_profiler_logs

    # ---- run test ----
    echo "Starting test ... MILES_LOG_DIR=${MILES_LOG_DIR}"
    python /root/miles/tests/test_weight_transfer_moe_multinode_qwen235b_16nodes.py \
        --multinode --mode rdma-shared --skip-validation \
        --head-node-ip ${HEAD_NODE_IP} --nnodes ${NNODES} --node-rank ${NODE_RANK} \
        --enable-nccl-nvls --released-mc-transfer-timeout --wait-after \
        --bucket-size '"${BUCKET_SIZE}"' \
        2>&1 | tee "${LOG_DIR}/node_${NODE_RANK}.log"

    # ---- head-node post-processing ----
    if [ "${NODE_RANK}" -eq 0 ]; then
        [ -d /root/rdma_profiler_logs ] && cp -r /root/rdma_profiler_logs "${LOG_DIR}/"

        find "${LOG_DIR}" -name "miles_timer_0.log" | while read logfile; do
            echo "Consolidating: ${logfile}"
            python /root/miles/consolidate_timer_log.py "${logfile}" || true
        done

        echo ""
        echo "=== Results directory structure ==="
        find "${LOG_DIR}/qwen235b-profile" -type f -name "*.log" 2>/dev/null | sort
        echo ""
        echo "Profiling complete"
    fi
}
'

# =============================================================================
# Job preamble
# =============================================================================
echo "=== SBATCH Job Started ==="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Job Name : ${SLURM_JOB_NAME}"
echo "Nodes    : ${SLURM_JOB_NUM_NODES}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Image    : ${IMAGE_PATH}"
echo "Timestamp: $(date)"
echo ""

# --------------- parse nodes ---------------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
num_nodes=${#nodes_array[@]}

echo "=== Node Allocation ==="
echo "Head node  : ${head_node}"
echo "All nodes  : ${nodes_array[*]}"
echo "Total nodes: ${num_nodes}"
echo ""

# --------------- discover head-node IP ---------------
echo "=== IP Discovery ==="
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
echo "Head node IP: ${head_node_ip}"

if [[ ! $head_node_ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Invalid IP address discovered: $head_node_ip"
    exit 1
fi
echo "Head node IP validated: ${head_node_ip}"
echo ""

# =============================================================================
# Launch head node (rank 0)
# =============================================================================
echo "=== Launching Head Node ==="
srun --nodes=1 --ntasks=1 -w "$head_node" $SRUN_CONTAINER_ARGS bash -c '
    '"$SETUP_FUNCTION"'
    setup_and_run 0 '"$num_nodes"' '"$head_node_ip"'
' &
head_pid=$!
echo "Head node launched with PID: $head_pid"

sleep 10

# =============================================================================
# Launch worker nodes (rank 1+)
# =============================================================================
echo ""
echo "=== Launching Worker Nodes ==="

worker_pids=()
if [ $num_nodes -gt 1 ]; then
    for ((i = 1; i < num_nodes; i++)); do
        worker_node=${nodes_array[$i]}
        echo "Starting worker node $i at ${worker_node}"

        srun --nodes=1 --ntasks=1 -w "$worker_node" $SRUN_CONTAINER_ARGS bash -c '
            sleep 20
            '"$SETUP_FUNCTION"'
            setup_and_run '"$i"' '"$num_nodes"' '"$head_node_ip"'
        ' &

        worker_pids+=($!)
        echo "Worker node $i launched with PID: ${worker_pids[-1]}"
        sleep 3
    done
else
    echo "Only 1 node allocated – single-node mode"
fi

# =============================================================================
# Wait + summary
# =============================================================================
echo ""
echo "=== Waiting for all nodes ==="
echo "Head PID   : $head_pid"
[ ${#worker_pids[@]} -gt 0 ] && echo "Worker PIDs: ${worker_pids[*]}"

wait $head_pid
head_exit_code=$?
[ $head_exit_code -eq 0 ] \
    && echo "Head node completed successfully" \
    || echo "Head node FAILED (exit $head_exit_code)"

worker_failures=0
if [ ${#worker_pids[@]} -gt 0 ]; then
    for idx in "${!worker_pids[@]}"; do
        wait ${worker_pids[$idx]}
        ec=$?
        if [ $ec -eq 0 ]; then
            echo "Worker $((idx+1)) completed successfully"
        else
            echo "Worker $((idx+1)) FAILED (exit $ec)"
            worker_failures=$((worker_failures + 1))
        fi
    done
fi

echo ""
echo "=== JOB COMPLETION SUMMARY ==="
echo "Timestamp : $(date)"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Head node : ${head_node} (${head_node_ip})"
echo "Total nodes: ${num_nodes}"

if [ $head_exit_code -eq 0 ] && [ ${worker_failures} -eq 0 ]; then
    echo "Status: SUCCESS – all nodes completed"
    exit 0
else
    echo "Status: FAILURE – head=$head_exit_code, worker_failures=$worker_failures"
    exit 1
fi

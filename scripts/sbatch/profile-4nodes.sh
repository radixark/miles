JOBID=5276407
EXEC_DATE=$(date +%Y-%m-%d_%H-%M)
EXP=cpu-direct-4node

srun --jobid=${JOBID} --nodes=4 --ntasks-per-node=1 --overlap bash -c "
    pid=\$(enroot list -f | grep 'pyxis_${JOBID}' -A 1 | awk '\$2 ~ /^[0-9]+\$/ {print \$2; exit}')

    if [ -z \"\$pid\" ]; then
    echo \"[ERROR] No container found on \$(hostname)\"
    exit 1
    fi

    enroot exec \"\$pid\" bash -c \"
    . /root/env.sh

    LOG_DIR='/data/logs/4node-profile/${EXEC_DATE}-${EXP}'
    mkdir -p \\\$LOG_DIR
    export MILES_LOG_DIR=\\\$LOG_DIR

    cd /sgl-workspace/sglang && \
    git fetch jd --quiet && \
    git reset --hard jd/remote-instance-loader-slime-integration
    
    cd /root/miles
    git fetch lt --quiet && git reset --hard lt/jd/rdma-sharable-cpu-replica

    # Clean stale profiler traces before run
    rm -rf /root/rdma_profiler_logs

    # Runs all 4 models (glm4, moonlight, qwen3-30b, qwen3-32b) x 3 modes (nccl, rdma, rdma-shared)
    # Results stored in: \\\$LOG_DIR/4node-profile/<model>/<mode>/
    python /root/miles/tests/test_weight_transfer_moe_multinode.py \\
        --multinode --mode rdma-shared --models glm4,moonlight,qwen3-30b, qwen3-32b\\
        --head-node-ip \\\$HEAD_NODE_IP --nnodes \\\$NNODES --node-rank \\\$NODE_RANK \\
        --enable-nccl-nvls --released-mc-transfer-timeout --wait-after --bucket-size 1 \\
        2>&1 | tee \\\$LOG_DIR/node_\\\$NODE_RANK.log

    if [ \\\$NODE_RANK -eq 0 ]; then
        # Copy profiler traces to log dir
        [ -d /root/rdma_profiler_logs ] && cp -r /root/rdma_profiler_logs \\\$LOG_DIR/

        # Consolidate all timer logs (nested under 4node-profile/<model>/<mode>/)
        find \\\$LOG_DIR -name 'miles_timer_0.log' | while read logfile; do
            echo \\\"Consolidating: \\\$logfile\\\"
            python /root/miles/consolidate_timer_log.py \\\"\\\$logfile\\\" || true
        done

        echo ''
        echo '=== Results directory structure ==='
        find \\\$LOG_DIR/4node-profile -type f -name '*.log' | sort
        echo ''
        echo 'Profiling complete'
    fi
    \"
"

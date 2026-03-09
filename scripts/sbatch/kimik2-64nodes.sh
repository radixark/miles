EXEC_DATE=$(date +%Y-%m-%d_%H-%M)
EXP=cpu-replica-rdma-direct-kimi
JOBID=6405768

srun --jobid=${JOBID} --nodes=64 --ntasks-per-node=1 --overlap bash -c "
    pid=\$(enroot list -f | grep 'pyxis_${JOBID}' -A 1 | awk '\$2 ~ /^[0-9]+\$/ {print \$2; exit}')

    if [ -z \"\$pid\" ]; then
    echo \"[ERROR] No container found on \$(hostname)\"
    exit 1
    fi

    enroot exec \"\$pid\" bash -c \"
    . /root/env.sh

    LOG_DIR='/data/logs/kimik2/${EXEC_DATE}-${EXP}'
    mkdir -p \\\$LOG_DIR
    export MILES_LOG_DIR=\\\$LOG_DIR

    cd /sgl-workspace/sglang && \
    git fetch jd --quiet && \
    git reset --hard jd/remote-instance-loader-slime-integration
    
    cd /root/miles
    git fetch lt --quiet && git reset --hard lt/jd/rdma-sharable-cpu-replica

    # Clean stale profiler traces before run
    rm -rf /root/rdma_profiler_logs

    python /root/miles/tests/test_weight_transfer_moe_multinode_kimik2_64nodes.py \\
        --multinode --mode rdma-shared \\
        --head-node-ip \\\$HEAD_NODE_IP --nnodes \\\$NNODES --node-rank \\\$NODE_RANK \\
        --enable-nccl-nvls --released-mc-transfer-timeout --wait-after --bucket-size 4 \\
        2>&1 | tee \\\$LOG_DIR/node_\\\$NODE_RANK.log

    if [ \\\$NODE_RANK -eq 0 ]; then
        [ -d /root/rdma_profiler_logs ] && cp -r /root/rdma_profiler_logs \\\$LOG_DIR/
        # Consolidate all timer logs (nested under kimik2-profile/<mode>/)
        find \\\$LOG_DIR -name 'miles_timer_0.log' | while read logfile; do
            echo \\\"Consolidating: \\\$logfile\\\"
            python /root/miles/consolidate_timer_log.py \\\"\\\$logfile\\\" || true
        done

        echo ''
        echo '=== Results directory structure ==='
        find \\\$LOG_DIR/kimik2-profile -type f -name '*.log' | sort
        echo ''
        echo 'Profiling complete'
    fi
    \"
"

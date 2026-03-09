# Example tests: multinode scenarios
# one node: H100 80G * 8

# 1 training node, 1 rollout node
# NODE 0: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer --head-node-ip h100-139-003 --node-rank 0 --nnodes 2 2>&1 | tee temp2_moe_2node.log
# NODE 1: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer --head-node-ip h100-139-003 --node-rank 1 --nnodes 2 2>&1 | tee temp2_moe_2node.log

# enable training dp = 2
# NODE 0: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer --head-node-ip h100-139-003 --node-rank 0 --nnodes 2 --train-tp 4 2>&1 | tee temp2_moe_2node.log
# NODE 1: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer --head-node-ip h100-139-003 --node-rank 1 --nnodes 2 --train-tp 4 2>&1 | tee temp2_moe_2node.log

# enable training ep = 8, dp = 2, attn-tp = 4
# NODE 0: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer --head-node-ip h100-139-003 --node-rank 0 --nnodes 2 --train-tp 4 --train-ep 8 --train-etp 1 2>&1 | tee temp2_moe_2node.log
# NODE 1: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer --head-node-ip h100-139-003 --node-rank 1 --nnodes 2 --train-tp 4 --train-ep 8 --train-etp 1 2>&1 | tee temp2_moe_2node.log

# enable training pp = 2 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer --head-node-ip h100-139-003 --node-rank 0 --nnodes 2 --train-pp 2 --train-tp 4 --train-etp 4 --decoder-last-pipeline-num-layers 14 2>&1 | tee temp2_moe_2node.log
# NODE 1: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer --head-node-ip h100-139-003 --node-rank 1 --nnodes 2 --train-pp 2 --train-tp 4 --train-etp 4 --decoder-last-pipeline-num-layers 14 2>&1 | tee temp2_moe_2node.log


# 2 training nodes, 1 rollout node
# NODE 0: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer  --head-node-ip h100-139-003 --node-rank 0 --num-train-gpus 16 --train-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_2training.log
# NODE 1: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer  --head-node-ip h100-139-003 --node-rank 1 --num-train-gpus 16 --train-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_2training.log
# NODE 2: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer  --head-node-ip h100-139-003 --node-rank 2 --num-train-gpus 16 --train-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_2training.log


# 1 training node, 2 rollout nodes
# NODE 0: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer  --head-node-ip h100-139-003 --node-rank 0 --num-rollout-gpus 16 --sglang-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_1training.log
# NODE 1: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer  --head-node-ip h100-139-003 --node-rank 1 --num-rollout-gpus 16 --sglang-tp 16 --nnodes 3  2>&1 | tee temp2_moe_3node_1training.log
# NODE 2: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer  --head-node-ip h100-139-003 --node-rank 2 --num-rollout-gpus 16 --sglang-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_1training.log

# 1 training node, 2 rollout nodes, rollout_ep = 16
# NODE 0: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer  --head-node-ip h100-139-003 --node-rank 0 --num-rollout-gpus 16 --sglang-tp 16 --sglang-ep 16 --nnodes 3 2>&1 | tee temp2_moe_3node_1training.log
# NODE 1: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer  --head-node-ip h100-139-003 --node-rank 1 --num-rollout-gpus 16 --sglang-tp 16 --sglang-ep 16 --nnodes 3  2>&1 | tee temp2_moe_3node_1training.log
# NODE 2: 
MASTER_ADDR=h100-139-003 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --mode rdma --pipelined-transfer  --head-node-ip h100-139-003 --node-rank 2 --num-rollout-gpus 16 --sglang-tp 16 --sglang-ep 16 --nnodes 3 2>&1 | tee temp2_moe_3node_1training.log



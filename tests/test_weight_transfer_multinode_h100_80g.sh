# Example tests: multinode scenarios
# one node: H100 80G * 8

# 1 training node, 1 rollout node
# NODE 0: 
MASTER_ADDR=h100-069-001 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --head-node-ip h100-069-001 --node-rank 0 --nnodes 2 2>&1 | tee temp2_moe_2node.log
# NODE 1: 
MASTER_ADDR=h100-069-001 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --head-node-ip h100-069-001 --node-rank 1 --nnodes 2 2>&1 | tee temp2_moe_2node.log

# 2 training nodes, 1 rollout node
# NODE 0: 
MASTER_ADDR=h100-069-001 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --head-node-ip h100-069-001 --node-rank 0 --num-train-gpus 16 --train-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_2training.log
# NODE 1: 
MASTER_ADDR=h100-069-001 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --head-node-ip h100-069-001 --node-rank 1 --num-train-gpus 16 --train-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_2training.log
# NODE 2: 
MASTER_ADDR=h100-069-001 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --head-node-ip h100-069-001 --node-rank 2 --num-train-gpus 16 --train-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_2training.log


# 1 training node, 2 rollout nodes
# NODE 0: 
MASTER_ADDR=h100-069-001 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --head-node-ip h100-069-001 --node-rank 0 --num-rollout-gpus 16 --sglang-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_1training.log
# NODE 1: 
MASTER_ADDR=h100-069-001 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --head-node-ip h100-069-001 --node-rank 1 --num-rollout-gpus 16 --sglang-tp 16 --nnodes 3  2>&1 | tee temp2_moe_3node_1training.log
# NODE 2: 
MASTER_ADDR=h100-069-001 python /root/miles/tests/test_weight_transfer_moe_multinode.py --multinode --head-node-ip h100-069-001 --node-rank 2 --num-rollout-gpus 16 --sglang-tp 16 --nnodes 3 2>&1 | tee temp2_moe_3node_1training.log




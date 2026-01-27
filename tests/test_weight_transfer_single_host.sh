# Example tests: at least 3 GPUs required 
# NCCL 1-> 1 base 
CUDA_VISIBLE_DEVICES="0,1" python ./tests/test_weight_transfer.py 
# RDMA 1-> 1 base
CUDA_VISIBLE_DEVICES="0,1" python ./tests/test_weight_transfer.py --mode rdma 
# RDMA 1-> 1 pipeline 
CUDA_VISIBLE_DEVICES="0,1" python ./tests/test_weight_transfer.py --mode rdma --pipelined-transfer 
# TP=2 1-> 1 
CUDA_VISIBLE_DEVICES="0,1,2" python ./tests/test_weight_transfer.py --mode rdma --pipelined-transfer --train-tp 2 --num-train-gpus 2 
# 1 -> 2 
CUDA_VISIBLE_DEVICES="0,1,2" python ./tests/test_weight_transfer.py --mode rdma --pipelined-transfer --num-rollout-gpus 2

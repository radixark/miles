# Example tests: 8 gpus scenario
# For H100 80G

# TP=2 2-> 2 
CUDA_VISIBLE_DEVICES="0,1,2,3" python ./tests/test_weight_transfer.py --mode rdma --pipelined-transfer --train-tp 2 --num-train-gpus 2 --sglang-tp 2 --num-rollout-gpus 2 2>&1 | tee qwen3_4b_test1.log
# train_tp=2, sglang_tp=4, 2 gpus -> 4 gpus
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python ./tests/test_weight_transfer.py --mode rdma --pipelined-transfer --train-tp 2 --num-train-gpus 2 --sglang-tp 4 --num-rollout-gpus 4 2>&1 | tee qwen3_4b_test2.log


#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# export SGLANG_LORA_PROFILE=1
# export SGLANG_LORA_PROFILE_INTERVAL=10
# export SGLANG_LORA_ENABLE_FUSION=1

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

LR=2e-5

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source /root/miles/scripts/models/qwen3-4B.sh

CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen3-4B
   --save /root/models/Qwen3-4B-lora-ckpt
   --save-interval 50
)

LORA_ARGS=(
   --lora-rank 64
   --lora-alpha 32
   --lora-dropout 0.0  # +fsdp
   --target-modules all-linear
)

ROLLOUT_ARGS=(
   --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 100 
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1
   # --over-sampling-batch-size 64
   # --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --global-batch-size 64
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR}
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-lora-test
   --wandb-group qwen3-4B-megatron-lora-dapo-lr${LR}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-decode-log-interval 1000
   # --sglang-enable-metrics # -fsdp
   --sglang-mem-fraction-static 0.4 # +fsdp, memory usage on H200 = 140*0.4=56GB per GPU
   --sglang-attention-backend fa3  # +fsdp
   --sglang-chunked-prefill-size 4096
)

MEGATRON_ARGS=(
   --no-offload-train
   --no-offload-rollout
   --megatron-to-hf-mode bridge
   # --offload-rollout-level kv_cache weight  # -fsdp: not supported in megatron
   # --train-backend fsdp  # -fsdp: use megatron instead
   --train-backend megatron  # +fsdp
   --attention-dropout 0.0  # +fsdp: default dropout in megatron is 0.1
   --hidden-dropout 0.0  # +fsdp: default dropout in megatron is 0.1
   --accumulate-allreduce-grads-in-fp32  # +fsdp, megatron specific
   --attention-softmax-in-fp32  # +fsdp, megatron specific
   --attention-backend flash  # +fsdp, megatron specific
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' # +fsdp, otherwise OOM
)

PERF_ARGS=(
   --gradient-checkpointing # +fsdp
   --sequence-parallel # +fsdp
   --use-dynamic-batch-size # +fsdpF
   --max-tokens-per-gpu 9216 # +fsdp, perf
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node ${NUM_GPUS}
   --colocate
   --calculate-per-token-loss # +fsdp
   --use-miles-router # +fsdp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats


RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_ALGO": "Ring",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
  }
}'


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MEGATRON_ARGS[@]}" \
   "${MISC_ARGS[@]}"


# TODO
# 1. GET /engine_metrics HTTP/1.1 404 Not Found error when enable sglang metrics
# 2. GET /engine_metrics HTTP/1.1 500 Internal Server Error 500 error
# (RolloutManager pid=211899)   File "/usr/local/lib/python3.12/dist-packages/fastapi/routing.py", line 355, in app
# (RolloutManager pid=211899)     raw_response = await run_endpoint_function(
# (RolloutManager pid=211899)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# (RolloutManager pid=211899)   File "/usr/local/lib/python3.12/dist-packages/fastapi/routing.py", line 243, in run_endpoint_function
# (RolloutManager pid=211899)     return await dependant.call(**values)
# (RolloutManager pid=211899)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# (RolloutManager pid=211899)   File "/root/miles/miles/router/router.py", line 134, in proxy
# (RolloutManager pid=211899)     worker_url = self._use_url()
# (RolloutManager pid=211899)                  ^^^^^^^^^^^^^^^
# (RolloutManager pid=211899)   File "/root/miles/miles/router/router.py", line 227, in _use_url
# (RolloutManager pid=211899)     url = min(self.worker_request_counts, key=self.worker_request_counts.get)
# (RolloutManager pid=211899)           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# (RolloutManager pid=211899) ValueError: min() iterable argument is empty

# 3. cudnn exception during forward (after rollout generation)
# work around: 
# pip list | grep cudnn
# pip uninstall the-thing-showed     
# ref: https://github.com/nvidia/megatron-lm/issues/1882
# ray::MegatronTrainRayActor.train() (pid=212283, ip=172.17.0.50, actor_id=5315cf13034d76675d7d459e02000000,                                      
# repr=<miles.backends.megatron_utils.actor.MegatronTrainRayActor object at 0x7f50c2a616d0>)                                                      
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                 
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                      
#    File "/root/miles/miles/backends/megatron_utils/actor.py", line 406, in train                                                                 
#    return self.train_actor(rollout_id, rollout_data)                                                                                           
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                           
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                      
#    File "/root/miles/miles/backends/megatron_utils/actor.py", line 464, in train_actor                                                           
#    self.compute_log_prob(                                                                                                                      
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                      
#    File "/root/miles/miles/backends/megatron_utils/actor.py", line 384, in compute_log_prob                                                      
#    return forward_only(                                                                                                                        
#             ^^^^^^^^^^^^^                                                                                                                        
#    File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 120, in decorate_context                                      
#    return func(*args, **kwargs)                                                                                                                
#             ^^^^^^^^^^^^^^^^^^^^^                                                                                                                
#    File "/root/miles/miles/backends/megatron_utils/model.py", line 577, in forward_only                                                          
#    forward_data_store += forward_backward_func(                                                                                                
#                            ^^^^^^^^^^^^^^^^^^^^^^                                                                                                
#    File "/root/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 632, in forward_backward_no_pipelining                            
#    output_tensor, num_tokens = forward_step(                                                                                                   
#                                  ^^^^^^^^^^^^^                                                                                                   
#    File "/root/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 417, in forward_step                                              
#    output_tensor, loss_func = forward_step_func(data_iterator, model)                                                                          
#                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                          
#    File "/root/miles/miles/backends/megatron_utils/model.py", line 540, in forward_step                                                          
#    output_tensor = model(                                                                                                                      
#                      ^^^^^^                                                                                                                      
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl                                   
#    return self._call_impl(*args, **kwargs)                                                                                                     
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                     
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl                                           
#    return forward_call(*args, **kwargs)                                                                                                        
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                        
#    File "/root/Megatron-LM/megatron/core/distributed/data_parallel_base.py", line 22, in forward                                                 
#    return self.module(*inputs, **kwargs)                                                                                                       
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                       
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl                                   
#    return self._call_impl(*args, **kwargs)                                                                                                     
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                     
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl                                           
#    return forward_call(*args, **kwargs)                                                                                                        
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                        
#    File "/root/Megatron-LM/megatron/core/transformer/module.py", line 456, in forward                                                            
#    outputs = self.module(*inputs, **kwargs)                                                                                                    
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                    
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl                                   
#    return self._call_impl(*args, **kwargs)                                                                                                     
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                     
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl                                           
#    return forward_call(*args, **kwargs)                                                                                                        
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                        
#    File "/root/Megatron-LM/megatron/core/models/gpt/gpt_model.py", line 481, in forward                                                          
#    hidden_states = self.decoder(                                                                                                               
#                      ^^^^^^^^^^^^^                                                                                                               
#    File "/root/Megatron-LM/megatron/core/transformer/transformer_block.py", line 586, in __call__                                                
#    return super().__call__(*args, **kwargs)                                                                                                    
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                    
#    File "/root/Megatron-LM/megatron/core/transformer/module.py", line 319, in __call__                                                           
#    return super().__call__(*args, **kwargs)                                                                                                    
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                    
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl                                   
#    return self._call_impl(*args, **kwargs)                                                                                                     
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                     
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl                                           
#    return forward_call(*args, **kwargs)                                                                                                        
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                        
#    File "/root/Megatron-LM/megatron/core/transformer/transformer_block.py", line 735, in forward                                                 
#    hidden_states, context = layer(                                                                                                             
#                               ^^^^^^                                                                                                             
#    File "/root/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 1044, in __call__                                               
#    return super().__call__(*args, **kwargs)                                                                                                    
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                    
#    File "/root/Megatron-LM/megatron/core/transformer/module.py", line 319, in __call__                                                           
#    return super().__call__(*args, **kwargs)                                                                                                    
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                    
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl                                   
#    return self._call_impl(*args, **kwargs)                                                                                                     
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                     
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl                                           
#    return forward_call(*args, **kwargs)                                                                                                        
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                        
#    File "/root/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 475, in forward                                                 
#    hidden_states, context = self._forward_attention(*args, **kwargs)                                                                           
#                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                           
#    File "/root/Megatron-LM/megatron/core/transformer/transformer_layer.py", line 549, in _forward_attention                                      
#    attention_output_with_bias = self.self_attention(                                                                                           
#                                  ^^^^^^^^^^^^^^^^^^^^                                                                                           
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl                                   
#    return self._call_impl(*args, **kwargs)                                                                                                     
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                     
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl                                           
#    return forward_call(*args, **kwargs)                                                                                                        
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                        
#    File "/root/Megatron-LM/megatron/core/transformer/attention.py", line 965, in forward                                                         
#    core_attn_out = self.core_attention(                                                                                                        
#                      ^^^^^^^^^^^^^^^^^^^^                                                                                                        
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl                                   
#    return self._call_impl(*args, **kwargs)                                                                                                     
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                     
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl                                           
#    return forward_call(*args, **kwargs)                                                                                                        
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                        
#    File "/root/Megatron-LM/megatron/core/extensions/transformer_engine.py", line 1111, in forward                                                
#    core_attn_out = super().forward(                                                                                                            
#                      ^^^^^^^^^^^^^^^^                                                                                                            
#    File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/external_utils.py", line 196, in nonrecursive_disable_wrapper                     
#    return fn(*args, **kwargs)                                                                                                                  
#             ^^^^^^^^^^^^^^^^^^^                                                                                                                  
#    File "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py", line      
# 1493, in forward                                                                                                                                
#    return self.fused_attention(                                                                                                                
#             ^^^^^^^^^^^^^^^^^^^^^                                                                                                                
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl                                   
#    return self._call_impl(*args, **kwargs)                                                                                                     
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                     
#    File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl                                           
#    return forward_call(*args, **kwargs)                                                                                                        
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                        
#    File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py", line 1044, in _fn                                                 
#    return fn(*args, **kwargs)                                                                                                                  
#             ^^^^^^^^^^^^^^^^^^^                                                                                                                  
#    File "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/backends.py", line 1916, in          
# forward                                                                                                                                         
#    output = FusedAttnFunc.apply(                                                                                                               
#             ^^^^^^^^^^^^^^^^^^^^                                                                                                               
#    File "/usr/local/lib/python3.12/dist-packages/torch/autograd/function.py", line 581, in apply                                                 
#    return super().apply(*args, **kwargs)  # type: ignore[misc]                                                                                 
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                       
#    File "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/backends.py", line 1267, in          
# forward                                                                                                                                         
#    out_, aux_ctx_tensors, *max_logit = fused_attn_fwd(                                                                                         
#                                        ^^^^^^^^^^^^^^^                                                                                         
#    File "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/cpp_extensions/fused_attn.py", line 297, in fused_attn_fwd           
#    output_tensors = tex.fused_attn_fwd(                                                                                                        
#                      ^^^^^^^^^^^^^^^^^^^                                                                                                        
# RuntimeError: /TransformerEngine/transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu:405 in function operator(): cuDNN      
# Error: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failedptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. For        
# more information, enable cuDNN error logging by setting CUDNN_LOGERR_DBG=1 and CUDNN_LOGDEST_DBG=stderr in the environment.
# Reproducibility

Reproducibility is a bedrock of scientific progress. By combining SGLang's [deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) with Megatron-LM's deterministic mode, miles can provide fully deterministic (bitwise) experiment reproducibility.

To enable deterministic training, you need to uninstall flash attention 3 via `pip uninstall flash_attn_3 -y` and set the following:

```bash
  # sglang config
  --sglang-enable-deterministic-inference
  --sglang-attention-backend flashinfer

  # megatron config
  --deterministic-mode
```

And set the following environment variables:

```bash
     "env_vars": {
        ...,
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
```

We provide a fully deterministic script for training GSM8K with Qwen2.5 0.5B.

You can initialize the training data and checkpoint with the following script:

```bash
# download
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# convert ckpt
cd miles/
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM/ python \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
   --save /root/Qwen2.5-0.5B-Instruct_torch_dist/
```

You can train with the following script:

```bash
bash script/run-qwen2.5-0.5B-reproducibility.sh
```

Wandb screenshots are recorded in this PR: [pull#370](https://github.com/THUDM/slime/pull/370).

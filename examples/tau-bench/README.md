# Tau bench 
This example shows miles training in an agentic multi-turn tool use environment. 


## Environment Setup 
Use the `zhuzilin/miles:latest` image and initialize the environment required for Search-R1:

```bash
cd /root/
git clone https://github.com/radixark/miles.git
cd miles
pip install -e . --no-deps
# for tau bench 
cd /root/
git clone https://github.com/JD-ETH/tau-bench.git
cd tau-bench
git checkout feature/litellm-retry
pip install -e . --no-deps 
```

Use the following script to generate mock data for miles training. 

```bash
cd /root/miles/examples/tau-bench
python tau1_mock.py --local_dir /root/tau-bench/
```

Initialize the Qwen2.5-3B-Instruct model needed for tool use:

```bash
# hf checkpoint
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507

# mcore checkpoint
cd /root/miles
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
    --save /root/Qwen3-4B-Instruct-2507_torch_dist
```

## Running the Script

The user simulator runs through litellm, so any litellm-supported provider works.
Select the provider/model and supply the matching key via environment variables —
no need to edit `generate_with_tau.py`. The launch script forwards these to the
ray workers.

**DeepSeek:**

```bash
export TAU_USER_MODEL_PROVIDER=deepseek
export TAU_USER_MODEL=deepseek-chat
export DEEPSEEK_API_KEY=sk-...

cd /root/miles
bash examples/tau-bench/run_qwen3_4B.sh
```

**Gemini (default):**

```bash
export TAU_USER_MODEL_PROVIDER=gemini          # optional, this is the default
export TAU_USER_MODEL=gemini-2.5-flash-lite    # optional, this is the default
export GEMINI_API_KEY=...

cd /root/miles
bash examples/tau-bench/run_qwen3_4B.sh
```

If the matching `*_API_KEY` is not set, the run fails fast at startup with a clear error.
# Multi-LoRA Tinker Gateway

> **Read the docs:** [Multi-LoRA training](https://miles.radixark.com/docs/advanced/lora#multi-lora-training).

- `run_gateway.py`: prepare Qwen3-30B-A3B and launch the gateway.
- `client.py`: check marker memorization for one client or adapter isolation across concurrent tenants.

## Layout

One 8-GPU node, disaggregated (multi-LoRA forbids `--colocate`):

- 4 training GPUs: TP2 for the dense layers, EP4 for the 128 routed experts.
- 4 sampling GPUs: two SGLang engines of 2 GPUs each, serving adapter versions by name.
- 4 adapter slots (`--multi-lora-n-adapters`), rank up to 32, covering attention
  (`linear_qkv`, `linear_proj`) and the per-expert MoE projections (`linear_fc1`, `linear_fc2`).

## Run

The gateway implements the `tinker==0.26.2` wire schema (newer SDKs renamed protobuf fields); install that exact version on the serving node and the client:

```bash
pip install "tinker==0.26.2"
```

Start the gateway:

```bash
python examples/multi_lora/run_qwen3_30b_a3b_tinker_server.py prepare   # once per node
python examples/multi_lora/run_qwen3_30b_a3b_tinker_server.py serve     # Tinker API on :10613
```

Checkpoints default to `<output_dir>/checkpoints/<run_id>`; use `--save-dir` to choose another root.

Install `tinker` on the client, then run the marker checks:

```bash
# one client: train, save for sampler, sample back the marker
python examples/multi_lora/client.py --base-model /root/models/Qwen3-30B-A3B --mode single

# four tenants training concurrently on the same prompt with different markers;
# passing means the adapters stayed isolated end to end
python examples/multi_lora/client.py --base-model /root/models/Qwen3-30B-A3B --mode multi --clients 4
```

# Multi-LoRA Tinker Gateway

Serve concurrent LoRA fine-tuning clients on one shared base model through the Tinker protocol.

`serve_tinker.py` turns a miles deployment into a [Tinker](https://tinker-docs.thinkingmachines.ai/)-compatible
training service: each client trains its own LoRA adapter in an isolated slot of a shared
Qwen3-30B-A3B base, and samples from its own saved adapter versions — all through the
official `tinker` SDK, with no miles code on the client side.

## Layout

One 8-GPU node, disaggregated (multi-LoRA forbids `--colocate`):

- 4 training GPUs: TP2 for the dense layers, EP4 for the 128 routed experts.
- 4 sampling GPUs: two SGLang engines of 2 GPUs each, serving adapter versions by name.
- 4 adapter slots (`--multi-lora-n-adapters`), rank up to 32, covering attention
  (`linear_qkv`, `linear_proj`) and the per-expert MoE projections (`linear_fc1`, `linear_fc2`).

## Run

Start the gateway (idles until clients connect):

```bash
python examples/multi_lora/run_gateway.py prepare   # once per node
python examples/multi_lora/run_gateway.py serve     # Tinker API on :10613
```

Then drive it with the smoke client, which only needs `pip install tinker`.
Each client is its own tenant: it teaches its adapter a private marker phrase,
then samples greedily and checks the completion reproduces the marker.

```bash
# one client: train, save for sampler, sample back the marker
python examples/multi_lora/client.py --base-model /root/models/Qwen3-30B-A3B --mode single

# four tenants training concurrently on the same prompt with different markers;
# passing means the adapters stayed isolated end to end
python examples/multi_lora/client.py --base-model /root/models/Qwen3-30B-A3B --mode multi --clients 4
```

Any Tinker training loop works the same way — point the SDK at the gateway:

```python
import tinker

service = tinker.ServiceClient(base_url="http://127.0.0.1:10613", api_key="my-tenant")
training = service.create_lora_training_client(base_model="/root/models/Qwen3-30B-A3B", rank=32)
```

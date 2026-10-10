# Harbor agents through Tinker

Run the normal Miles Tinker gateway, then run this recipe in a separate client
virtualenv from the Miles checkout. The cookbook renderer owns message rendering;
`miles/tinker/client` records actual sample tokens and logprobs for training.

```bash
uv venv --python 3.12 .venv-tinker
uv pip install --python .venv-tinker/bin/python -r examples/multi_lora/requirements.txt
uv pip install --python .venv-tinker/bin/python \
  "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"

export TINKER_API_KEY=tml-my-tenant
export HARBOR_ENV_TYPE=e2b
# Configure the provider credential and endpoint as in examples/experimental/harbor.
.venv-tinker/bin/python -m examples.multi_lora.harbor_tinker.run_harbor_tinker \
  gateway=http://gateway:10613 model_name=Qwen/Qwen3-4B-Instruct-2507 \
  renderer_name=qwen3_instruct tasks_dir=/data/tasks \
  group_size=4 groups_per_batch=4 max_tokens=4096 max_datum_tokens=32768
```

Each cookbook rollout group starts an adapter on an available port; each trial
gets a separate session bound to that iteration's policy. The default listener
is loopback, suitable for agents making model calls in the client process.
For agents calling from a sandbox, set `listen_host=0.0.0.0` and
`advertised_host=<address-reachable-from-sandbox>`. Permit the adapter's dynamic
ports on that trusted network. Session URLs grant sampling access; keep them
private. The gateway tenant key stays in the cookbook process.

`max_parallel_trials_per_group` bounds sandbox trials within each group. With
`groups_per_batch=4` and `max_parallel_trials_per_group=4`, up to 16 trials can
run concurrently in the synchronous recipe. `max_turns` bounds each trace;
`max_datum_tokens` must not exceed the gateway's per-Datum token cap. The adapter
rejects an oversized request before sampling; it does not discard samples or
truncate a finished trace to fit training.

The supported agents speak non-streaming OpenAI chat completions, for example
`terminus-2` and `mini-swe-agent`. Claude Code's Anthropic API and multimodal
messages are outside this adapter's scope. Function tools use the selected
cookbook renderer's tool convention. Harbor configuration and sandbox lifecycle
reuse `examples/experimental/harbor/harbor_agent_function.py`.

The trial verdict supplies the trajectory reward; cookbook computes group
advantages and training Datums. An `AgentError` aborts the group so infrastructure
failures cannot become zero-reward policy samples. Repeated prompts and samples
after length stops remain in the token trace. Edited, compacted, or re-tokenized
histories become separate Datums unless the actual token prefix still matches.
Checkpoint TTL is disabled because the Miles gateway owns checkpoint retention.

CPU regression coverage runs in the cookbook environment through
`python tests/e2e/lora/test_tinker_client.py`. It checks actual renderer output,
HTTP session isolation, trace fidelity, and cookbook's Datum masks/logprobs.

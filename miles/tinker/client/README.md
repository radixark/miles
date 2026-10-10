# Cookbook session adapter

The adapter runs in the **cookbook client environment**, alongside the rollout
strategy. The Miles gateway serves the unchanged Tinker token API. Install the
client dependencies from `examples/multi_lora/requirements.txt` in a separate
virtualenv; the cookbook and Miles training stack have different Transformers
version constraints.

```
agent messages → cookbook renderer → SamplingClient → Miles gateway
                                         ↓
                                  immutable token trace
                                         ↓
                            Trajectory → trajectory_to_data → Datum
```

Every request renders the complete messages supplied by the caller. Editing,
compacting or repeating a history is allowed. The adapter never matches messages
to past turns or substitutes old tokens into a newly rendered prompt.

`TokenTrace` copies the actual prompt tokens, generated tokens, logprobs and stop
reason before parsing the response. Parsing only constructs the OAI response.
`turns_to_trajectory` retains every recorded sample, including repeated prompts
and turns following a length stop. Cookbook's `trajectory_to_data` merges turns
only when the next prompt has the previous prompt plus output as an exact token
prefix; otherwise it creates another Datum. Context tokens have zero loss mask;
only sampled output tokens carry logprobs and training signal.

A `ChatSession` holds the current policy's public `sampling_client`; sampler
ownership, tenant authentication and model version lifetime stay with the SDK
and gateway. The OAI `model` field is a response label, not a policy selector.
`SessionServer` is a context-managed HTTP adapter. Register a session in Python,
give its unguessable URL to the agent, and read its trace after closing it.
There is no remote bind/export API or session TTL to coordinate. Treat session
URLs as credentials. Use a trusted network or TLS proxy when exposing them.

The supported surface is non-streaming, text-only `/v1/chat/completions`, one
completion per request, with function tools where the chosen renderer supports
them. Sampling defaults come from the policy and renderer; explicit stop lists
(including `[]`) override renderer stops. Per-request output limits may lower the
policy cap. The configured per-Datum limit counts model-input tokens (prompt plus
output minus one); requests exceeding that budget are rejected before sampling.
Unsupported request options fail instead of being silently ignored.

This is independent of full-model rollout TITO. The trace is training evidence,
not an authority over future prompts. A renderer may remove thinking or change
tokenization; that simply prevents two turns from sharing a Datum.

# Agentic training with Tinker capture

The Miles Tinker gateway supports agentic RL with your own sandbox or harness through the
official Tinker cookbook capture recipe. Agents call an OpenAI- or Anthropic-compatible proxy;
each model call is recorded with its exact prompt tokens, sampled tokens and log-probs, and each
episode is read back as a cookbook `Trajectory` for training. Training uses the sampled tokens
directly, with no re-tokenization of the chat transcript.

`tinker_agent` is a standalone client package with no Miles dependency. It runs against the
Miles gateway or the hosted Tinker service.

## Requirements

- A Tinker endpoint: the Miles gateway from [Multi-LoRA Tinker Gateway](../README.md) (port
  `10613`), or the hosted Tinker service.
- Python 3.12 or newer, with the client package installed. The package pins `tinker==0.26.2` and
  `tinker-cookbook`:

  ```bash
  uv pip install ./examples/multi_lora/tinker_agent
  ```

- The cookbook renderer for the served model: `qwen3` for Qwen3-30B-A3B, the checkpoint the
  gateway launcher serves; `qwen3_instruct` for Qwen3 Instruct-2507 models. See
  `tinker_cookbook.renderers.get_renderer` for all renderers.

## Example

`agent_rl.py` runs one two-turn episode through the proxy, assigns it a reward and applies one
PPO step to the captured tokens. `run_agent` stands in for the agent.

```python
import argparse
import asyncio
import tempfile

import httpx
import tinker
from tinker_agent.proxy import serve_proxy
from tinker_agent.trajectory import flush_capture, read_samples, rows_to_trajectory
from tinker_cookbook import renderers
from tinker_cookbook.capture.store import capture_to_store
from tinker_cookbook.rl.data_processing import trajectory_to_data

RUN_ID = "demo"


async def run_agent(proxy_url):
    # Stand-in for your agent: any OpenAI-compatible client with base_url=proxy_url + "/v1" works.
    messages = [{"role": "user", "content": "Reply with the number 1."}]
    async with httpx.AsyncClient(base_url=proxy_url, timeout=300) as http:
        for _ in range(2):
            response = await http.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 32})
            response.raise_for_status()
            messages += [response.json()["choices"][0]["message"], {"role": "user", "content": "Now reply with 2."}]
    return 1.0  # the episode's reward, from your verifier


async def main(base_url, base_model, renderer_name, store_url):
    service = tinker.ServiceClient(base_url=base_url, api_key="tml-demo")
    training = await service.create_lora_training_client_async(base_model=base_model, rank=8)
    saved = await (await training.save_weights_for_sampler_async(name="step-0"))
    sampler = await service.create_sampling_client_async(model_path=saved.path)
    renderer = renderers.get_renderer(renderer_name, sampler.get_tokenizer())

    async with serve_proxy(renderer, sampler, model_label=base_model, default_max_tokens=32) as port:
        reward = await run_agent(f"http://127.0.0.1:{port}/r/run/{RUN_ID}/group/0/traj/0")
    await flush_capture()

    rows = await read_samples(store_url, RUN_ID, group_idx=0, traj_idx=0)
    trajectory = rows_to_trajectory(rows, "completed")
    # With several trajectories per task, pass reward minus the group mean instead.
    datums = trajectory_to_data(trajectory, traj_advantage=reward)
    # "mask" is a client-side field; the cookbook trainer also drops it before forward_backward.
    datums = [
        tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"},
        )
        for datum in datums
    ]
    await (await training.forward_backward_async(datums, loss_fn="ppo"))
    await (await training.optim_step_async(tinker.types.AdamParams(learning_rate=1e-5)))
    print(f"trained on {len(rows)} captured calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--renderer", default="qwen3")
    args = parser.parse_args()
    with capture_to_store(RUN_ID, data_dir=tempfile.mkdtemp(prefix="tinker-capture-")) as captured:
        asyncio.run(main(args.base_url, args.base_model, args.renderer, captured.base_url))
```

Run it against the gateway:

```bash
python agent_rl.py --base-url http://<gateway-host>:10613 --base-model /root/models/Qwen3-30B-A3B
```

The [Harbor recipe](../harbor_tinker/README.md) applies this workflow to Harbor tasks in sandboxes,
trained with the cookbook's RL loop.

## Workflow

1. **Capture.** `capture_to_store(run_id, data_dir=...)` starts a local capture store and records
   every Tinker sampling call made within its scope. The whole job runs inside it.
2. **Proxy.** `serve_proxy(renderer, sampling_client, model_label=..., default_max_tokens=...)`
   serves the proxy on a free local port and yields the port. `default_max_tokens` applies to
   requests that do not set `max_tokens`.
3. **Episode URL.** Each episode uses its own base URL,
   `http://<host>:<port>/r/run/<run_id>/group/<group_idx>/traj/<traj_idx>`:
   - `run_id` matches the `capture_to_store` run.
   - `group_idx` identifies a set of episodes that are compared with each other, such as several
     attempts at one task. The `group/<group_idx>` segment is optional.
   - `traj_idx` identifies one episode.
   - OpenAI clients append `/v1` to this URL (`/v1/chat/completions`); Anthropic clients use it
     as is (`/v1/messages`).
4. **Flush.** After the episodes finish, exiting `serve_proxy` waits for in-flight calls.
   `flush_capture()` then flushes the exporter and raises if any call was not recorded, so an
   incomplete episode never reaches training.
5. **Read.** `read_samples(store_url, run_id, group_idx=..., traj_idx=...)` returns an episode's
   calls in call order. `rows_to_trajectory(rows, stop_reason)` converts them to a `Trajectory`
   without re-tokenizing; `stop_reason` records how the episode ended, for example
   `"completed"`.
6. **Train.** `trajectory_to_data(trajectory, traj_advantage)` builds the Datums.
7. **Policy update.** After `optim_step`, save new sampler weights and serve the next episodes
   through a new proxy with the updated sampling client.

## Sandboxed agents

The proxy listens on `127.0.0.1` by default. To accept connections from a sandbox, pass a
reachable `host` such as `0.0.0.0` together with an `auth_token`; the proxy refuses a
non-loopback bind without a token. The agent presents the token as its API key, as
`Authorization: Bearer <token>` (OpenAI) or `x-api-key: <token>` (Anthropic). The URL given to
the agent must use an address the sandbox can reach, not `0.0.0.0`.

## Limitations

- **Rewards.** Captured calls carry no reward. The harness computes the reward and passes it as
  the advantage.
- **Failed calls.** Failed calls are not retried and produce no row, so a trajectory contains
  only the successful calls. Retry policy belongs to the agent.
- **Context window.** A prompt that exceeds the gateway's context window fails with HTTP 400
  `prompt is too long`; the agent is expected to compact its history or end the episode. See
  [Context window](../README.md#context-window).
- **Multiple samples.** The proxy rejects `n` other than 1. Run each attempt as a separate
  episode with its own `traj_idx` in the same group.
- **Recovery.** The package provides no retries, refusal handling, request rewriting or
  partial-episode recovery.

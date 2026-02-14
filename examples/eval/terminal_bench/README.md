# Terminal Bench Eval

This folder wires Terminal Bench (TB) into Miles as an eval delegate. The run happens on the host via `harbor run` (Terminal Bench 2.0, default) or `tb run` (Terminal Bench 1.0, legacy). Metrics extraction lives in `utils/metrics.py` and command construction lives in `utils/runner.py`.

## What runs where

- Miles runs your training/eval loop inside the Docker container.
- Miles calls the TB delegate client.
- The TB delegate server (`tb_server.py`) runs `harbor run ...` or `tb run ...` on the host.
- The server reads the latest TB JSON results and returns metrics to Miles.

## 1) Get the code (host)

```bash
mkdir miles-tb
cd miles-tb
git clone https://github.com/radixark/miles.git
```

## 2) Launch the Miles container

```bash
docker run \
  -itd \
  --gpus all \
  --shm-size 32g \
  --network host \
  --ipc=host \
  --privileged \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -v /data/cache:/root/.cache \
  -v $(pwd):/shared/miles-tb \
  --name <miles container name> \
  radixark/miles:latest \
  /bin/bash
```

## 3) Inside the Miles container

```bash
docker exec -it <miles container name> /bin/bash
```

## 4) Terminal Bench environment (host)

Run on the machine that will host `tb_server.py`:

```bash
# Host machine terminal (outside Docker)
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r miles/examples/eval/terminal_bench/requirements.txt
```

Terminal Bench 2.0 (default, via harbor):

```bash
uv pip install harbor
```

Terminal Bench 1.0 (legacy, via tb CLI):

```bash
git clone https://github.com/laude-institute/terminal-bench
uv pip install terminal-bench/.
```

Notes:
- Use your local repo paths if they are not `./miles` and `./terminal-bench`.

## 5) Start the Terminal Bench server

Run on the host (same machine where `tb`/`harbor` works). Match the port in your
eval config (examples use `9051`):

```bash
python miles/examples/eval/terminal_bench/tb_server.py --host 0.0.0.0 --port 9051 
```

What it does:
- Uses `OPENAI_API_KEY=EMPTY`
- For `runner: harbor`, builds a command like:
  `harbor run -d terminal-bench@2.0 --jobs-dir <output_path> --job-name <run_id> --model openai/<model> --agent <agent> --agent-kwarg api_base=... --n-concurrent <n> ...`
- For `runner: tb`, builds a command like:
  `tb run -d terminal-bench-core==0.1.1 --output-path <output_path> --run-id <run_id> --model openai/<model> --agent <agent> --agent-kwarg api_base=... --n-concurrent <n> ...`
- Waits for completion, then returns TB metrics (`accuracy`, `n_resolved`,
  `n_unresolved`, `pass_at_k/*`, `total_input_tokens_mean/median/min/max`,
  `total_output_tokens_mean/median`) or Harbor metrics (`n_trials`, `n_errors`,
  `metrics` entries like `mean`, `reward_stats/*`, `exception_stats/*`,
  `n_input_tokens/*`, `n_output_tokens/*`).

## 6) Run the eval script (example)

If you use the provided Qwen eval launcher (`run-eval-tb-qwen.sh`), follow the steps below to run Terminal-Bench evaluation. Configure the runner via `harbor_runner.yaml` or `tb_runner.yaml`. runner_kwargs is used to pass through extra CLI arguments, new parameters can be added directly via runner_kwargs.


Then download the HuggingFace model checkpoint inside the Miles container:

```bash
huggingface-cli download open-thoughts/OpenThinker-Agent-v1 \
--local-dir /root/.cache/huggingface/OpenThinker-Agent-v1
```

After downloading, convert the HuggingFace checkpoint to Miles's torch distributed format. From the Miles root directory, run:

```bash
cd /shared/miles-tb/miles
source scripts/models/qwen3-8B.sh

export PYTHONPATH=/root/Megatron-LM:/shared/miles-tb/miles

python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/.cache/huggingface/OpenThinker-Agent-v1 \
  --save /root/.cache/huggingface/OpenThinker-Agent-v1_torch_dist
```

Finally, run the following command inside the Miles container:

```bash
bash miles/examples/eval/scripts/terminal_bench/run-eval-tb-qwen.sh 2>&1 | tee run.log
```

## 7) Common Issues

When running Miles inside a Docker container with `--network host`, Ray may encounter port conflicts due to shared networking with the host.

In some cases, this manifests as Ray failing to start or reporting Redis- or session-related errors. This can usually be resolved by explicitly assigning unused ports when starting the Ray head node, for example by setting a non-default `--port` and `--dashboard-port`.

In more severe cases, Ray job submission may fail with errors indicating that no available agent can accept jobs. This typically happens when the dashboard agent or runtime environment agent ports are also in conflict. In such situations, explicitly specifying the agent-related ports (e.g. `--dashboard-agent-listen-port`, `--dashboard-agent-grpc-port`, and `--runtime-env-agent-port`) when starting Ray can resolve the issue.

If the TB server cannot connect to the Miles server through the sglang router (`InternalServerError`), check which address is actually listening on the router port (e.g. 30005 in this example) and update the `api_base` in `harbor_runner.yaml` or `tb_runner.yaml` accordingly:

```bash
ss -lntp | grep 30005
```

You may see `Parser warnings`, `Context length exceeded`, `Command 1 should end with newline`, `Harness execution failed`, `Provider List` in `tb_server.py` logs. They are warnings from Terminal Bench and can be ignored if runs proceed normally.

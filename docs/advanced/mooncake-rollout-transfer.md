---
title: Mooncake Rollout Data Transfer
description: Configure Mooncake Store as the rollout object-store backend in Miles.
---

Miles uses Ray's object store by default to pass rollout data from rollout workers to
trainers. Mooncake Store can replace Ray for this handoff. Both `train.py` and
`train_async.py` support the Mooncake backend, and the trainer receives the same
rollout dictionary with either backend.

Rollout-data transfer and model-weight transfer are separate settings:

- `--object-store-backend` selects the rollout object store.
- `--update-weight-transfer-mode` selects the model-weight transfer path.

## Requirements

Before starting a Miles job:

- Run the same Miles revision and Mooncake version on every Ray node.
- Start `mooncake_master`, or provide a Mooncake HA endpoint, and make the endpoint
  reachable from every node. Miles connects as a client and does not manage the
  endpoint lifecycle.
- Use routable data-network addresses for Ray and Mooncake clients.
- Reserve enough host memory for `global_segment_size` and `local_buffer_size`.
- For RDMA, expose the RDMA device to the Miles environment, allow memory locking,
  and set the local device name on each node.

The current Miles CUDA 13 image includes the Mooncake structured-object APIs used by
this backend. For a custom environment, install a Mooncake package that matches the
CUDA runtime and is compatible with the Miles revision. Follow the
[Mooncake installation guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html)
for current package names and supported platforms.

## Add Mooncake to a Miles command

Add the backend selector and Store configuration to an existing Miles recipe.

Set the Mooncake endpoint first:

```bash
export MOONCAKE_MASTER_ADDR="<mooncake-master-host>:50051"
```

<Tabs>
  <Tab title="TCP">

    ```bash
    --object-store-backend mooncake \
    --mooncake-store-init-kwargs \
      "{\"protocol\":\"tcp\",\"master_server_address\":\"${MOONCAKE_MASTER_ADDR}\"}"
    ```

  </Tab>
  <Tab title="RDMA">

    Set the local RDMA device before starting Ray on each node. Device names may
    differ between nodes.

    ```bash
    export MOONCAKE_DEVICE="<local-rdma-device>"
    ```

    Then add:

    ```bash
    --object-store-backend mooncake \
    --mooncake-store-init-kwargs \
      "{\"protocol\":\"rdma\",\"master_server_address\":\"${MOONCAKE_MASTER_ADDR}\"}"
    ```

  </Tab>
</Tabs>

Use a data-network address that every client can reach. The JSON value must remain
one shell argument.

## Two-node example

The example below runs three synchronous rollout and training iterations with FSDP.
It uses one eight-GPU node for rollout and one eight-GPU node for training. Set the
variables to match your cluster.

| Node | Address variable | Role |
| --- | --- | --- |
| Head | `HEAD_IP` | Ray head, Mooncake master, rollout manager, rollout engines |
| Worker | `WORKER_IP` | Training actors and Mooncake Store segment |

Both nodes must use the same Python environment, Miles revision, and Mooncake
version. The model and dataset paths referenced by the job must be available to the
processes that use them.

### 1. Start Mooncake and Ray on the head

Activate the Miles environment first, then run:

```bash
export HEAD_IP="<head-data-network-ip>"
export MOONCAKE_LOG="${MOONCAKE_LOG:-mooncake_master.log}"

# RDMA only. Use the device attached to HEAD_IP.
# export MOONCAKE_DEVICE="<head-rdma-device>"

mooncake_master --rpc_port 50051 --metrics_port 50052 \
  >"${MOONCAKE_LOG}" 2>&1 &

ray stop --force
ray start --head \
  --node-ip-address "${HEAD_IP}" \
  --num-gpus 8 \
  --disable-usage-stats \
  --dashboard-host 0.0.0.0 \
  --dashboard-port 8265
```

### 2. Join the worker

Activate the same Miles environment on the worker, then run:

```bash
export HEAD_IP="<head-data-network-ip>"
export WORKER_IP="<worker-data-network-ip>"

# RDMA only. Use the device attached to WORKER_IP.
# export MOONCAKE_DEVICE="<worker-rdma-device>"

ray stop --force
ray start \
  --address="${HEAD_IP}:6379" \
  --node-ip-address "${WORKER_IP}" \
  --num-gpus 8 \
  --disable-usage-stats
```

Mooncake uses each Ray node IP as `local_hostname` by default. Starting Ray with the
data-network addresses above therefore keeps both systems on the same network.

### 3. Select TCP or RDMA

On the head, choose one Store configuration. The 2 GiB values are suitable for this
small example; production jobs should size them for live rollout data, replicas, and
concurrent transfers.

```bash
export MOONCAKE_MASTER_ADDR="${HEAD_IP}:50051"
```

<Tabs>
  <Tab title="TCP">

    ```bash
    export MOONCAKE_STORE_INIT_KWARGS="{\"protocol\":\"tcp\",\"master_server_address\":\"${MOONCAKE_MASTER_ADDR}\",\"global_segment_size\":\"2gb\",\"local_buffer_size\":\"2gb\",\"chunk_bytes\":67108864}"
    ```

  </Tab>
  <Tab title="RDMA">

    ```bash
    export MOONCAKE_STORE_INIT_KWARGS="{\"protocol\":\"rdma\",\"master_server_address\":\"${MOONCAKE_MASTER_ADDR}\",\"global_segment_size\":\"2gb\",\"local_buffer_size\":\"2gb\",\"chunk_bytes\":67108864}"
    ```

  </Tab>
</Tabs>

### 4. Submit training from the head

Set the repository, model, and dataset paths before submitting the job:

```bash
export MILES_HOME="<path-to-miles>"
export MODEL_PATH="<path-to-Qwen2.5-0.5B-Instruct>"
export DATASET_PATH="<path-to-gsm8k-train.parquet>"
export RAY_DASHBOARD_ADDR="${RAY_DASHBOARD_ADDR:-http://127.0.0.1:8265}"

RUNTIME_ENV_JSON='{
  "env_vars": {
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1"
  }
}'

ray job submit --address="${RAY_DASHBOARD_ADDR}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${MILES_HOME}/train.py" \
  --train-backend fsdp \
  --hf-checkpoint "${MODEL_PATH}" \
  --prompt-data "${DATASET_PATH}" \
  --input-key messages \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type math \
  --num-rollout 3 \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 2 \
  --rollout-max-response-len 64 \
  --rollout-temperature 0.8 \
  --global-batch-size 16 \
  --advantage-estimator grpo \
  --eps-clip 0.2 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus 8 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 8 \
  --sglang-mem-fraction-static 0.7 \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --pin-rollout-manager-to-head \
  --update-weight-transfer-mode broadcast \
  --object-store-backend mooncake \
  --mooncake-store-init-kwargs "${MOONCAKE_STORE_INIT_KWARGS}" \
  --wandb-mode disabled
```

For fully asynchronous training, keep the same Mooncake options and use the normal
Miles async entrypoint and flags:

```diff
- python3 "${MILES_HOME}/train.py" ...
+ python3 "${MILES_HOME}/train_async.py" ...
+   --fully-async
```

See [Fully Async Rollout](/user-guide/fully-async) for the remaining async settings.

## Configuration reference

`--mooncake-store-init-kwargs` accepts the following fields:

| Field | Default source | Description |
| --- | --- | --- |
| `master_server_address` | `MOONCAKE_MASTER` | Mooncake master address or HA endpoint. |
| `local_hostname` | `MOONCAKE_LOCAL_HOSTNAME`, then Ray node IP | Data-network address advertised by this client. |
| `metadata_server` | `MOONCAKE_TE_META_DATA_SERVER`, then `P2PHANDSHAKE` | Transfer Engine metadata endpoint. |
| `protocol` | `MOONCAKE_PROTOCOL`, then `rdma` | Transfer protocol, normally `tcp` or `rdma`. |
| `device_name` | `MOONCAKE_DEVICE` | Local RDMA device. Prefer the environment variable when device names differ by node. |
| `global_segment_size` | `MOONCAKE_GLOBAL_SEGMENT_SIZE`, then `8gb` | Store capacity contributed by an eligible client. |
| `local_buffer_size` | `MOONCAKE_LOCAL_BUFFER_SIZE`, then `32gb` | Local transfer and staging capacity. |
| `chunk_bytes` | Mooncake default | Structured-object PUT chunk size. |

Values supplied in `--mooncake-store-init-kwargs` take precedence over environment
variables. Use data-network addresses in multi-node jobs; loopback addresses work
only when all clients run on one node.

Mooncake stores one replica by default. Request additional replicas with:

```bash
--mooncake-replica-num 2
```

Additional replicas require additional Store capacity.

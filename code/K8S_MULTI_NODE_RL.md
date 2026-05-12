# K8s 多机 RL：`multi_node.yaml` 与 `k8s_init.sh` / `ray_init.sh` 搭配说明

本文说明如何用 Kubeflow MPIJob 占位 Pod，再在宿主机上用脚本批量初始化环境与拉起 Ray 集群。

## 三者分别做什么

| 组件 | 运行位置 | 作用 |
|------|----------|------|
| [multi_node.yaml](multi_node.yaml) | `kubectl apply` 提交到集群 | 创建 Kubeflow **MPIJob**：1 个 Launcher Pod + `Worker.replicas` 个 GPU Worker（当前多为 `sleep 28d`，便于 `kubectl exec`）；挂载 `/fs`、家目录、`DATA_LINK_SRC` 等 |
| [k8s_init.sh](k8s_init.sh) | **集群外**（需 `kubectl`） | 按 Pod 名前缀找到 **Running** 的 Worker，对**每一个**执行 `STARTUP_COMMANDS`（默认可改为你的 `pip install`）。用于多机**相同环境**的一次性初始化 |
| [ray_init.sh](ray_init.sh) | **集群外** | 按 Worker Pod 前缀：`{prefix}-0` 上 `ray start --head`，其余 GPU Worker（及可选 CPU Worker）`ray start --address=head_ip:6379`；并按 `ROLLOUT_NODE_COUNT` 等为节点打上不同的 `--resources` JSON（actor / rollout / agent） |

## 推荐执行顺序

1. **提交任务**：`kubectl apply -f multi_node.yaml`（namespace 与 YAML 中一致，例如 `nlp-train`）。
2. **确认 Pod 名**：`kubectl get pods -n <namespace>`，核对 Worker 是否为 `<job-name>-worker-0` … 若实际名称带额外后缀，需调整 `GPU_WORKER_PREFIX`。
3. **环境初始化**：编辑 [k8s_init.sh](k8s_init.sh) 中的 `STARTUP_COMMANDS`，执行 `./k8s_init.sh`。可选用：`NAMESPACE`、`PARALLEL=1`、`MAX_PARALLEL` 等。
4. **拉起 Ray**：设置与集群一致的 `NAMESPACE`、`GPU_WORKER_PREFIX`（见下），执行 `./ray_init.sh`。
5. **启动 RL**：`kubectl exec` 进入 Launcher 或 `worker-0`（Ray head），运行训练入口；按 miles 要求连接 Ray（例如 `ray.init` / Ray Client）。

## `ray_init.sh` 环境变量（与 [multi_node.yaml](multi_node.yaml) 对齐）

与模板 `metadata.name: ycy-miles-test-massnode-job`、`namespace: nlp-train` 对应时，脚本默认已设为：

- `NAMESPACE=nlp-train`
- `GPU_WORKER_PREFIX=ycy-miles-test-massnode-job-worker-`

若你使用其它 MPIJob 名称或 namespace，请在运行前导出覆盖，例如：

```bash
export NAMESPACE=nlp-train
export GPU_WORKER_PREFIX=ycy-miles-test-massnode-job-worker-
./ray_init.sh
```

沿用其它同事的旧模板（示例：`nlp` + `chenweikai-slime-massnode-job-worker-`）时，同样在运行 `./ray_init.sh` 前用 `export` 覆盖 `NAMESPACE` 与 `GPU_WORKER_PREFIX` 即可。

另有单独的 **CPU Worker** MPIJob 时，设置 `CPU_WORKER_PREFIX`；仅 GPU Worker 时可以不设，脚本会跳过 CPU 分支。

### 可选补丁与依赖（Pod 内路径）

以下变量可在宿主机导出，指向 **Pod 内可见**的路径（通常来自挂载的 `/fs`）。若文件或目录不存在，[ray_init.sh](ray_init.sh) 会跳过对应步骤并打日志，而不是失败退出：

- `SGLANG_PATCH_FILE`：sglang `git apply` 用的 patch
- `SWE_AGENT_PATCH_FILE`：SWE-agent patch
- `MINISWE_AGENT_SRC`：`pip install -e` 的 mini-swe-agent 路径

## `ROLLOUT_NODE_COUNT` 与少量 Worker

例如 **4 个 GPU Worker**：Ray head 为 `-0`，非 head 共 **3** 台。脚本会把「尾部」若干台标为 `rollout_node`，前面的非 head 为 `actor_node`。若默认 `ROLLOUT_NODE_COUNT` 大于非 head 数量，会被限制为非 head 数量；若你希望「部分 GPU 只做 actor」，把 `ROLLOUT_NODE_COUNT` 调小并理解脚本中对非 head Pod 的排序与分段逻辑（见 `ray_init.sh` 内注释与实现）。

## Launcher 与 Ray

- `ray_init.sh` **只操作匹配前缀的 Worker Pod**，不会在 Launcher 容器里启动 Ray。
- Launcher 常用作中控：`kubectl exec` 进去再发起任务；若栈以 Ray 为中心，通常 **head 在 `worker-0`**，训练入口在 head 或其它已配置客户端的 Pod。

## 小结

- YAML：**占位 Pod + 挂载**。
- `k8s_init.sh`：**各 Worker 同质化安装**。
- `ray_init.sh`：**跨 Pod 组成 Ray 并打资源标签**。

顺序：**apply → Running → k8s_init → ray_init → exec 启动 RL**。

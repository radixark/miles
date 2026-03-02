# CUDA IPC `_share_cuda_()` 错误调查报告

## 错误现象

运行 `scripts/small-next.sh`（Qwen3-Next 模型，`--colocate --actor-num-gpus-per-node 8`）时，
在**第一次** `update_weights()` 调用（`train.py:27`，训练开始前）就报错：

```
torch.AcceleratorError: CUDA error: invalid argument
  at storage._untyped_storage._share_cuda_()
```

错误发生在 `update_weight_from_tensor.py` 的 `_send_to_colocated_engine()` 中，
具体在 `MultiprocessingSerializer.serialize()` → `ForkingPickler.dump()` → `reduce_tensor()` → `storage._share_cuda_()` 路径。

---

## 关键配置

- `--colocate` → `offload_train=True`（TMS 启用，LD_PRELOAD 拦截 cudaMalloc）
- `--tensor-model-parallel-size 2`
- `--expert-model-parallel-size 8`
- `--rollout-num-gpus-per-engine 4`
- `--experimental-attention-variant gated_delta_net`
- PyTorch 2.9.1+cu129，CUDA 分配器为 `native`（非 cudaMallocAsync）
- `with_ref=False`（kl_coef=0），使用 `_TensorBackuperNoop`

---

## 技术背景

### CUDA IPC（`cuIpcGetMemHandle`）的限制

`cuIpcGetMemHandle` **只支持** `cuMemAlloc` 或 `cuMemAllocPitch` 分配的内存。
以下类型的内存**不能**做 IPC 共享：

1. **VMM 虚拟内存**（`cuMemAddressReserve` + `cuMemCreate` + `cuMemMap`）→ TMS 的分配方式
2. **cudaMallocAsync**（流序分配器）
3. **pinned host memory**

### torch_memory_saver (TMS) 的工作原理

- `TMS_INIT_ENABLE=1`：进程启动时，所有 `cudaMalloc` 调用被 TMS 拦截，使用 VMM 虚拟内存代替
- `pause()`：将所有 VMM 内存的物理页 unmap（`cuMemUnmap`），同时创建 CPU 备份
- `disable()` 上下文管理器：
  1. `tms_set_interesting_region(False)` → 新的 cudaMalloc 不再走 VMM
  2. 创建 `torch.cuda.MemPool()`，所有分配走 MemPool（理论上是真正的 cudaMalloc）
  3. yield 后，释放 MemPool，恢复 `interesting_region=True`

### 权重同步的完整流程

```
actor.update_weights()
  ├── reload_process_groups()          # 在 disable() 之外！创建 NCCL 组
  │                                     # NCCL warm-up 可能产生 VMM 分配（interesting_region=True）
  └── with torch_memory_saver.disable():   # 进入 disable()
        └── weight_updater.update_weights()
              ├── weights_getter()         # _TensorBackuperNoop.get("actor")
              │     ├── source_getter()    # named_params_and_buffers(translate_gpu_to_cpu=True)
              │     │     └── _maybe_get_cpu_backup(param)  # 查找 TMS CPU 备份
              │     └── _compute_hash_dict(ans)  # 哈希校验
              │
              ├── _get_megatron_full_params()
              │     ├── .to(cuda, non_blocking=True)  # CPU→GPU（MemPool 分配）
              │     ├── torch.cuda.synchronize()
              │     ├── EP broadcast (ep_size=8)
              │     └── all_gather_params_async() (tp=2)
              │
              ├── convert_to_hf()          # Megatron→HF 格式转换
              │
              └── _send_to_colocated_engine()
                    ├── FlattenedTensorBucket(named_tensors)
                    │     └── torch.cat(flattened_tensors)  # 创建 flattened_tensor
                    │
                    └── MultiprocessingSerializer.serialize()
                          └── ForkingPickler.dump()
                                └── reduce_tensor()
                                      └── storage._share_cuda_()  # ← 错误在这里！
                                            └── cuIpcGetMemHandle()  # 对 VMM 指针失败
```

---

## 调查过程与分析

### 1. 确认基本环境

- PyTorch 分配器为 `native`（非 cudaMallocAsync）→ 排除 cudaMallocAsync 导致的 IPC 失败
- `PYTORCH_CUDA_ALLOC_CONF` 未设置 → 排除 expandable_segments 等配置
- `torch.cuda.MemPool()` 在隔离测试中 IPC 正常 → MemPool 本身没问题

### 2. 排除的假设

| 假设 | 排除原因 |
|------|----------|
| `torch.cat` 返回 view 而非 copy | 测试确认总是 copy |
| `expandable_segments` 启用 | 未启用 |
| `cudaMallocAsync` 后端 | 分配器为 native |
| `del pool` 过早释放内存 | `ray.get(refs)` 在 pool 释放之前等待完成 |
| 多 dtype 拆分导致空桶 | `supports_multi_dtypes=True`，所有张量在一个桶 |
| metadata 包含 CUDA 张量 | metadata 只有 Python 原始类型 |
| 非连续张量导致 `get_cpu_backup` 失败 | 会报 AssertionError，不是 CUDA error |

### 3. 最可能的根因分析

**核心问题：`_share_cuda_()` 报 `cudaErrorInvalidValue`，说明目标指针不是 `cuMemAlloc` 分配的，而是 VMM 分配的。**

#### 可能性 A：异步 CUDA 错误（Sticky Context Error）

某个上游操作（如 NCCL EP broadcast 或 TP all_gather）尝试读取了 VMM 内存（TMS paused 后物理页已 unmap），
产生了异步 CUDA 错误。该错误是 **"sticky"** 的——一旦 CUDA context 进入错误状态，
后续所有 CUDA API 调用都会返回该错误码，包括 `cuIpcGetMemHandle`。

这会导致即使 `flattened_tensor` 本身是 MemPool 分配的（IPC 兼容），
`_share_cuda_()` 也会因为 context 错误而失败。

**关键线索**：如果在 `FlattenedTensorBucket` 创建之前添加 `torch.cuda.synchronize()`，
异步错误会在 sync 点被报出，traceback 会指向真正的错误源。

#### 可能性 B：`_maybe_get_cpu_backup` 对某些参数返回了 VMM 张量

`torch_memory_saver.get_cpu_backup(x)` 通过 `(gpu_ptr, nbytes)` 查找 CPU 备份。
如果 TMS 找不到匹配的备份（例如参数的 ptr 不在 TMS 的跟踪列表中），返回 None，
然后 `_maybe_get_cpu_backup` 返回原始的 VMM GPU 张量。

```python
def _maybe_get_cpu_backup(x):
    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x)) is not None:
        return cpu_tensor
    return x  # ← 如果找不到备份，返回 VMM 张量！
```

此后 `.to(cuda, non_blocking=True)` 对同设备 VMM 张量返回**同一个张量**（不 copy），
导致 VMM 张量一路流入 `FlattenedTensorBucket.torch.cat()`。

- 如果 VMM 物理页已 unmap → `torch.cat` 读取 VMM 数据会产生异步 CUDA 错误 → 回到可能性 A
- 如果 VMM 物理页仍然 mapped（可读但不可 IPC）→ `torch.cat` 成功，但其输出分配在 MemPool 是安全的

#### 可能性 C：MemPool 没有真正隔离分配

`torch.cuda.use_mem_pool(pool)` 理论上将分配路由到独立的 MemPool，
但如果 PyTorch 的 CUDACachingAllocator 在某些情况下复用了默认缓存中的 VMM 块，
MemPool 分配的张量可能实际上是 VMM 内存。

**注意**：`tms_set_interesting_region(False)` 可能是进程全局或线程局部的。
如果是线程局部的，且 PyTorch 在后台线程做 `cudaMalloc`，
那个线程的 `interesting_region` 仍为 True → 分配到 VMM。

---

## 修复方案

在 `_send_to_colocated_engine()` 中添加了三层防护：

### 第一层：`torch.cuda.synchronize()` 诊断

在 `FlattenedTensorBucket` 创建前后各添加一次 sync：

```python
torch.cuda.synchronize()       # ← 同步点1：捕获上游异步 CUDA 错误
flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
torch.cuda.synchronize()       # ← 同步点2：捕获 torch.cat 读取 VMM 导致的错误
```

**效果**：
- 如果同步点1抛出异常 → 上游（NCCL/VMM 访问）是根因，traceback 指向真正的错误源
- 如果同步点2抛出异常 → `torch.cat` 读取了 VMM 数据（物理页已 unmap）

### 第二层：IPC 兼容性检测

通过 ctypes 直接调用 `cuIpcGetMemHandle`（无 PyTorch ref-counter 副作用）：

```python
def _cuda_ipc_supported(tensor):
    ret = libcuda.cuIpcGetMemHandle(handle, tensor.data_ptr())
    return ret == 0  # CUDA_SUCCESS
```

如果 `flattened_tensor` 不支持 IPC（VMM 分配），打印详细诊断信息：
- flattened_tensor 的 device/dtype/shape/ptr
- 每个输入张量是否支持 IPC（定位哪个参数是 VMM 的）

### 第三层：Clone 回退

如果检测到 `flattened_tensor` 不可 IPC，立即 clone 强制创建新的 MemPool 分配：

```python
if not _cuda_ipc_supported(ft):
    logger.error(...)  # 打印诊断信息
    ft = ft.clone()    # 强制 MemPool 分配
```

---

## 预期运行结果

| 场景 | 表现 |
|------|------|
| 同步点1抛异常 | traceback 指向上游真正的 CUDA 错误源，需要继续排查 |
| 同步点2抛异常 | 确认 `torch.cat` 读取了 VMM 数据，需要排查哪个参数没有 CPU 备份 |
| IPC 检测失败 + clone | 打印 VMM 张量诊断日志，clone 修复问题，**运行可以继续** |
| 全部通过 | 正常运行，无额外开销 |

---

## 修改的文件

- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
  - 新增 `_cuda_ipc_supported()` 辅助函数
  - 修改 `_send_to_colocated_engine()`：添加 sync + IPC 检测 + clone 回退

---

## 后续建议

1. **运行 `scripts/small-next.sh`**，观察输出：
   - 如果看到 `"flattened_tensor not IPC-shareable"` 日志 → 确认是 VMM 分配问题，clone 已修复
   - 如果 sync 点抛异常 → 需要进一步排查上游的异步 CUDA 错误

2. **如果确认是 VMM 泄漏到 IPC 路径**，后续可以：
   - 在 `_maybe_get_cpu_backup` 中添加断言，确保返回的一定是 CPU 张量
   - 或者在 `_get_megatron_full_params` 中强制 `.clone()` 所有参数
   - 或者修复 TMS 的 `get_cpu_backup` 对所有模型参数都能正确返回 CPU 备份

3. **诊断用的 sync 调用**可以在问题修复后移除以减少开销（每个 bucket 两次 sync ≈ 0.5-1s 总额外耗时）

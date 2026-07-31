# Miles Docker Build 重构设计（草稿）

状态：讨论中的设计草稿。已定决策见 §3，未定分叉见 §6。证据基线：working tree `c2635a40d` + CI run [30503613452](https://github.com/radixark/miles/actions/runs/30503613452)（2026-07-30 schedule）。实现权限尚未授予——本文只记录决策，不代表任何改动已落地。

参照系：`/sgl-workspace/sglang` 的 CI docker build 实践（`docker/Dockerfile`、`.github/workflows/_docker-build-and-publish.yml`、`release-whl-kernel.yml`、`_pr-test-sgl-kernel-build.yml`、`patch-docker-dev.yml`、`release-docker-dev.yml`）。

## 1. 动机与决策

### 1.1 实测成本（fact）

- 一次自动构建的 `build-and-push` job 总计 **72 分钟**（cu13 多架构 44m + cu12-x86 27m），跑在 `["h200","2gpu"]` GPU runner 池上；触发频率 = 每天最多 2 次 cron + 每个 touch docker 路径的 main push；touch docker 的 PR 另占 ~44m（只建 cu13 的 `pr-<num>` tag）。
- **零层缓存**：全 build 日志无一行 `CACHED`。原因：`setup-buildx-action` 的 `docker-container` driver 每次 run 新建一次性 builder，且 `docker/build.py` 无任何 `--cache-from/--cache-to`。哪怕只有 sglang-miles HEAD 动了（第 33/39 层），前面 32 层照样全部重跑。
- cu13 的 44 分钟拆解：amd64 链 28.5m、arm64 链 41.6m（两链并行，wall clock 由 arm64 决定）+ 导出/推送 ~6.5m。
- 单步 Top 3：mbridge pip install 在 amd64 上 13.7m（同步骤 arm64 仅 0.6m——纯 Python 包 20 倍差异 = 对 GitHub 的网络抖动，非计算）；nccl-tests 编译在 QEMU arm64 下 12.6m（x86 原生 1.4m，~9 倍 QEMU 惩罚）；基础镜像拉取每架构 2.5–2.8m。
- cu12 与 cu13 零层共享（`FROM` tag 不同），27 分钟是完整第二遍。
- 历史事故（fact，`docker/verify_transformer_engine.py` 注释）："a missing onnxscript shipped a green image that failed every GPU test"——scheduled build 盲推 `dev` tag，坏镜像的第一发现者是全 fleet 的红 CI。

### 1.2 要决策的事

把 miles 的 docker build 体系（`docker/Dockerfile` + `docker/build.py` + `docker-build.yml` + `pr-test.yml` 的 `docker-build` job）向 sglang 实践迁移，在不改变镜像内容契约的前提下压缩构建时长、消除 GPU 池占用、并把盲推改为有验证门的发布。

### 1.3 成功标准

- "只有上游 HEAD 动了"的常规重建从 72m 降到十几分钟量级（只重跑失效层）。
- `requirements.txt` 改一行的重建不再连坐前面的重量层。
- docker build 不再占用 CI 池的 GPU runner 槽位。
- 坏镜像在 `dev` tag 晋升前被 GPU 冒烟门拦截。
- 副产品：给定同一组 SHA 可复现同一镜像（回应 `docs/ci/02-docker-build.md` 悬置的 "Image retention (open)"）。

## 2. 约束与非目标

- **无原生 arm64 build 机器，短期不会有**（用户决策）。arm64 继续在 x86 机器上 QEMU 构建；优化方向是"让 QEMU 路径变薄"而非消除 QEMU。
- **范围 = cu13 + cu12-x86**（用户决策）。ROCm 变体（`Dockerfile.rocm`、`rocm-*`）只保证不被破坏，不做优化。
- **镜像内容契约不变**：装的东西、版本 pin 语义、`TARGETARCH` 分派 wheels 的机制保持等价。
- **fleet 现实**：自建 runner 全是 GPU 机器；无 CPU-only build box（由此产生 §3.1 的共驻方案）。
- 非目标：sglang 的 `workflow_call`/`tag_config` 重构（可维护性收益，非速度收益，本轮不做）；PR 侧 build 流程语义（`pr-<num>` tag、resolve-ci-image 逻辑）不动，只动它跑在哪、快不快。

## 3. 已定决策

每条含被否决的备选与理由。

### 3.1 共驻 build node（选 B，弃 A）

现 CI 机器中钉**若干台**（数量按池容量定，用户决策：可能好几台），各自加挂**第二个 runner 实例**，label `docker-build`；`docker-build.yml` 与 `pr-test.yml` 的 docker build job 的 runs-on 迁到该 label，build 落在任一 labeled 机器。机器不退出 CI 池（原 CI runner 实例照旧）。多台并存的附带收益：cu13 与 cu12-x86 可拆成两个并行 job 落不同 node，冷态 wall clock 由 72m（串行）回到 44m（取较长者）。

- 依据（fact）：build 全程碰不到 GPU——buildx `docker-container` driver 不向 RUN 步骤暴露 GPU 设备，过去每次 44 分钟 build 本来就在无 GPU 沙箱里完成；`verify_transformer_engine.py` 只读 `importlib.metadata`，`TMS_CUDA_MAJOR` 读 torch 编译期常量，源码编译只需 nvcc。
- 共存性（fact）：CI job 容器以 `--gpus all` 起（`_run-ci.yml:38`），纯设备映射无独占；默认 compute mode 下多进程共享一张卡成立。冒烟探针瞬时几百 MB、存活秒级，H200 141GB 余量下无 OOM 风险，无需选卡/等待逻辑。真正的 CI job 唯一性由原 CI runner 实例并发=1 保证。
- 弃 A（整机退出 CI 池）：构建频率（每天 2–4 次 × 44–72m）不足以支付一台双卡 H200 的 24/7 独占；若日后观测到共驻干扰（build 编译负载拖慢同机 CI job），再升级到 A。
- 一次性运维确认：build node 上 `nvidia-smi -q | grep "Compute Mode"` 应为 Default（若被设成 EXCLUSIVE_PROCESS/MIG 则探针共存不成立）。

### 3.2 缓存后端：钉子机上的持久本地 builder（registry 缓存降为可选二级）

build node 上预建持久命名 buildx builder，层缓存与 `--mount=type=cache` 内容跨 build 存活于本机。

**自动 prune 卫生：推迟到后期**（用户决策：偏 heavy，放到后面做）。附加依据：sglang 的 `docker system prune -af --filter "until=72h"` 跑在专用 build node 上才安全；miles 的 build node 与 CI runner 共驻同一 docker daemon，`system prune` 会误删 CI job 复用的镜像（`radixark/miles:dev`、`pr-<num>` 等），逼 CI 重拉。日后磁盘有压力时的正确姿势是**只对命名 builder 单独** `docker buildx prune --builder <name> --filter "until=..."`（不触碰镜像存储），而非照搬 sglang 的 system prune。

- 依据：sglang 本地持久缓存的前提是"build node 状态持久且定期轮到 build"（其 kernel wheel 构建钉在 `x64-kernel-build-node`，缓存在 `$HOME/.cache/sgl-kernel/buildx`）。§3.1 为多台 build node：每台各养一份本地缓存——层缓存不按时间过期，机器定期轮到 build 则稳定层常驻；代价是每台各付一次冷启动、磁盘各占一份、平均命中率随机器数稀释（build 落到久未轮到的机器就冷一截）。
- 弃"registry 缓存为主"：本地缓存零网络开销、运维最简，少数几台机器轮换足够密时命中率可接受。registry 缓存（`--cache-to type=registry,mode=max` 推专用 cache tag）保留为二级，**启用条件具体化**：build node 数量增多或轮换稀疏导致命中率可感知下降时，或 fork PR/整组机器换代的灾备场景。
- 参照系诚实记录（fact）：sglang **发布**镜像走 `--no-cache` 全冷（全仓 38 处），冷得起靠专用原生双架构 node + 并行 multi-stage + 重编译产物全 wheel 化；miles 无此硬件，层缓存是"专用硬件"的替代品。sglang 自己在受约束的高频路径（kernel wheel、rust CI）同样选择缓存。

### 3.3 缓存失效：上游 SHA 作为 build-arg 注入（镜像变可复现构建）

`check-upstream`（`docker-build.yml:69-99`）已在轮询 sglang-miles HEAD、Megatron miles-main HEAD、miles-wheels 各 rolling release 的资产指纹——现状只拿来做"要不要 build"的开关。改为：workflow 起始统一解析（sglang SHA、megatron SHA、miles SHA、wheels 指纹），经 `build.py` 透传为 build-arg；Dockerfile 对应层改为 `git checkout <SHA>`，wheels 下载层前置 `ARG WHEELS_FP` 作为缓存键。

- 正确性论证：Docker 层缓存按"指令文本 + 父层"复用。现状 `git clone -b miles-main` 指令文本永不变而分支每天动——今天全冷构建掩盖了这个坑；开启层缓存后会静默复用旧 HEAD。SHA 进指令文本后：上游没动 → 命中且命中恒正确；动了 → 精确失效对应层往后。
- 硬约束（无静默回退）：SHA build-arg 为空时 Dockerfile 必须显式失败，禁止回退到 branch HEAD——否则失效机制被静默绕过。
- 弃"粗粒度 CACHE_BUST 参数"：clone 层之后永不命中，镜像仍不可复现。
- 副产品：同一组 SHA → 同一镜像，可按历史 SHA 重建旧镜像；配合 image label 记录 SHA 组（参照 sglang `docker_build_metadata_args.py` 的 build-commit/build-url label 实践），回应 02 doc 的 retention 悬案。
- 现状盘点：`SGLANG_COMMIT`、`MILES_COMMIT` args 已存在；`MEGATRON_*` 只有 branch 需加 SHA arg；mbridge、fast-hadamard、TMS、Emerging-Optimizers 等 git 安装已 SHA pin（天然缓存安全）；逐个核对残余 branch pin。

### 3.4 GPU 冒烟门：build 后、tag 晋升前

build node 上 `docker run --gpus all <刚构建的镜像>` 跑秒级探针集，通过才晋升 `dev`/`pr-<num>` tag。探针集内容待定稿（§6），最小集合方向：`torch.cuda.is_available()`、TE 真实 import + 一个小 op（覆盖 onnxscript 类事故）、sglang/miles import、nccl-tests 二进制存在。

- 边界（fact）：x86 node 只能冒烟 amd64 镜像；arm64 镜像验证接受空缺，将来有 ARM GPU 机器再补。
- GPU 在 build node 上的唯一角色就是这个门；build 本身仍然无 GPU。

### 3.5 层序重排 + 依赖单一清单 + 全面 cache mount（sglang deps/source 分层模式）

参照 sglang 的两层拆分（`COPY python/pyproject.toml` + 空壳包装依赖层，源码 "LAST for better caching"）：

- 层序按**变化频率升序**：稳定贵重层（apt、nccl-tests、wheels 安装、TE、apex）在前 → `requirements.txt` 依赖层居中 → 高频源码层押后（Megatron → sglang → miles → router）。目标：`requirements.txt` 改一行只重跑依赖层 + 其后的廉价源码层。
- **pip 纪律（迁自 sglang，miles 场景下更彻底）**。sglang 的模式（fact）：全 Dockerfile 只在 `torch_deps` stage 放开求解一次（空壳包 `pip install ".[all]"`，行 249），随即 `pip freeze > constraints.txt` 冻结（行 259）；此后所有安装要么 `--no-deps`（9 处：kernel wheels、nixl、hpc-ops、最终 sglang editable 等），要么 `-c constraints.txt` 受限求解（行 594），pip 再无权移动环境。miles 连自己的大求解都没有——完整环境来自 `lmsysorg/sglang` 基础镜像，miles 的每次安装都属于"求解之后"，**默认桶就是 `--no-deps`**。
- 收编桶型：**第 0 层快照**——进场 `pip freeze > constraints.txt`（过滤 editable/direct-ref 行）；**默认 no-deps 桶（最大）**——新增 `docker/requirements-nodeps.txt` 单层 `pip install -r ... --no-deps` 装完，成员为全部 wheel 安装与 deps 已在 base 的包（flash-attn、flash_attn_3、apex、flash-linear-attention、tilelang、mbridge、megatron-bridge、energon、multi-storage-client、tile_kernels、Emerging-Optimizers 等，逐包核对 deps-in-base 后定案）；**受限求解桶**——真引入新传递依赖的走 `-r requirements.txt -c constraints.txt`（blake3、polars、psycopg、mcp、modelopt 等新叶子包），冻结件冲突时 pip 响亮失败而非静默重装；**保留独立层**——真依赖序或特殊 flag（FlashQLA 必须在 tilelang 之后且 `--no-build-isolation`，TE 三件套 `--force-reinstall --no-deps` + 手补运行时依赖）。形态说明：sglang 对散包用 Dockerfile 内联清单（一行一包 + `-c`），无 requirements 文件；文件化收编是 miles 本地化选择（包多、diff 友好），两种形态缓存失效语义等价。
- 现状隐患（本桶型消除的）：裸 `-r requirements.txt` 有权静默重装前面层的关键件（如新依赖与已装 torch 版本约束冲突时）——今天不出事仅因约束恰好满足。
- 所有 apt/pip 步骤加 `--mount=type=cache`（持久 builder 下真正跨 build 存活）；删除 `rm -rf /root/.cache/pip` 层。
- **wheels release 下载缓存 mount**（用户提议，采纳并修正）：下载步骤挂 `--mount=type=cache,target=/wheels-cache`，缺的资产下进缓存后 cp 进层内 `/tmp/wheels`。两个必要修正：(a) 现有 skip-if-exists 必须升级为**资产同一性检查**（按 release API 已解析的 asset `id` 比对；rolling release 同名重传是常态——check-upstream 的指纹机制即为此而生——否则持久 mount + skip-if-exists = 永久装旧 wheel 的静默事故）；(b) mount 不能直接盖 `/tmp/wheels`（会遮蔽 `COPY wheel[s]/` 本地 wheel 工作流，且 cache mount 只在声明它的 RUN 内可见，后续 `pip install /tmp/wheels/...` 层看不到）。边界与分工模型：层缓存缓存**计算**（命中=整步不执行，正确性由 BuildKit 键保证，故需 SHA 进指令文本）；cache mount 缓存**数据**（RUN 每次执行，新鲜度责任在 RUN 内程序——pip/apt 自带校验逻辑所以直接挂，curl 脚本必须自补 asset-id 检查）。判据：pull 下来的数据归 mount，装好之后的状态归层缓存。此 mount 省的是指纹变化时未变资产的重下（分钟级），不替代层缓存对安装/编译层的跳过。
- miles 依赖清单现状（fact）：`requirements.txt` 是唯一运行时清单（`pyproject.toml` 无 dependencies 段，miles 自身 `pip install -e . --no-deps`）——地基已对，缺的是层位置和收编。

### 3.6 显式架构声明

Dockerfile 增加 `ARG TORCH_CUDA_ARCH_LIST`（含 nccl-tests 的对应 `NVCC_GENCODE`），按 miles 实际 fleet 收窄。矩阵宽窄待拍板（§6）。

- 现状（fact）：Dockerfile 零架构声明；源码编译走各家无 GPU fallback（torch cpp_extension 探测不到 GPU 时编译全部支持架构；nccl-tests Makefile 按 CUDA 版本全架构 gencode）——能跑通但又慢又肥，QEMU 下 12.6m 的 nccl-tests 大半在编译用不到的架构。
- 原则（对齐 sglang docker/Dockerfile:319-334）：发布镜像的架构支持是**声明的契约**，不是构建机探测的结果；靠本机 GPU 探测对发布镜像本来就是错误做法。运行时由 CUDA runtime 按实际卡从 fatbin 选 cubin/JIT PTX。

## 4. 设计要点（机制细节）

- **runner 拓扑**：每台 build node（H200 机器，若干台）= 原 CI runner 实例（label 照旧，接 CI job，并发 1）+ 新 build runner 实例（label `docker-build`，接 docker build job，并发 1）。单机上 build 串行排队；跨机器可并行（cu13/cu12 拆 job）。compute mode 一次性确认（§3.1）对每台 labeled 机器执行。
- **SHA 解析点**：workflow 单一步骤解析全部上游引用（schedule/push/dispatch 三种触发统一走这条路），输出 SHA 组 → 既做 should_build 判断（对比缓存的上次值，逻辑不变）又做 build-arg 注入 + image label。
- **`build.py` 改动**：透传任意 `--build-arg`（02 doc 已记录这是缺口）；接持久 builder（`--builder <name>`）；可选 `--cache-to/--cache-from` 参数留二级缓存口子。
- **QEMU 原则（迁自 sglang）**："编译必须原生，薄层随便 QEMU"。sglang 自己在 ubuntu-latest 上 QEMU 构建多架构 overlay 镜像（`release-docker-dev.yml`，现场生成 `FROM 镜像 + snippet` 两行 Dockerfile）——QEMU 可接受的条件是层里无编译。miles 的 arm64 链在 P2（§6 wheel 化）完成后退化为此类薄层。
- **cu13/cu12 关系（fact）**：`FROM` tag 不同 → 零层共享，缓存各自独立；两者在同一钉子机上串行（warm 后各自收缩，串行可接受）。

## 5. 实施切分与验证

实施权限未授予；以下为建议切分，每刀独立可回滚。

1. **PR-1 机器迁移**：build node 双 runner 实例 + runs-on 切换。此刀**不开层缓存**（builder 保持一次性或显式 `--no-cache`），因为 SHA 注入未落地前层缓存不安全。收益：GPU 池占用归零。验证：一次 dispatch build 全绿，产物 digest 与旧路径等价（`pip freeze` diff 为空）。
2. **PR-2 SHA 注入 + 开启持久缓存**：`check-upstream` 重构为统一解析步骤，`build.py` 透传 build-args，Dockerfile clone 层改 SHA checkout（空 SHA 硬失败），建持久 builder（不带自动 prune，见 §3.2）。验证：同 SHA 组连打两次，第二次近全命中且 wall clock 达标；伪造单个 SHA 变化，确认只有预期层失效；`pip freeze` diff 为空。
3. **PR-3 层序重排 + 收编 + cache mount**：§3.5 全部。验证：`requirements.txt` 改一行 → 重量层全命中；镜像内容等价性同上。
4. **PR-4 GPU 冒烟门**：探针集 + 晋升逻辑。验证：人为破坏（如构建时卸掉 onnxscript）必须拦截。
5. **PR-5 架构声明**：`TORCH_CUDA_ARCH_LIST`/`NVCC_GENCODE`。验证：编译时长下降；探针在 H200 实卡通过。
6. **后续独立轨道**：P2 wheel 化（见 §6）；builder 缓存 prune 卫生（§3.2 推迟项——磁盘水位有信号后再做，形态为对命名 builder 单独 `buildx prune`）。

通用验证口径：每刀前后各留一次完整 build 的分段耗时（本文 §1.1 的解析脚本可复用），镜像内容等价 = `pip freeze` diff + 关键二进制/入口存在性清单。

## 6. 未定分叉与待讨论

- **multi-stage 并行改造**（§3 未讨论）：miles 39 层串行链是否拆 sglang 式并行 builder stage + `COPY --from` 收集。与 §3.5 层序重排有重叠收益，需评估在"增量镜像"（基于 lmsysorg/sglang 而非裸 CUDA）场景下的边际收益是否值得复杂度。
- **P2：QEMU 源码编译 wheel 化**（§3 未讨论）：nccl-tests、FlashQLA、fast-hadamard（arm64 fallback 路径）、TMS 等进 miles-wheels 预编译发布，使 arm64 链退化为纯装 wheel 薄层。涉及 miles-wheels repo 的构建矩阵扩展，独立轨道。附带收益：§3.5 "保留独立层"中 build 期 import 依赖类（FlashQLA 须 tilelang 先装、TMS 须 torch 在场——源码构建 `--no-build-isolation` 在当前环境跑 setup.py 所致）随 wheel 化消失，全部退化为 no-deps 桶成员；最终独立层只剩事务性替换类（TE 三件套 uninstall→triplet→补依赖→verify、mooncake/router force-reinstall、TE patch）。
- **`TORCH_CUDA_ARCH_LIST` 矩阵宽窄**：只覆盖现役 fleet（最快最小）vs 保守全谱（现状等价，无提速）。
- **冒烟探针集定稿**：最小集合 vs 加轻量端到端（如 sglang server 起停）。
- **散装 pip 层收编清单**：逐包核对 flag/顺序约束后出最终清单。
- **cu12-x86 的构建频率**：是否维持每次自动构建都跟建，或降为按需/低频（零层共享使它永远是全价构建）。
- **mbridge 类网络抖动**（13.7m 异常）：层缓存命中后大幅摊薄；残余风险可加 pip retry 配置，暂不单独立项。

## 7. 风险

- **缓存稀释与冷启动**（多台 build node 形态）：单机故障不再是单点（其余 labeled 机器接管，冷但可用），代价转为每台一份冷启动与命中率稀释；机器数与轮换密度失衡时按 §3.2 条件启用 registry 二级缓存。磁盘水位按 §3.2 推迟项处理。
- **共驻干扰**：build 编译负载（`make -j$(nproc)` 等）拖慢同机 CI job。缓解：观测；恶化则升级为方案 A（整机退出 CI 池）或限制 build 并发核数。
- **SHA 注入接线错误**：空值/错值静默回退到 branch HEAD 会让缓存失效机制形同虚设——设计上以"空 SHA 硬失败"封死，实现时作为验收项。
- **层缓存膨胀**：持久 builder 缓存无自动 prune（§3.2 决策）会缓慢增长——接受为已知风险，观测 build node 磁盘水位；触线时对命名 builder 单独 `buildx prune`，禁止在共驻机上 `docker system prune`。
- **arm64 验证空缺**：冒烟门只覆盖 amd64；arm64 坏镜像仍可能盲推。接受并记录，待 ARM GPU 机器补齐。

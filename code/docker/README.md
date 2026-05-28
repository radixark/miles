# 九章云 H100 个人开发流程（jzgpu17 / nlp-intern / houjue）

按手册「调试机使用」+「启动训练任务」流程组织，开发容器默认开启 **DinD**
（容器内独立 dockerd）。

- 调试机：`jzgpu17.unidev.ai`（已装 Docker 28.0.1、8×H100）
  - 内网：`172.31.72.17:22`
  - 公网：`120.220.102.24:60017`（SSH 登录调试机宿主）
  - 调试容器 SSH 端口在 `[60020, 60099]` 范围**自动**选空闲端口
- 用户：`houjue` (uid=2244, gid=2244)，组 `nlp-intern(2216)` / `docker(3001)`，sudo 免密
- Harbor：`harbor.unisound.ai/unisound/houjue-dev:<tag>`
- K8s：组 namespace `nlp-intern`（GPU 配额 24，已用 18）→ 仅用于提交 MPIJob 跑训练
- DinD：容器内独立 dockerd + buildx + compose + nvidia-container-toolkit
  - jzgpu17：`make run` 默认 `--privileged` + named volume `houjue-dev-docker`

## 目录

```text
.
├── Dockerfile.houjue                # 个人开发镜像（手册 Dockerfile.qiudelai 同款风格 + DinD）
├── docker-daemon.json               # 容器内 dockerd 配置（nvidia runtime + 内网 registry mirror）
├── entrypoint.sh                    # 启动时重置密码 + 起 dockerd + 起 sshd
├── build.sh                         # docker login / build / push
├── Makefile                         # 全流程封装（make build 自动 hardlink offline/）
├── README.md
├── offline/                         # 离线 docker 二进制 + 离线 deb 包（~270MB，归档不依赖外部目录）
└── examples/
    ├── run-debug-container.sh       # 在 jzgpu17 起调试容器（支持 dind/dood/off 三种模式）
    └── mpijob-houjue.yaml           # K8s MPIJob 训练模板
```

## 一图流程

```text
   [jzgpu17 调试机]                                  [Harbor]                      [K8s 集群]
       │                                                │                              │
  ① docker build  ────────────────────────────►   image:vYYYYMMDD                      │
       │                                                │                              │
  ② docker push  ─────────────────────────────►   image:vYYYYMMDD                      │
       │                                                                               │
  ③ docker run (本地调试容器, VSCode 进去日常开发)                                      │
       │                                                                               │
  ④ kubectl apply -f mpijob-houjue.yaml ───────────────────────────────────────────►  MPIJob
                                                                                       │
                                                              ⑤ Launcher / Worker Pod 拉同一个 image 跑训练
```

## 0. 前置

```bash
# 公网登录 jzgpu17（注意端口 60017）
ssh houjue@120.220.102.24 -p 60017

# 进到工作目录（NFS 上的个人区）
cd /fs/nlp-intern/houjue/docker
```

VSCode 直接连调试机宿主：

```text
Host jzgpu17
    HostName 120.220.102.24
    User houjue
    Port 60017
```

## 1. 构建镜像 → 推到 Harbor

```bash
make login                         # 等价于 docker login harbor.unisound.ai
make build                         # 构建 harbor.unisound.ai/unisound/houjue-dev:v<日期>
make push                          # 推送
# 或者一步：
make build-push
```

> **离线 docker 包**：jzgpu17 所在网段对 `download.docker.com` / `nvidia.github.io`
> 的 HTTPS 流量会被中间盒按 SNI RST，所以镜像里的 docker stack 全部走离线安装。
> 离线二进制和 deb 包已经归档在仓库 `offline/` 目录（~270MB，纳入版本控制）：
>
> | 文件 | 说明 |
> |---|---|
> | `offline/docker-pkg/` | docker 28.0.1 全套二进制（dockerd, containerd, runc...） |
> | `offline/debian-gpu/` | nvidia-container-toolkit 1.19.0 的 4 个 deb |
> | `offline/docker-buildx`, `offline/docker-compose` | CLI 插件 |
> | `offline/docker.initd` | cgroup v2 自适应的 `/etc/init.d/docker` 启动脚本 |
>
> `make build` 会先校验上述清单完整，缺文件直接报错退出。
>
> Ubuntu apt 包源走 jzgpu17 自己的 Nexus apt-proxy
> (`http://172.31.72.17:48081/repository/apt-proxy/`)，毫秒级响应。
>
> 镜像里 dockerd 默认的 registry mirror 是 `http://172.31.72.17:45000`，
> `docker pull` 会自动走内网加速。

镜像默认基于 `nvcr.io/nvidia/pytorch:26.04-py3`（NVIDIA 官方 NGC，CUDA + PyTorch +
NCCL/cuDNN/Apex，适用 H100，约 23GB / jzgpu17 已缓存）。常见可选基础：

```bash
BASE_IMAGE=ubuntu:22.04 make build                                       # 极简，无 CUDA
BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 make build         # 仅 CUDA
BASE_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3 make build                   # 更稳定的旧版 NGC
BASE_IMAGE=harbor.unisound.ai/unisound/ms-swift:v4.0.1 make build        # 自带 ms-swift 训练框架
```

⚠️ 手册要求：**不要重名 tag**，不同版本一定换 IMAGE_TAG。

```bash
IMAGE_TAG=v20260514a make build-push
```

## 2. 起调试容器（每天日常开发用）

```bash
make run                           # 自动在 60020-60099 选空闲端口；sshd 由 entrypoint 自启
make ssh-info                      # 打印 SSH 连接信息（实际端口）
```

`make run` 启动后会做三件事：

1. 自动挑两个空闲端口：一个映射容器 `22`（SSH），一个映射 `2222`（备用）
2. 把分配结果写到 `.runtime/houjue-dev.env`，方便后续命令读取
3. `entrypoint.sh` 在容器内以 root 身份自动起 `sshd`，直接 ssh 进来即可

> 不想要 sshd 自启？`docker run -e ENABLE_SSHD=0 ...`。
> sshd 意外挂了的兜底：`make start-sshd`（等价 `docker exec -u 0 houjue-dev /etc/init.d/ssh start`）。

如果想固定端口，覆盖环境变量：

```bash
HOST_SSH_PORT=60022 HOST_ALT_PORT=60024 make run
```

实际 docker 命令等价于（默认 DIND_MODE=dind）：

```bash
docker run -itd --name houjue-dev --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --privileged \
  -v houjue-dev-docker:/var/lib/docker \
  -v /fs:/fs \
  -p <auto>:22 -p <auto>:2222 \
  harbor.unisound.ai/unisound/houjue-dev:vYYYYMMDD bash
```

切换 DinD 模式：

```bash
DIND_MODE=dood make run            # 挂宿主 /var/run/docker.sock，跟宿主共享 dockerd
DIND_MODE=off  make run            # 容器内只有 docker CLI，没 daemon
DIND_MODE=dind make run            # 默认，容器内独立 dockerd
```

进去看一眼：

```bash
make exec                          # docker exec -it -u houjue houjue-dev bash
# 容器里：
id                                 # uid=2244(houjue) gid=2244(houjue) groups=...,3001(docker),2216(nlp-intern)
ls /fs/nlp-intern/houjue           # NFS 透传可见
docker info                        # 容器内 dockerd（Storage Driver: overlay2）
docker run --rm hello-world        # 嵌套子容器验证
```

## 2.1 DinD 使用速查

容器内 docker 验证 / 排错：

```bash
make dind-info                     # docker info + 最后 20 行 dockerd 日志
make dind-test                     # docker run hello-world
make dind-logs                     # tail -f /var/log/dockerd.log
make rm-all                        # 删容器 + 删 named volume（释放镜像缓存）
```

DinD 子容器用 GPU：

```bash
# 容器内（已装 nvidia-container-toolkit + daemon.json 注册了 nvidia runtime）
docker run --rm --gpus all --runtime=nvidia \
    nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

> 注意：子容器能用 GPU 依赖外层容器看得到 `/dev/nvidia*`，所以
> 外层 `make run` 的 `--gpus all` 不能去掉。

`/var/lib/docker` 数据卷：

| 模式 | 数据位置 | 删容器后 |
|---|---|---|
| `make run`（jzgpu17） | named volume `houjue-dev-docker` | 镜像缓存保留 |

## 3. VSCode Remote 接入容器

入口：

| 地址 | 用途 |
|------|------|
| `120.220.102.24:60017` | 公网 → jzgpu17 宿主 |
| `172.31.72.17:22` | 内网 → jzgpu17 宿主 |
| `<jzgpu17>:<auto>` | 宿主 → 调试容器（端口由 `make run` 自动分配） |

`make run` 后跑 `make ssh-info` 会输出当前实际端口。下面用 `<PORT>`
代指容器映射端口（每次重建容器可能不同）。

### 方案 A（推荐）：ProxyJump，一跳直达容器

```text
Host jzgpu17
    HostName 120.220.102.24
    User houjue
    Port 60017

Host houjue-dev
    HostName 127.0.0.1                # 在 jzgpu17 内部回环连容器
    User houjue
    Port <PORT>                       # make ssh-info 看实际值
    ProxyJump jzgpu17
```

VSCode 直接连 `houjue-dev` 即可。

> 端口换了？只改 `houjue-dev` 这一处的 `Port`。

### 方案 B：在公司内网时直连

```text
Host houjue-dev
    HostName 172.31.72.17
    User houjue
    Port <PORT>
```

### 方案 C：直接连调试机宿主，不进容器

```text
Host jzgpu17
    HostName 120.220.102.24
    User houjue
    Port 60017
```

## 4. 容器稳定后保存为新镜像

环境跑稳了想沉淀下来：

```bash
make commit                        # docker commit houjue-dev <image>:<tag>-snapshot
docker push <镜像>                 # 手动确认后 push
```

## 5. 提交训练任务（K8s MPIJob）

改 `examples/mpijob-houjue.yaml`：

- `image:` 换成你刚 push 的 tag
- `Worker.replicas`：1 机就 1，多机改成实际节点数
- 多机时打开 `nvidia.com/hostdev: 10`
- `args:` 换成实际训练入口（手册的 `mpirun_xxx.sh` 模式）

```bash
make mpi-apply                     # kubectl apply -f examples/mpijob-houjue.yaml
make mpi-status                    # 看 MPIJob / Pod
make mpi-logs                      # 跟 launcher 日志
make mpi-delete                    # 清理
```

查看配额：

```bash
make quota
# Resource                  Used  Hard
# pods                      ...    200
# requests.nvidia.com/gpu   18     24    ← 当前 nlp-intern 还剩 6 卡
```

## 6. 数据 / 安全（手册要点）

- ❌ 禁止 `ls` / 代码 list 超大目录（>10 万文件）；用 `find -maxdepth N`、`rg` 限定路径
- ❌ 禁止 VSCode 在大目录里全文搜索；本地 / 远端用 `rg`
- ✅ VSCode 关闭 `files.followSymlinks`
- ✅ 个人代码与数据放 `/fs/nlp-intern/houjue`，权限 775
- ✅ 大文件传输到 10.10.20.51 上做（参考手册「数据拷贝」）
- ✅ 镜像版本控制：`v20260514`、`v20260514a`、`v0.1.0` 这种，**不要覆盖** `latest`

## 7. 常用命令速查

```bash
# 镜像
make login / build / push / build-push / print-image

# 调试容器（jzgpu17 上，默认 DinD）
make run               # 启动（自动选端口 + --privileged + named volume + 挂 authorized_keys；sshd 自启）
make start-sshd        # 兜底：sshd 挂了再用，正常不用调
make exec / exec-root  # 进容器（houjue / root）
make logs / ps         # 查日志/进程
make dind-info         # 看容器内 dockerd 状态
make dind-test         # docker run hello-world 验证
make dind-logs         # 跟 dockerd 日志
make commit            # 当前容器 -> 新镜像
make stop / rm         # 停止/删除（rm 不删 DinD volume）
make rm-all            # 删容器 + 删 DinD volume（彻底清理）
make ssh-info          # 打印 SSH 串

# K8s 训练
make mpi-apply / mpi-status / mpi-logs / mpi-delete
make quota
```

## 8. 排错（高频）

**MPIJob 一直 Pending**
→ `make quota`，看 `requests.nvidia.com/gpu` 是否超配额；删掉旧的：

```bash
kubectl get mpijob -n nlp-intern
kubectl delete mpijob <name> -n nlp-intern
```

**Pod 起来后 sshd 没起 / mpirun 连不上 worker**
→ 手册 FAQ #1：检查 `sshAuthMountPath: /home/houjue/.ssh`，并且 worker
`lifecycle.postStart` 里有 `sudo /etc/init.d/ssh start`（模板已加）。

**本地 VSCode/ssh 连 `houjue-dev` 报 `channel 0: open failed: connect failed: Connection refused`**
→ ProxyJump 第二跳失败，跳板机 jzgpu17 找不到容器 22 端口上的 sshd。按顺序自查：

1. 容器活着吗？ `make ps`
2. 容器真实端口是多少？ `make ssh-info`（自动 `docker port` 读，每次 run 可能变）
   → 跟 `~/.ssh/config` 的 `Port` 对一下，不一致就更新本地 ssh config
3. 容器内 sshd 在吗？ `docker exec -u 0 houjue-dev pgrep -a sshd`
   → 没进程就 `make start-sshd`；entrypoint 已自启，如果还是没起来看 `make logs` 找 `[entrypoint] WARN: sshd` 行

**docker push 401**
→ `make login` 重新登录 Harbor。

**容器里 `id` 看到的不是 houjue/2244**
→ 检查 Dockerfile.houjue 的 `ARG USER_UID/USER_GID/...` 是否和宿主机
`id` 输出一致。当前 jzgpu17 上是 `uid=2244 gid=2244 groups=2244,2216,3001`，
模板已对齐。

**容器里 `docker info` 报 `Cannot connect to the Docker daemon`**
→ 看 `make dind-info`：
  1. 容器是不是 `--privileged` 起来的？`make run` 默认是
  2. `/var/log/dockerd.log` 报错？常见：
     - `Failed to start daemon: error initializing graphdriver: prerequisites for overlayfs are not satisfied` ← 内核太旧 / 没 mount cgroup；试 `docker-daemon.json` 把 `storage-driver` 改成 `vfs`（慢但兼容）
     - iptables 报错 ← 一般无害；如果妨碍，加 `--iptables=false` 启 dockerd

**`docker run` 报 `cannot enter cgroupv2 "/sys/fs/cgroup/docker" with domain controllers -- it is in threaded mode`**
→ cgroup v2 经典坑。entrypoint.sh 已自带修复：把 PID 1 进入根 cgroup 的进程
全部移到 `/init` 子 cgroup，然后启用所有 controllers 到根 subtree_control，这样
dockerd 创建 `/sys/fs/cgroup/docker` 时拿得到 `io/memory` 等 domain-only controllers，
type 才是 `domain` 而不是 `threaded`。验证方法（在容器里执行）：

```bash
cat /sys/fs/cgroup/cgroup.type            # 期望: domain（不是 domain threaded）
cat /sys/fs/cgroup/docker/cgroup.type     # 期望: domain（不是 threaded）
cat /sys/fs/cgroup/cgroup.subtree_control # 应包含 io memory
```

**容器里 `docker run --gpus all` 报 nvidia runtime 找不到**
→ 外层 `make run` 必须带 `--gpus all`（默认带）。容器内 dockerd 会
自动从 `/etc/docker/daemon.json` 加载 nvidia runtime；如果 nvidia-container-toolkit
构建时跳过了，重 build 镜像或手动 `apt-get install nvidia-container-toolkit`。

**DinD 占盘**
→ `/var/lib/docker` 在 named volume 里。jzgpu17 上：
`docker volume ls | grep houjue-dev-docker`、`docker system df`；
彻底清：`make rm-all`。

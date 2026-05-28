#!/usr/bin/env bash
# 在调试机 jzgpu17 上启动个人调试容器（手册「调试机使用」流程）
# 端口策略：HOST_SSH_PORT 默认 auto，会在 [PORT_LO, PORT_HI] 区间自动选空闲端口
set -euo pipefail

NAME="${NAME:-houjue-dev}"
IMAGE="${IMAGE:-harbor.unisound.ai/unisound/houjue-dev:v$(date +%Y%m%d)}"
HOST_SSH_PORT="${HOST_SSH_PORT:-auto}"     # auto = 自动找空闲端口
HOST_ALT_PORT="${HOST_ALT_PORT:-auto}"     # 备用端口（容器 2222）
PORT_LO="${PORT_LO:-60020}"                # 自动选端口下界
PORT_HI="${PORT_HI:-60099}"                # 自动选端口上界
FS_MOUNT="${FS_MOUNT:-/fs}"
GPUS="${GPUS:-all}"
HOST_INTERNAL_IP="${HOST_INTERNAL_IP:-172.31.72.17}"
HOST_PUB_IP="${HOST_PUB_IP:-120.220.102.24}"
HOST_PUB_PORT="${HOST_PUB_PORT:-60017}"
# 复用宿主 authorized_keys，免密 SSH 进容器（设为空字符串可关闭）
AUTH_KEYS_FILE="${AUTH_KEYS_FILE:-${HOME}/.ssh/authorized_keys}"

# === DinD 配置 ===
# DIND_MODE=dind|dood|off
#   dind (默认): 容器内跑独立 dockerd，需要 --privileged，/var/lib/docker 走 named volume
#   dood:        挂宿主 /var/run/docker.sock，跟宿主共享 dockerd 和镜像缓存
#   off:         容器内不暴露 docker（除了 CLI 还在）
DIND_MODE="${DIND_MODE:-dind}"
DIND_VOLUME="${DIND_VOLUME:-${NAME}-docker}"   # named volume，删容器不丢镜像缓存

# 选一个未被占用的 TCP 端口
# 1) /proc/net/tcp{,6} 监听判断；2) docker 已发布端口
pick_free_port() {
  local lo="$1" hi="$2" exclude="${3:-}"
  local listen_hex
  # /proc/net/tcp 第 4 列 == 0A 表示 LISTEN
  listen_hex=$(awk 'NR>1 && $4=="0A" {split($2,a,":"); print a[2]}' /proc/net/tcp /proc/net/tcp6 2>/dev/null \
                 | sort -u)
  local docker_used
  docker_used=$(docker ps --format '{{.Ports}}' 2>/dev/null \
                  | grep -oE '0\.0\.0\.0:[0-9]+' | awk -F: '{print $2}' | sort -u || true)

  local p hex
  for p in $(seq "$lo" "$hi"); do
    [[ ",$exclude," == *",$p,"* ]] && continue
    hex=$(printf '%04X' "$p")
    if echo "$listen_hex" | grep -qx "$hex"; then continue; fi
    if echo "$docker_used" | grep -qx "$p"; then continue; fi
    echo "$p"
    return 0
  done
  return 1
}

# 处理 auto
if [ "${HOST_SSH_PORT}" = "auto" ]; then
  HOST_SSH_PORT=$(pick_free_port "${PORT_LO}" "${PORT_HI}") \
    || { echo "[error] ${PORT_LO}-${PORT_HI} 范围内没有空闲端口"; exit 1; }
fi
if [ "${HOST_ALT_PORT}" = "auto" ]; then
  HOST_ALT_PORT=$(pick_free_port "${PORT_LO}" "${PORT_HI}" "${HOST_SSH_PORT}") \
    || { echo "[error] 第二个端口选不到"; exit 1; }
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "容器 ${NAME} 已存在；如需重建：docker rm -f ${NAME}"
  exit 1
fi

GPU_ARG=()
if [ -n "${GPUS}" ] && [ "${GPUS}" != "none" ]; then
  GPU_ARG=(--gpus "${GPUS}")
fi

# 把宿主 authorized_keys 以只读方式挂进容器，做免密登录
AUTH_KEYS_ARG=()
AUTH_KEYS_NOTE="(关闭)"
if [ -n "${AUTH_KEYS_FILE}" ] && [ -f "${AUTH_KEYS_FILE}" ]; then
  AUTH_KEYS_ARG=(-v "${AUTH_KEYS_FILE}:/home/houjue/.ssh/authorized_keys:ro")
  AUTH_KEYS_NOTE="${AUTH_KEYS_FILE} -> /home/houjue/.ssh/authorized_keys (ro)"
elif [ -n "${AUTH_KEYS_FILE}" ]; then
  echo "[warn] AUTH_KEYS_FILE=${AUTH_KEYS_FILE} 不存在，跳过免密挂载"
fi

# === DinD 参数 ===
DIND_ARG=()
DIND_NOTE=""
case "${DIND_MODE}" in
  dind)
    DIND_ARG=(
      --privileged
      -v "${DIND_VOLUME}:/var/lib/docker"
    )
    DIND_NOTE="DinD（容器内独立 dockerd, 数据卷 ${DIND_VOLUME}）"
    ;;
  dood)
    DIND_ARG=(
      -v /var/run/docker.sock:/var/run/docker.sock
      -e ENABLE_DIND=0
    )
    DIND_NOTE="DooD（挂宿主 /var/run/docker.sock）"
    ;;
  off)
    DIND_ARG=(-e ENABLE_DIND=0)
    DIND_NOTE="off（容器内 docker CLI 装着但没 daemon）"
    ;;
  *)
    echo "[error] DIND_MODE 仅支持 dind|dood|off (当前=${DIND_MODE})"; exit 1
    ;;
esac

echo "==> 选定端口: SSH=${HOST_SSH_PORT}  ALT=${HOST_ALT_PORT}"
echo "==> 免密公钥: ${AUTH_KEYS_NOTE}"
echo "==> Docker:   ${DIND_NOTE}"
set -x
docker run -itd \
  --name "${NAME}" \
  --init \
  "${GPU_ARG[@]}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${DIND_ARG[@]}" \
  -v "${FS_MOUNT}":"${FS_MOUNT}" \
  "${AUTH_KEYS_ARG[@]}" \
  -p "${HOST_SSH_PORT}":22 \
  -p "${HOST_ALT_PORT}":2222 \
  "${IMAGE}" \
  bash
set +x

# 把分配到的端口落盘，方便 Makefile 等读取
mkdir -p .runtime
cat > .runtime/${NAME}.env <<EOF
NAME=${NAME}
IMAGE=${IMAGE}
HOST_SSH_PORT=${HOST_SSH_PORT}
HOST_ALT_PORT=${HOST_ALT_PORT}
DIND_MODE=${DIND_MODE}
DIND_VOLUME=${DIND_VOLUME}
EOF

cat <<EOF

==> 容器已启动: ${NAME}
==> 进入容器:    docker exec -it -u houjue ${NAME} bash    # 日常身份
                docker exec -it ${NAME} bash               # root 调试
==> sshd:        entrypoint 已自启，直接 ssh 即可（兜底重启: make start-sshd）
==> 测 DinD:     docker exec -u houjue ${NAME} docker info
==> 端口已记录:  $(pwd)/.runtime/${NAME}.env

== SSH 连接 ==
内网:  ssh houjue@${HOST_INTERNAL_IP} -p ${HOST_SSH_PORT}
公网:  ssh -J houjue@${HOST_PUB_IP}:${HOST_PUB_PORT} houjue@127.0.0.1 -p ${HOST_SSH_PORT}

== VSCode (~/.ssh/config) ==
Host jzgpu17
    HostName ${HOST_PUB_IP}
    User houjue
    Port ${HOST_PUB_PORT}

Host houjue-dev
    HostName 127.0.0.1
    User houjue
    Port ${HOST_SSH_PORT}
    ProxyJump jzgpu17
EOF

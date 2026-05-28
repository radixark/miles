#!/bin/bash
# 容器入口：
#   1) root 身份下，从 $USER_PASSWORD 重置 $DEV_USER 密码（K8s Secret / -e 注入）
#   2) DinD 自动启动 dockerd（三种模式自动识别）：
#        - 挂了宿主 /var/run/docker.sock        -> DooD 模式，不启 dockerd
#        - 没挂 socket 且容器是 --privileged    -> 启 dockerd（DinD）
#        - 非 root 或 ENABLE_DIND=0             -> 跳过
#   3) 自动启动 sshd（root 身份）：
#        - 默认开（ENABLE_SSHD=auto/1），方便 ProxyJump 直连容器
#        - CMD 已经是 sshd 自身 / ENABLE_SSHD=0 时跳过，避免端口抢占
#   4) exec CMD（默认 bash；K8s 那边是 sshd -D -e）
set -e

# ---- 1) 密码重置（仅 root） ----
if [ "$(id -u)" = "0" ] && [ -n "${USER_PASSWORD:-}" ]; then
    echo "${DEV_USER:-houjue}:${USER_PASSWORD}" | chpasswd 2>/dev/null || true
fi
unset USER_PASSWORD

# ---- 2) DinD：按需起 dockerd ----
start_dockerd_if_needed() {
    [ "$(id -u)" = "0" ] || return 0
    [ "${ENABLE_DIND:-auto}" = "0" ] && return 0

    if [ -S /var/run/docker.sock ]; then
        # 已有宿主 socket，DooD 模式
        if [ "${ENABLE_DIND:-auto}" = "auto" ]; then
            echo "[entrypoint] detected /var/run/docker.sock -> DooD mode (skipping dockerd)"
            # 让 houjue 能访问宿主 socket：把 socket gid 改成镜像里 docker 组的 gid(3001)
            chgrp docker /var/run/docker.sock 2>/dev/null || true
            chmod g+rw /var/run/docker.sock 2>/dev/null || true
            return 0
        fi
    fi

    command -v dockerd >/dev/null 2>&1 || {
        echo "[entrypoint] dockerd not installed, skipping"
        return 0
    }

    if [ ! -x /etc/init.d/docker ]; then
        echo "[entrypoint] /etc/init.d/docker missing, skipping dockerd"
        return 0
    fi

    # cgroup v2 修复（必须在 dockerd 启动前、且根 cgroup 还几乎只有 entrypoint
    # 自己时跑）：把根里所有进程移到 /init 子 cgroup，根空了才能写 subtree_control
    # 把 io/memory 等 domain-only controllers 启用。否则 dockerd 创建的
    # /sys/fs/cgroup/docker 会变 threaded，导致 `docker run` 报：
    #   "cannot enter cgroupv2 ... with domain controllers -- it is in threaded mode"
    if [ "$(stat -fc %T /sys/fs/cgroup 2>/dev/null)" = "cgroup2fs" ]; then
        echo "[entrypoint] cgroup v2 fix: moving root procs to /init + enabling controllers"
        mkdir -p /sys/fs/cgroup/init 2>/dev/null || true
        xargs -rn1 < /sys/fs/cgroup/cgroup.procs \
            > /sys/fs/cgroup/init/cgroup.procs 2>/dev/null || true
        sed -e 's/ / +/g; s/^/+/' < /sys/fs/cgroup/cgroup.controllers \
            > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true
        echo "[entrypoint]   subtree_control: $(cat /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null)"
    fi

    # /etc/init.d/docker 是离线包里的脚本：nohup dockerd 并把 stdout 重定向到
    # /proc/1/fd/1（K8s logs 可见）；它自己的 fix_cgroups 不移进程所以效果有限，
    # 真正的 cgroup v2 修复在上面这段
    echo "[entrypoint] starting dockerd via /etc/init.d/docker start ..."
    /etc/init.d/docker start || {
        echo "[entrypoint] WARN: dockerd 启动失败。常见原因："
        echo "[entrypoint]   1. 容器没用 --privileged / securityContext.privileged"
        echo "[entrypoint]   2. 想完全禁用 DinD：传 -e ENABLE_DIND=0"
        return 0
    }

    # 探测 readiness：dockerd 起来后还要等 socket / 内部初始化
    for i in $(seq 1 20); do
        if docker info >/dev/null 2>&1; then
            echo "[entrypoint] dockerd is ready (took ${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "[entrypoint] WARN: dockerd 启动后 20s 仍未就绪，docker info 自查"
}

start_dockerd_if_needed

# ---- 3) sshd：root 身份下默认随容器自启 ----
# 之前是手动 `make start-sshd`，容易忘 -> ProxyJump 报 "channel 0: open
# failed: connect failed: Connection refused"。这里在 entrypoint 里直接拉起，
# 让 `make run` 之后 ssh 立刻可用。
#
# 跳过自启的情况：
#   - 非 root（没权限写 /var/run/sshd、绑 22）
#   - ENABLE_SSHD=0 显式关闭
#   - CMD 本身就是 sshd（K8s 模板用 `sshd -D -e`，下面 exec 会接管，不重复起）
start_sshd_if_needed() {
    [ "$(id -u)" = "0" ] || return 0
    [ "${ENABLE_SSHD:-auto}" = "0" ] && return 0

    case "${1:-}" in
        */sshd|sshd) return 0 ;;
    esac

    if ! command -v sshd >/dev/null 2>&1; then
        echo "[entrypoint] sshd not installed, skipping"
        return 0
    fi

    if ! [ -x /etc/init.d/ssh ]; then
        echo "[entrypoint] /etc/init.d/ssh missing, skipping sshd"
        return 0
    fi

    mkdir -p /var/run/sshd
    # 首次启动时 host key 可能缺失（debian/ubuntu 包正常会装，但镜像被裁剪过就没有）
    ssh-keygen -A >/dev/null 2>&1 || true

    echo "[entrypoint] starting sshd via /etc/init.d/ssh start ..."
    /etc/init.d/ssh start || echo "[entrypoint] WARN: sshd 启动失败，可手动 /etc/init.d/ssh start 自查"
}

start_sshd_if_needed "$@"

# ---- 4) 交给后面的 CMD ----
exec "$@"

# syntax=docker/dockerfile:1.6
# 九章云 H100 集群个人开发镜像（参考手册 Dockerfile.qiudelai 写法）
#
# 默认基础：NVIDIA 官方 PyTorch（CUDA 12 + PyTorch 2 + NCCL/cuDNN/Apex 等）
# 想换基础（按需）：
#   ubuntu:22.04                                  # 极简，无 CUDA
#   nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04    # 仅 CUDA，自己装 PyTorch
#   nvcr.io/nvidia/pytorch:24.10-py3 / 25.01-py3  # 其它 NGC 版本
#   harbor.unisound.ai/unisound/ms-swift:v4.0.1   # 自带 ms-swift 训练框架
#
# 用法:
#   docker build -t harbor.unisound.ai/unisound/houjue-dev:v1 -f Dockerfile.houjue .
#   docker build --build-arg BASE_IMAGE=ubuntu:22.04 -t ... -f Dockerfile.houjue .
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_IMAGE}

# 用户/用户组 — 严格对齐宿主机（jzgpu17 上 id 输出）
#   docker:3001  nlp-intern:2216  houjue:2244
ARG DOCKER_GID=3001
ARG NLP_GID=2216
ARG DOCKER_GID=3001
ARG NLP_TRAIN_GID=2239
ARG USER_NAME=yangchengyi
ARG USER_UID=2243
ARG USER_GID=2243
# 容器内 SSH 密码：镜像不写默认值，由 entrypoint 在容器启动时
# 从环境变量 USER_PASSWORD 注入并 chpasswd（≥12 位 + 字母/数字/特殊符号）
# K8s: 通过 secretKeyRef 注入；docker run: 通过 -e USER_PASSWORD=...

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# 手册必装包：pciutils iproute2 pdsh vim wget htop git language-pack-zh-hans openssh-server sudo
# 再加几个开发常用：curl ca-certificates git rsync tmux jq less tree net-tools dnsutils
# 加 iptables：dockerd 启 bridge 网络需要
# 注意：locale-gen 必须在 install locales 之后执行，否则 LANG=zh_CN.UTF-8 会回退到 C
# jzgpu17 的 archive.ubuntu.com 走外网很慢，把源换成内网 Nexus apt 代理
ARG APT_MIRROR=http://172.31.72.17:48081/repository/apt-proxy
RUN sed -i "s|http://archive.ubuntu.com/ubuntu/|${APT_MIRROR}/|g; \
            s|http://security.ubuntu.com/ubuntu/|${APT_MIRROR}/|g" \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg \
        pciutils iproute2 pdsh iptables \
        vim htop git tmux jq less tree rsync \
        net-tools dnsutils iputils-ping \
        language-pack-zh-hans locales tzdata \
        openssh-server openssh-client sudo \
    && ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo "zh_CN.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen zh_CN.UTF-8 \
    && update-locale LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8 LANGUAGE=zh_CN:zh \
    && printf 'export LANG=zh_CN.UTF-8\nexport LC_ALL=zh_CN.UTF-8\nexport LANGUAGE=zh_CN:zh\n' \
        > /etc/profile.d/lang.sh \
    && rm -rf /var/lib/apt/lists/*

# 显式覆盖：locales 装好后才 ENV，确保子进程 / SSH 会话都是中文
ENV LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8 \
    LANGUAGE=zh_CN:zh
ARG APT_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    python -m pip install --upgrade pip
# Python 工具：gpustat（彩色 GPU 状态）
RUN pip install --no-cache-dir gpustat

# === Docker-in-Docker：完全离线安装 ===
# jzgpu17 所在网段对 download.docker.com / nvidia.github.io 的 TLS 流量会被
# 中间盒 RST，所以放弃外网 apt 源，改用仓库内 offline/ 归档的离线包：
#   offline/docker-pkg/   docker 28.0.1 全套二进制（dockerd/containerd/runc/...）
#   offline/debian-gpu/   nvidia-container-toolkit 1.19.0 deb 包
#   offline/docker-buildx / docker-compose  插件二进制
#   offline/docker.initd  cgroup v2 自适应的 /etc/init.d/docker 启动脚本
# 这些文件已实拷贝归档（独立 inode），不依赖任何外部目录。
RUN mkdir -p /etc/docker /usr/libexec/docker/cli-plugins /var/run
COPY offline/docker-pkg/  /usr/bin/
COPY offline/debian-gpu/  /opt/debian-gpu/
RUN dpkg -i /opt/debian-gpu/*.deb && rm -rf /opt/debian-gpu
COPY offline/docker-buildx   /usr/libexec/docker/cli-plugins/docker-buildx
COPY offline/docker-compose  /usr/libexec/docker/cli-plugins/docker-compose
COPY offline/docker-compose  /usr/bin/docker-compose
COPY offline/docker.initd    /etc/init.d/docker
RUN chmod +x /usr/bin/docker /usr/bin/dockerd /usr/bin/containerd \
             /usr/bin/containerd-shim-runc-v2 /usr/bin/ctr /usr/bin/runc \
             /usr/bin/docker-init /usr/bin/docker-proxy /usr/bin/docker-compose \
             /etc/init.d/docker \
             /usr/libexec/docker/cli-plugins/docker-*

# /etc/docker/daemon.json：overlay2 + nvidia runtime + 内部 registry mirror
COPY docker-daemon.json /etc/docker/daemon.json

# /var/lib/docker 必须挂 volume；run 时用 named volume / emptyDir 顶上来
VOLUME /var/lib/docker

# sshd 配置（手册同款）：跨节点免密交互更顺畅
RUN mkdir -p /var/run/sshd \
    && sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config \
    && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# 创建用户/组（镜像里不设密码，账号默认是锁定状态）
# - sudo 已 NOPASSWD，无需密码
# - SSH 公钥免密由挂载/Secret 处理
# - SSH 密码登录依赖 entrypoint 在启动时 chpasswd
# 注意：docker-ce 安装时会自动建 docker 组（动态 GID），这里强制 modify 成
#       宿主 DOCKER_GID=3001，否则 houjue 在容器里看到 docker 组 gid 跟宿主对不上
RUN (groupmod -g ${DOCKER_GID} docker 2>/dev/null \
        || groupadd -g ${DOCKER_GID} docker) \
    && groupadd -f -g ${NLP_GID} nlp-intern \
    && groupadd -f -g ${NLP_TRAIN_GID} nlp-train \
    && groupadd -f -g ${USER_GID} ${USER_NAME} \
    && useradd -m -u ${USER_UID} -g ${USER_NAME} -G nlp-intern,nlp-train,docker \
        -s /bin/bash ${USER_NAME} \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && install -d -m 700 -o ${USER_NAME} -g ${USER_NAME} /home/${USER_NAME}/.ssh \
    && touch /home/${USER_NAME}/.ssh/authorized_keys \
    && chown ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.ssh/authorized_keys \
    && chmod 600 /home/${USER_NAME}/.ssh/authorized_keys

# entrypoint：每次启动以 root 跑——做两件事：
#   1) 从 $USER_PASSWORD 重置 $DEV_USER 密码（K8s Secret / -e 注入）
#   2) 必要时启动 dockerd（DinD 模式）
# 之前末尾切到 USER houjue 会让这两件事都做不了——所以保留 root 主进程，
# 日常用 `docker exec -u houjue` / `su - houjue` 切普通账号。
ENV DEV_USER=${USER_NAME}
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY server.py /home/${USER_NAME}/server.py
RUN chmod +x /usr/local/bin/entrypoint.sh
COPY init-home-symlinks.sh /usr/local/bin/init-home-symlinks.sh
RUN chmod +x /usr/local/bin/init-home-symlinks.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# ENTRYPOINT ["/usr/local/bin/init-home-symlinks.sh"]
CMD ["bash"]

WORKDIR /home/${USER_NAME}

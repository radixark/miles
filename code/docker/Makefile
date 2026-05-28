# 九章云 H100 个人开发流程（jzgpu17 / nlp-intern / houjue）
# 用法: make help

SHELL := /bin/bash
.DEFAULT_GOAL := help
.ONESHELL:

# ===== Harbor 镜像 =====
REGISTRY    ?= harbor.unisound.ai
NAMESPACE   ?= unisound
IMAGE_NAME  ?= houjue-dev
IMAGE_TAG   ?= v$(shell date +%Y%m%d)
BASE_IMAGE  ?= nvcr.io/nvidia/pytorch:26.04-py3
HARBOR_USER ?= admin
HARBOR_PASS ?= Harbor12345
FULL_IMAGE  := $(REGISTRY)/$(NAMESPACE)/$(IMAGE_NAME):$(IMAGE_TAG)

# 离线 docker 包：必须存在以下文件，否则 build 会报错（jzgpu17 网段无法直连
# download.docker.com / nvidia.github.io 装 docker，所以靠这套预下载二进制）
DIND_OFFLINE_FILES := \
    offline/docker-pkg/dockerd \
    offline/docker-pkg/containerd \
    offline/docker-pkg/runc \
    offline/debian-gpu/nvidia-container-toolkit_1.19.0-1_amd64.deb \
    offline/docker-buildx \
    offline/docker-compose \
    offline/docker.initd

# ===== 调试容器：HOST_SSH_PORT=auto 让 run 时自动选空闲端口 =====
# 分配结果会写入 .runtime/$(CONTAINER).env
CONTAINER       ?= houjue-dev
HOST_SSH_PORT   ?= auto
HOST_ALT_PORT   ?= auto
PORT_LO         ?= 60020
PORT_HI         ?= 60099
FS_MOUNT        ?= /fs
GPUS            ?= all
# DinD: dind | dood | off （详见 examples/run-debug-container.sh）
DIND_MODE       ?= dind
DIND_VOLUME     ?= $(CONTAINER)-docker

# ===== jzgpu17 网络入口 =====
# 内网 SSH： ssh houjue@172.31.72.17  ；公网 SSH： ssh houjue@120.220.102.24 -p 60017
HOST_INTERNAL_IP ?= 172.31.72.17
HOST_PUB_IP      ?= 120.220.102.24
HOST_PUB_PORT    ?= 60017

# ===== Kubernetes (训练 / 推理) =====
K8S_NS      ?= nlp-intern
KUBECTL     ?= kubectl
MPI_FILE    ?= examples/mpijob-houjue.yaml

export REGISTRY NAMESPACE IMAGE_NAME IMAGE_TAG BASE_IMAGE HARBOR_USER HARBOR_PASS

help:  ## 列出所有命令
	@grep -hE '^[a-zA-Z0-9_.-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "镜像:        $(FULL_IMAGE)"
	@echo "调试容器:    $(CONTAINER)  (SSH -> $(HOST_SSH_PORT))"
	@echo "K8s ns:      $(K8S_NS)"

# ===== 镜像 =====
login:        ## docker login Harbor
	./build.sh login

offline:      ## 校验 offline/ 离线 docker 包齐全（不存在则报错）
	@missing=0; \
	for f in $(DIND_OFFLINE_FILES); do \
	  [ -e "$$f" ] || { echo "  [missing] $$f"; missing=1; }; \
	done; \
	if [ $$missing -ne 0 ]; then \
	  echo ""; \
	  echo "[error] offline/ 离线包不完整。这套二进制已随仓库归档在 offline/，"; \
	  echo "        如果误删，可从历史 git 或团队备份恢复。完整清单："; \
	  for f in $(DIND_OFFLINE_FILES); do echo "          $$f"; done; \
	  exit 1; \
	fi; \
	echo "==> offline/ OK ($$(du -sh offline/ | cut -f1))"

build: offline ## 构建镜像 (Dockerfile.houjue，先校验 offline/)
	./build.sh build

push:         ## 推送当前 tag
	./build.sh push

build-push: offline ## build + push
	./build.sh build-push

print-image:  ## 打印当前镜像信息
	./build.sh print

# ===== 调试容器（在 jzgpu17 上） =====
run:          ## 启动调试容器（自动选空闲端口；默认 DinD 模式）
	IMAGE=$(FULL_IMAGE) NAME=$(CONTAINER) \
	  HOST_SSH_PORT=$(HOST_SSH_PORT) HOST_ALT_PORT=$(HOST_ALT_PORT) \
	  PORT_LO=$(PORT_LO) PORT_HI=$(PORT_HI) \
	  HOST_INTERNAL_IP=$(HOST_INTERNAL_IP) \
	  HOST_PUB_IP=$(HOST_PUB_IP) HOST_PUB_PORT=$(HOST_PUB_PORT) \
	  FS_MOUNT=$(FS_MOUNT) GPUS=$(GPUS) \
	  DIND_MODE=$(DIND_MODE) DIND_VOLUME=$(DIND_VOLUME) \
	  bash examples/run-debug-container.sh

start-sshd:   ## 兜底：手动启动容器内 sshd（entrypoint 已自启，正常用不到）
	docker exec -u 0 $(CONTAINER) /etc/init.d/ssh start

ps:           ## 看自己容器
	docker ps -a --filter name=$(CONTAINER)

logs:         ## 调试容器日志
	docker logs --tail=200 $(CONTAINER)

exec:         ## 进入容器（houjue 身份）
	docker exec -it -u houjue $(CONTAINER) bash

exec-root:    ## 进入容器（root 身份调试）
	docker exec -it -u 0 $(CONTAINER) bash

dind-info:    ## 检查容器内 dockerd 状态
	@echo "== docker info（houjue 身份）=="; \
	docker exec -u houjue $(CONTAINER) docker info 2>&1 | head -30; \
	echo ""; echo "== dockerd 日志最后 20 行 =="; \
	docker exec -u 0 $(CONTAINER) tail -20 /var/log/dockerd.log 2>/dev/null \
	  || echo "（无 /var/log/dockerd.log，可能不是 DinD 模式）"

dind-test:    ## 容器内拉 hello-world 跑一遍验证 DinD 可用
	docker exec -u houjue $(CONTAINER) docker run --rm hello-world

dind-logs:    ## 跟随容器内 dockerd 日志
	docker exec -u 0 -it $(CONTAINER) tail -f /var/log/dockerd.log

stop:         ## 停止调试容器
	-docker stop $(CONTAINER)

rm:           ## 删除调试容器（不删 DinD 数据卷）
	-docker rm -f $(CONTAINER)

rm-all:       ## 删除调试容器 + DinD 数据卷（彻底清理）
	-docker rm -f $(CONTAINER)
	-docker volume rm $(DIND_VOLUME)

commit:       ## 把当前容器 commit 成新镜像（手册推荐：稳定后保存）
	@read -p "新 tag (默认 $(IMAGE_TAG)-snapshot): " t; t=$${t:-$(IMAGE_TAG)-snapshot}; \
	  docker commit $(CONTAINER) $(REGISTRY)/$(NAMESPACE)/$(IMAGE_NAME):$$t && \
	  echo "==> 已生成 $(REGISTRY)/$(NAMESPACE)/$(IMAGE_NAME):$$t  请手动 docker push"

ssh-info:     ## 打印 SSH 连接信息（自动从 docker port 读取实际端口）
	@p=$$(docker port $(CONTAINER) 22/tcp 2>/dev/null | head -1 | awk -F: '{print $$NF}'); \
	if [ -z "$$p" ]; then \
	  echo "[warn] 容器 $(CONTAINER) 未运行；先 make run"; \
	  exit 0; \
	fi; \
	echo "登录调试机宿主:"; \
	echo "  公网:  ssh houjue@$(HOST_PUB_IP) -p $(HOST_PUB_PORT)"; \
	echo "  内网:  ssh houjue@$(HOST_INTERNAL_IP)"; \
	echo ""; \
	echo "登录调试容器（容器 22 -> 宿主 $$p）:"; \
	echo "  内网:  ssh houjue@$(HOST_INTERNAL_IP) -p $$p"; \
	echo "  公网:  ssh -J houjue@$(HOST_PUB_IP):$(HOST_PUB_PORT) houjue@127.0.0.1 -p $$p"

# ===== Kubernetes 训练任务 =====
mpi-apply:    ## kubectl apply MPIJob
	$(KUBECTL) apply -f $(MPI_FILE)

mpi-status:   ## 查看 MPIJob 与 Pod
	@echo "== MPIJob =="; $(KUBECTL) get mpijob -n $(K8S_NS)
	@echo "== Pods ==";   $(KUBECTL) get po -n $(K8S_NS) -o wide

mpi-logs:     ## 查看 launcher 日志（POD=xxx 指定 pod 名）
	@p="$(POD)"; [ -z "$$p" ] && p=$$($(KUBECTL) get po -n $(K8S_NS) -l training.kubeflow.org/job-role=launcher -o name | head -1); \
	  echo ">> $$p"; $(KUBECTL) logs -n $(K8S_NS) $$p -f

mpi-delete:   ## 删除 MPIJob
	-$(KUBECTL) delete -f $(MPI_FILE) --ignore-not-found

quota:        ## 查看 nlp-intern 配额
	$(KUBECTL) describe resourcequota -n $(K8S_NS)

.PHONY: help login offline build push build-push print-image \
        run start-sshd ps logs exec exec-root stop rm rm-all commit ssh-info \
        dind-info dind-test dind-logs \
        mpi-apply mpi-status mpi-logs mpi-delete quota

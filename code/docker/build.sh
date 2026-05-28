#!/usr/bin/env bash
# 构建 / 推送个人开发镜像到 Harbor
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REGISTRY="${REGISTRY:-harbor.unisound.ai}"
NAMESPACE="${NAMESPACE:-unisound}"
IMAGE_NAME="${IMAGE_NAME:-houjue-dev}"
IMAGE_TAG="${IMAGE_TAG:-v$(date +%Y%m%d)}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.04-py3}"
HARBOR_USER="${HARBOR_USER:-admin}"
HARBOR_PASS="${HARBOR_PASS:-Harbor12345}"

FULL_IMAGE="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

usage() {
  cat <<EOF
用法: $0 <command>

命令:
  login          docker login Harbor (默认 admin/Harbor12345)
  build          docker build -f Dockerfile.houjue
  push           docker push 当前 tag
  build-push     build + push
  print          打印镜像信息
  clean          删除本地镜像

环境变量（可在调用前 export）:
  REGISTRY     默认 harbor.unisound.ai
  NAMESPACE    默认 unisound
  IMAGE_NAME   默认 houjue-dev
  IMAGE_TAG    默认 v\$(date +%Y%m%d)，注意手册要求不同版本不要重名
  BASE_IMAGE   默认 nvcr.io/nvidia/pytorch:26.04-py3
               其它常用：ubuntu:22.04 / nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /
                          harbor.unisound.ai/unisound/ms-swift:v4.0.1
  HARBOR_USER  默认 admin
  HARBOR_PASS  默认 Harbor12345

示例:
  ./build.sh build-push
  IMAGE_TAG=v20260514 ./build.sh build-push
  BASE_IMAGE=ubuntu:22.04 ./build.sh build           # 极简基础（无 CUDA）
  BASE_IMAGE=harbor.unisound.ai/unisound/ms-swift:v4.0.1 ./build.sh build  # 自带训练框架
EOF
}

cmd_print() {
  echo "FULL_IMAGE = ${FULL_IMAGE}"
  echo "BASE_IMAGE = ${BASE_IMAGE}"
}

cmd_login() {
  echo "${HARBOR_PASS}" | docker login "${REGISTRY}" -u "${HARBOR_USER}" --password-stdin
}

cmd_build() {
  echo ">>> docker build ${FULL_IMAGE}"
  docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${FULL_IMAGE}" \
    -f Dockerfile.houjue \
    .
}

cmd_push() {
  echo ">>> docker push ${FULL_IMAGE}"
  docker push "${FULL_IMAGE}"
}

cmd_clean() {
  docker rmi "${FULL_IMAGE}" 2>/dev/null || true
}

case "${1:-}" in
  login)        cmd_login ;;
  build)        cmd_build ;;
  push)         cmd_push ;;
  build-push)   cmd_build; cmd_push ;;
  print)        cmd_print ;;
  clean)        cmd_clean ;;
  ""|-h|--help) usage ;;
  *) echo "未知命令: $1"; usage; exit 1 ;;
esac

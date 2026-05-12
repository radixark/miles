#!/usr/bin/env bash
# 在 Pod 挂载 volume 之后执行（由 lifecycle.postStart 调用）。
# 与本机家目录一致：~/data -> /fs/nlp-intern/yangchengyi/
#
# 环境变量（可在 MPIJob YAML 的 env 里覆盖）：
#   HOME_SYMLINK_TARGET  默认 /home/yangchengyi
#   DATA_LINK_SRC        默认 /fs/nlp-intern/yangchengyi

set +e

H="${HOME_SYMLINK_TARGET:-/home/yangchengyi}"
DATA_SRC="${DATA_LINK_SRC:-/fs/nlp-intern/yangchengyi}"

# 若误存在同名空目录（旧脚本 mkdir 过），删掉才能建名为 data 的软链
if [[ -d "${H}/data" && ! -L "${H}/data" ]]; then
  rmdir "${H}/data" 2>/dev/null || true
fi

ln -sfn "${DATA_SRC}" "${H}/data"
chown -h yangchengyi:yangchengyi "${H}/data" 2>/dev/null || true
exit 0

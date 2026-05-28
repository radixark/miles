#!/usr/bin/env bash
# Pod / 调试机启动后初始化家目录软链（需 root 或 sudo 执行）。
#
# 1) ~/data -> /fs/nlp-intern/yangchengyi/
# 2) ~/.local / ~/.cache -> /fs 上共享目录（uv Python + cache，跨节点/Pod 可用）
# 3) dotfiles（.bashrc 等）-> /fs/.../dotfiles/（跨节点/Pod 共享用户配置）
#
# 环境变量:
#   HOME_SYMLINK_TARGET  默认 /home/yangchengyi
#   DATA_LINK_SRC        默认 /fs/nlp-intern/yangchengyi
#   DOTFILES_SRC         默认 ${DATA_LINK_SRC}/dotfiles
#   RUN_USER             默认 yangchengyi
#   DOTFILES_LIST        空格分隔，默认见下方

set +e

H="${HOME_SYMLINK_TARGET:-/home/yangchengyi}"
DATA_SRC="${DATA_LINK_SRC:-/fs/nlp-intern/yangchengyi}"
DOTFILES_SRC="${DOTFILES_SRC:-${DATA_SRC}/dotfiles}"
RUN_USER="${RUN_USER:-yangchengyi}"
DOTFILES_LIST="${DOTFILES_LIST:-.bashrc .bash_profile .profile .gitconfig .tmux.conf}"

FS_LOCAL="${DATA_SRC}/.local"
FS_CACHE="${DATA_SRC}/.cache"

mkdir -p "${FS_LOCAL}/share/uv" "${FS_CACHE}/uv" "${DOTFILES_SRC}"
chown -R "${RUN_USER}:${RUN_USER}" "${FS_LOCAL}" "${FS_CACHE}" "${DOTFILES_SRC}" 2>/dev/null || true

link_if_needed() {
  local name="$1" target="$2"
  local path="${H}/${name}"
  if [[ -L "$path" ]]; then
    ln -sfn "${target}" "$path"
  elif [[ ! -e "$path" ]]; then
    ln -sfn "${target}" "$path"
  fi
  chown -h "${RUN_USER}:${RUN_USER}" "$path" 2>/dev/null || true
}

# ~/data 软链
if [[ -d "${H}/data" && ! -L "${H}/data" ]]; then
  rmdir "${H}/data" 2>/dev/null || true
fi
ln -sfn "${DATA_SRC}" "${H}/data"
chown -h "${RUN_USER}:${RUN_USER}" "${H}/data" 2>/dev/null || true

# ~/.local / ~/.cache -> /fs（uv 的 Python 与 cache 跨 Pod 共享）
link_if_needed ".local" "${FS_LOCAL}"
link_if_needed ".cache" "${FS_CACHE}"

# dotfiles -> /fs/.../dotfiles（仅当目标文件存在且 home 下无实体文件时链接）
for f in ${DOTFILES_LIST}; do
  [[ -e "${DOTFILES_SRC}/${f}" ]] || continue
  path="${H}/${f}"
  if [[ -L "$path" ]] || [[ ! -e "$path" ]]; then
    ln -sfn "${DOTFILES_SRC}/${f}" "$path"
    chown -h "${RUN_USER}:${RUN_USER}" "$path" 2>/dev/null || true
  fi
done

# 让 yangchengyi 能在 home 下创建文件（Pod 里 home 有时属 root）
chown "${RUN_USER}:${RUN_USER}" "${H}" 2>/dev/null || true

exit 0

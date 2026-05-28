#!/usr/bin/env bash
# 一次性初始化：把用户配置放到 /fs 共享盘，并在调试机 home 建立软链。
# 用法: bash setup-fs-dotfiles.sh [--force]
#
# 之后请在 /fs/nlp-intern/yangchengyi/dotfiles/ 里维护 .bashrc 等配置；
# 调试机和 K8s Pod 启动时都会链到同一份文件。

set -euo pipefail

DATA_SRC="${DATA_LINK_SRC:-/fs/nlp-intern/yangchengyi}"
DOTFILES_SRC="${DOTFILES_SRC:-${DATA_SRC}/dotfiles}"
H="${HOME_SYMLINK_TARGET:-/home/yangchengyi}"
FORCE="${1:-}"

DOTFILES=(.bashrc .bash_profile .profile .gitconfig .tmux.conf)

mkdir -p "${DOTFILES_SRC}"

for f in "${DOTFILES[@]}"; do
  src="${H}/${f}"
  dst="${DOTFILES_SRC}/${f}"
  if [[ ! -e "$dst" ]]; then
    if [[ -e "$src" && ! -L "$src" ]]; then
      cp -a "$src" "$dst"
      echo "copied $src -> $dst"
    elif [[ -L "$src" ]]; then
      real=$(readlink -f "$src" || true)
      if [[ -n "$real" && -e "$real" ]]; then
        cp -a "$real" "$dst"
        echo "copied (from symlink) $real -> $dst"
      fi
    else
      echo "skip $f: not found in $H"
    fi
  elif [[ "$FORCE" == "--force" && -e "$src" && ! -L "$src" ]]; then
    cp -a "$src" "$dst"
    echo "force updated $dst from $src"
  else
    echo "keep existing $dst"
  fi
done

# 调试机：~/data 软链 + dotfiles 软链（无需 root）
if [[ -d "${H}/data" && ! -L "${H}/data" ]]; then
  echo "warn: ${H}/data is a real directory, not touching"
else
  ln -sfn "${DATA_SRC}" "${H}/data"
  echo "linked ${H}/data -> ${DATA_SRC}"
fi

for f in "${DOTFILES[@]}"; do
  [[ -e "${DOTFILES_SRC}/${f}" ]] || continue
  path="${H}/${f}"
  if [[ -e "$path" && ! -L "$path" ]]; then
    if [[ "$FORCE" == "--force" ]]; then
      rm -f "$path"
    else
      echo "skip link $path: real file exists (use --force to replace with symlink)"
      continue
    fi
  fi
  ln -sfn "${DOTFILES_SRC}/${f}" "$path"
  echo "linked $path -> ${DOTFILES_SRC}/${f}"
done

echo ""
echo "Done. Edit dotfiles at: ${DOTFILES_SRC}/"
echo "Work directory: ${DATA_SRC}/  (~/data)"

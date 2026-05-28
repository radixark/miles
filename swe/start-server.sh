#!/usr/bin/env bash
# 启动 Harbor agent-env server
#
# 依赖：在宿主机以 yangchengyi 执行 uv sync。
# Pod 里 .venv 在 /fs 上可见，但 .venv/bin/python 常是指向
# ~/.local/share/uv/python/... 的 symlink；若 Pod 所在节点没有 uv Python，会自动 fallback 到 uv run。
#
# 用法:
#   ./start-server.sh              # 前台运行
#   ./start-server.sh -d           # 后台运行
#   ./start-server.sh -d --stop    # 停旧进程后后台启动（推荐）
#   ./start-server.sh --check      # 只校验环境，不启动
#
# 环境变量（可选）:
#   HARBOR_DIR / USER_HOME / SERVER_PY / PORT / MAX_CONCURRENT / OPENAI_API_KEY / HARBOR_TASKS_DIR / LOG_FILE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_USER="${RUN_USER:-yangchengyi}"
USER_HOME="${USER_HOME:-/home/${RUN_USER}}"

SERVER_PY="${SERVER_PY:-${SCRIPT_DIR}/server.py}"
PORT="${PORT:-11000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
LOG_FILE="${LOG_FILE:-/tmp/server.log}"
HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-/fs/nlp-intern/yangchengyi/harbor/datasets/swegym}"
FS_DATA="${FS_DATA:-/fs/nlp-intern/yangchengyi}"

setup_uv_paths() {
  # uv 的 Python/cache 放在 /fs，与 Pod 内 init-home-symlinks.sh 的 ~/.local ~/.cache 一致
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${FS_DATA}/.cache/uv}"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${FS_DATA}/.local/share/uv/python}"
  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" 2>/dev/null || true
}

DAEMON=0
STOP_OLD=0
CHECK_ONLY=0
RUNNER_MODE="direct"   # direct | uv
PYTHON=""

usage() {
  sed -n '2,15p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--daemon) DAEMON=1; shift ;;
    --stop) STOP_OLD=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

resolve_harbor_dir() {
  if [[ -n "${HARBOR_DIR:-}" ]]; then
    echo "$HARBOR_DIR"
    return
  fi
  echo "/fs/nlp-intern/${RUN_USER}/harbor_project/harbor"
}

venv_python() {
  echo "${HARBOR_DIR}/.venv/bin/python"
}

ensure_uv() {
  setup_uv_paths
  if [[ -f "${USER_HOME}/.local/bin/env" ]]; then
    # shellcheck disable=SC1091
    source "${USER_HOME}/.local/bin/env"
  fi
  command -v uv >/dev/null 2>&1
}

warn_if_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    echo "WARN: 当前是 root。建议: su - ${RUN_USER} -c '$0 $*'" >&2
  fi
}

diagnose_broken_python_symlink() {
  local py="$1"
  if [[ -L "$py" && ! -e "$py" ]]; then
    echo "FAIL: .venv 目录存在，但 bin/python 是断链 symlink:" >&2
    ls -la "$py" >&2 || true
    echo "  指向: $(readlink "$py" 2>/dev/null || echo '?')" >&2
    echo "  原因: .venv 在 /fs 共享，但 uv 安装的 Python 在 ${USER_HOME}/.local/share/uv/" >&2
    echo "  当前 Pod/节点上可能还没有这份 Python。" >&2
    return 0
  fi
  if [[ ! -e "$py" ]]; then
    echo "FAIL: 未找到 ${py}" >&2
  else
    echo "FAIL: ${py} 存在但不可执行" >&2
    ls -la "$py" >&2 || true
  fi
}

resolve_python_runner() {
  local py venv_dir
  py="$(venv_python)"
  venv_dir="${HARBOR_DIR}/.venv"

  if [[ ! -d "$venv_dir" ]]; then
    echo "FAIL: 未找到 ${venv_dir}" >&2
    echo "请在宿主机以 ${RUN_USER} 执行: cd ${HARBOR_DIR} && uv sync" >&2
    exit 1
  fi

  local venv_owner
  venv_owner="$(stat -c '%U' "$venv_dir" 2>/dev/null || echo unknown)"
  if [[ "$venv_owner" == "root" ]]; then
    echo "FAIL: .venv 属主是 root" >&2
    echo "修复: sudo chown -R ${RUN_USER}:${RUN_USER} ${venv_dir}" >&2
    exit 1
  fi

  if [[ -x "$py" ]]; then
    RUNNER_MODE="direct"
    PYTHON="$py"
    echo "OK  runner=direct  python=${py}"
    return 0
  fi

  diagnose_broken_python_symlink "$py"

  if ensure_uv; then
    RUNNER_MODE="uv"
    PYTHON="uv"
    echo "OK  runner=uv  (将通过 cd ${HARBOR_DIR} && uv run python ...)"
    echo "提示: 若首次在此 Pod/节点运行，uv 可能会下载 Python 到 ${USER_HOME}/.local/share/uv/" >&2
    return 0
  fi

  echo "修复（以 ${RUN_USER} 在 Pod 或宿主机执行一次）:" >&2
  echo "  cd ${HARBOR_DIR} && uv sync" >&2
  exit 1
}

run_python() {
  setup_uv_paths
  if [[ "$RUNNER_MODE" == uv ]]; then
    (cd "$HARBOR_DIR" && uv run python "$@")
  else
    "$PYTHON" "$@"
  fi
}

verify_env() {
  local harbor_dir_real
  harbor_dir_real="$(cd "$HARBOR_DIR" && pwd)"

  if ! run_python - <<PY
import sys
from pathlib import Path

harbor_dir = Path("${harbor_dir_real}").resolve()
py = Path(sys.executable).resolve()

import harbor
import fastapi
import uvicorn

harbor_file = Path(harbor.__file__).resolve()
src_dir = harbor_dir / "src" / "harbor"
if not str(harbor_file).startswith(str(src_dir)):
    raise SystemExit(
        f"harbor 包不在 {src_dir} 下（可能装错环境）: {harbor_file}"
    )

ver = getattr(harbor, "__version__", None)
print(f"OK  python={py}")
print(f"OK  harbor={harbor_file}")
print(f"OK  harbor_version={ver or 'unknown'}")
print(f"OK  fastapi={fastapi.__version__}  uvicorn={uvicorn.__version__}")
PY
  then
    echo "FAIL: Python 环境校验未通过（见上方 Python 报错）" >&2
    exit 1
  fi
}

stop_existing() {
  local pids
  pids="$(pgrep -f "[p]ython.*server.py.*--port ${PORT}" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "停止已有 server 进程: $pids"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

wait_for_health() {
  local i
  for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "健康检查通过: http://127.0.0.1:${PORT}/health"
      return 0
    fi
    sleep 1
  done
  echo "WARN: 30s 内 /health 未就绪，请查看日志: ${LOG_FILE}" >&2
  return 1
}

launch_server() {
  setup_uv_paths
  local -a server_args=(
    "$SERVER_PY"
    --host 0.0.0.0
    --port "$PORT"
    --max-concurrent "$MAX_CONCURRENT"
  )

  if [[ "$RUNNER_MODE" == uv ]]; then
    if [[ "$DAEMON" -eq 1 ]]; then
      nohup bash -c "cd \"${HARBOR_DIR}\" && exec uv run python $(printf '%q ' "${server_args[@]}")" \
        >"$LOG_FILE" 2>&1 &
      echo "server 已在后台启动 (pid=$!, log=${LOG_FILE})"
      wait_for_health || true
    else
      cd "$HARBOR_DIR"
      exec uv run python "${server_args[@]}"
    fi
  else
    if [[ "$DAEMON" -eq 1 ]]; then
      nohup "$PYTHON" "${server_args[@]}" >"$LOG_FILE" 2>&1 &
      echo "server 已在后台启动 (pid=$!, log=${LOG_FILE})"
      wait_for_health || true
    else
      exec "$PYTHON" "${server_args[@]}"
    fi
  fi
}

HARBOR_DIR="$(resolve_harbor_dir)"
if [[ ! -f "${HARBOR_DIR}/pyproject.toml" ]]; then
  echo "找不到 harbor 项目: ${HARBOR_DIR}" >&2
  exit 1
fi
if [[ ! -f "$SERVER_PY" ]]; then
  echo "找不到 server.py: $SERVER_PY" >&2
  exit 1
fi

warn_if_root "$@"

echo "=== 环境校验 (${HARBOR_DIR}) ==="
resolve_python_runner
verify_env

export OPENAI_API_KEY
export HARBOR_TASKS_DIR

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "=== 校验通过，未启动 server ==="
  exit 0
fi

if [[ "$STOP_OLD" -eq 1 ]]; then
  stop_existing
fi

echo "=== 启动参数 ==="
echo "HARBOR_DIR=${HARBOR_DIR}"
echo "RUNNER_MODE=${RUNNER_MODE}"
echo "PYTHON=${PYTHON}"
echo "SERVER_PY=${SERVER_PY}"
echo "PORT=${PORT}  MAX_CONCURRENT=${MAX_CONCURRENT}"
echo "HARBOR_TASKS_DIR=${HARBOR_TASKS_DIR}"

launch_server

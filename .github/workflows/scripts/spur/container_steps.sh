#!/usr/bin/env bash
# The ROCm CI steps of _run-ci-rocm.yml, run inside the CI container that
# job.sh starts on the allocated node. The checkout is mounted at
# $GITHUB_WORKSPACE (/miles-checkout).
#
# Environment (passed through from the workflow step by launch.sh / job.sh):
#   CI_COMMAND                  the suite command (run_suite.py ...)
#   CI_SKIP_DEPENDENCY_INSTALL  "true" keeps the SGLang / Megatron-LM baked
#                               into the image
#   SGLANG_PR, MEGATRON_PR      refs to fetch over the baked copies (empty: keep)
#   BRIDGE_REF                  Megatron Bridge ref to install (empty: keep)

set -euo pipefail
: "${CI_COMMAND:?}" "${GITHUB_WORKSPACE:?}"
cd "$GITHUB_WORKSPACE"
# The checkout belongs to the runner account, not root; without this git
# refuses the repo and setuptools-scm cannot resolve the package version.
git config --global --add safe.directory "$GITHUB_WORKSPACE"

echo "::group::Wait for GPU ready"
# The job's GPUs can still be releasing a previous job's context. Bounded so
# a genuinely broken GPU fails fast.
ready=0
for i in $(seq 1 60); do
  if rocm-smi --showproductname >/dev/null 2>&1; then
    echo "GPU ready after $((i-1)) retry(s)"
    rocm-smi --showproductname 2>&1 | head -20 || true
    ready=1
    break
  fi
  sleep 2
done
if (( ! ready )); then
  echo "ERROR: rocm-smi did not become ready after 120s" >&2
  rocm-smi 2>&1 | tail -20 || true
  exit 1
fi
echo "::endgroup::"

echo "::group::Install dependencies"
if [[ "${CI_SKIP_DEPENDENCY_INSTALL:-true}" != "true" ]]; then
  if command -v rustup >/dev/null 2>&1; then
    rustup default stable
  fi
  if ! command -v protoc >/dev/null 2>&1; then
    apt-get update
    apt-get install -y protobuf-compiler
  fi
  rustc --version
  protoc --version
  if [ -n "${SGLANG_PR:-}" ]; then
    cd /sgl-workspace/sglang && git reset --hard HEAD && git clean -fd && git fetch origin "$SGLANG_PR" && git checkout -f FETCH_HEAD && git log --oneline -1 && pip install -e python --no-deps --break-system-packages && \
      pip install -e radix_transfer_plugins --no-deps --break-system-packages
    # SGLang builds its Rust extensions on demand under rust/, which pins a
    # toolchain the image lacks. Install it now so the server's first
    # `cargo --version` does not fail.
    if [ -f rust/rust-toolchain.toml ]; then
      (cd rust && cargo --version --verbose)
    fi
  fi
  if [ -n "${MEGATRON_PR:-}" ]; then
    cd /root/Megatron-LM
    git reset --hard HEAD
    git clean -fd
    git fetch origin "$MEGATRON_PR"
    git checkout -f FETCH_HEAD
    git log --oneline -1
    pip install -e . --no-deps --break-system-packages
  fi
fi
cd "$GITHUB_WORKSPACE" && pip install -e . --no-deps --break-system-packages
pip install polars zstandard xxhash blake3 "numba==0.65.1" --break-system-packages
echo "::endgroup::"

if [ -n "${BRIDGE_REF:-}" ]; then
  echo "::group::Install Megatron Bridge override"
  # Same install and check as .github/actions/install-megatron-bridge, which
  # cannot run here: it checks out with actions/checkout on the runner host.
  BRIDGE_DIR="$GITHUB_WORKSPACE/Megatron-Bridge"
  rm -rf "$BRIDGE_DIR"
  git clone --filter=blob:none https://github.com/radixark/Megatron-Bridge.git "$BRIDGE_DIR"
  git -C "$BRIDGE_DIR" fetch origin "$BRIDGE_REF"
  git -C "$BRIDGE_DIR" checkout -f FETCH_HEAD
  git -C "$BRIDGE_DIR" log -1 --format='Bridge commit: %H'
  python -m pip install --no-deps --no-build-isolation --force-reinstall "$BRIDGE_DIR"
  python - <<'PY'
import json
import os
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path

package = distribution("megatron-bridge")
checkout = Path(os.environ["GITHUB_WORKSPACE"]) / "Megatron-Bridge"
assert json.loads(package.read_text("direct_url.json"))["url"] == checkout.resolve().as_uri()
expected = Path(package.locate_file("megatron/bridge/__init__.py")).resolve()
actual = Path(find_spec("megatron.bridge").origin).resolve()
assert actual == expected, f"megatron.bridge resolved to {actual}, expected {expected}"
print(f"megatron-bridge {package.version}: {actual}")
PY
  echo "::endgroup::"
fi

echo "::group::Load one HIP runtime"
torch_lib="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
# Torch wheels that bundle their own copy get it pointed at /opt/rocm's; the
# ROCm 10 image's torch bundles none and already uses /opt/rocm's.
for lib in libamdhip64.so libhsa-runtime64.so; do
  if [ ! -e "$torch_lib/$lib" ]; then
    echo "$lib: not bundled with torch, using /opt/rocm/lib"
    continue
  fi
  target="$(readlink -f "/opt/rocm/lib/${lib}")"
  [ -f "$target" ] && ln -sfn "$target" "$torch_lib/$lib"
  ls -l "$torch_lib/$lib"
done
echo "::endgroup::"

cd "$GITHUB_WORKSPACE"
bash -c "$CI_COMMAND"

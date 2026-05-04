#!/bin/bash
# Apply the SGLang patches required to run the random_async example on
# 1P1D4G disaggregation. Set SGLANG_PARENT to override the auto-detected
# sglang location.
set -euo pipefail

SGLANG_PARENT="${SGLANG_PARENT:-$(python3 -c 'import os, sglang; print(os.path.dirname(os.path.dirname(sglang.__file__)))')}"
PATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Applying SGLang patches under ${SGLANG_PARENT}"
for patch in "${PATCH_DIR}"/*.patch; do
    name=$(basename "$patch")
    echo "  [apply] $name"
    patch -p2 -d "${SGLANG_PARENT}" < "$patch"
done
echo "All patches applied successfully."

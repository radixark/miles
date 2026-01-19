#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOC_LANG=${1:-}

# make sure language is only en or zh
if [ "$DOC_LANG" != "en" ] && [ "$DOC_LANG" != "zh" ]; then
    echo "Language must be en or zh"
    exit 1
fi

cd "$SCRIPT_DIR"
SOURCE_DIR="./$DOC_LANG"
if [ ! -d "$SOURCE_DIR" ]; then
    if [ "$DOC_LANG" = "zh" ] && [ -d "./en" ]; then
        echo "[build] docs/zh not found; building zh output from docs/en sources." >&2
        SOURCE_DIR="./en"
    else
        echo "[build] Source directory not found: $SOURCE_DIR" >&2
        exit 1
    fi
fi

MILES_DOC_LANG="$DOC_LANG" sphinx-build -b html -D language="$DOC_LANG" --conf-dir ./ "$SOURCE_DIR" "./build/$DOC_LANG"

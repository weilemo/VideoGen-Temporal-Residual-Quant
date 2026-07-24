#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:?Usage: $0 MODEL_DIR [LONGCAT_REPO_DIR]}"
LONGCAT_REPO_DIR="${2:-forcing/longcatvideo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/patches/attention_sdpa_fallback.patch"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -d "${LONGCAT_REPO_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video.git "${LONGCAT_REPO_DIR}"
fi

if ! git -C "${LONGCAT_REPO_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  git -C "${LONGCAT_REPO_DIR}" apply --check "${PATCH_FILE}"
  git -C "${LONGCAT_REPO_DIR}" apply "${PATCH_FILE}"
fi

hf download meituan-longcat/LongCat-Video \
  --local-dir "${MODEL_DIR}"

test -d "${MODEL_DIR}/dit"
test -d "${MODEL_DIR}/text_encoder"
test -d "${MODEL_DIR}/tokenizer"
test -d "${MODEL_DIR}/vae"
test -d "${MODEL_DIR}/scheduler"
echo "LongCat repository and checkpoint are ready: ${MODEL_DIR}"

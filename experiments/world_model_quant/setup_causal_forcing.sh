#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAUSAL_REPO="${CAUSAL_REPO:-${REPO_ROOT}/forcing/causalforcing_official}"
WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${HOME}/storage/models/Wan2.1-T2V-1.3B}"
CAUSAL_MODEL_ROOT="${CAUSAL_MODEL_ROOT:-${HOME}/storage/models/Causal-Forcing}"
CAUSAL_COMMIT="1fc7bbc19a503c1bce80ecef08158b20e702f386"
PATCH_FILE="${REPO_ROOT}/experiments/world_model_quant/patches/causal_forcing_trq.patch"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TMPDIR="${TMPDIR:-${HOME}/storage/tmp}"
mkdir -p "${TMPDIR}" "${CAUSAL_MODEL_ROOT}"

if [[ ! -d "${CAUSAL_REPO}/.git" ]]; then
  git clone https://github.com/thu-ml/Causal-Forcing.git "${CAUSAL_REPO}"
fi
git -C "${CAUSAL_REPO}" fetch origin "${CAUSAL_COMMIT}"
git -C "${CAUSAL_REPO}" checkout "${CAUSAL_COMMIT}"

if git -C "${CAUSAL_REPO}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
  echo "Causal Forcing TRQ patch is already applied."
else
  git -C "${CAUSAL_REPO}" apply --check "${PATCH_FILE}"
  git -C "${CAUSAL_REPO}" apply "${PATCH_FILE}"
fi

if [[ ! -f "${WAN_MODEL_ROOT}/Wan2.1_VAE.pth" ]]; then
  hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "${WAN_MODEL_ROOT}"
fi
if [[ ! -f "${CAUSAL_MODEL_ROOT}/chunkwise/causal_forcing.pt" ]]; then
  hf download zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt \
    --local-dir "${CAUSAL_MODEL_ROOT}"
fi

test -f "${WAN_MODEL_ROOT}/Wan2.1_VAE.pth"
test -f "${WAN_MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"
test -f "${CAUSAL_MODEL_ROOT}/chunkwise/causal_forcing.pt"
echo "Causal Forcing backend and weights are ready."

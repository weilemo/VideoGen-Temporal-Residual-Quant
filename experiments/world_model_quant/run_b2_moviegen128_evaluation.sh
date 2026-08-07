#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
configure_videoquant_runtime

causal_gpu="${CAUSAL_GPU:-6}"
longcat_gpu="${LONGCAT_GPU:-7}"
run_id="${RUN_ID:-b2_moviegen128_$(date +%Y%m%d)}"
prompts="${PROMPTS_SOURCE:-${REPO_ROOT}/temporalresidualkvquant/assets/moviegenbench_resume_128.txt}"
causal_b1="${CAUSAL_B1_ROOT:?Set CAUSAL_B1_ROOT to the validated MovieGen32 Causal index root}"
causal_b2="${CAUSAL_B2_ROOT:?Set CAUSAL_B2_ROOT to the B2 Causal result root}"
longcat_b1="${LONGCAT_B1_ROOT:?Set LONGCAT_B1_ROOT to the validated MovieGen32 LongCat index root}"
longcat_b2="${LONGCAT_B2_ROOT:?Set LONGCAT_B2_ROOT to the B2 LongCat result root}"
index_root="${INDEX_ROOT:-${WORLD_MODEL_RESULTS_ROOT}/indexes/${run_id}}"
eval_root="${EVAL_ROOT:-${WORLD_MODEL_RESULTS_ROOT}/evaluation/${run_id}}"
status_root="${WORLD_MODEL_RESULTS_ROOT}/orchestration/${run_id}_evaluation"
run_vbench="${RUN_VBENCH:-1}"
dry_run="${DRY_RUN:-0}"

[[ "${causal_gpu}" != "${longcat_gpu}" ]] || { echo "CAUSAL_GPU and LONGCAT_GPU must differ" >&2; exit 2; }
python "${script_dir}/validate_moviegen_prompts.py" \
  --prompts "${prompts}" --expected 128 \
  --prefix "${REPO_ROOT}/temporalresidualkvquant/assets/moviegenbench_resume_32.txt" \
  --prefix-count 32 >/dev/null
for root in "${causal_b1}" "${causal_b2}" "${longcat_b1}" "${longcat_b2}"; do
  [[ -d "${root}" ]] || { echo "missing source root: ${root}" >&2; exit 2; }
done
mkdir -p "${status_root}" "${eval_root}"

if [[ "${dry_run}" == 1 ]]; then
  cat <<EOF
RUN_ID=${run_id}
GPU ${causal_gpu}: Causal MovieGen128 manifest, paired metrics, VBench-derived
GPU ${longcat_gpu}: LongCat MovieGen128 manifest, paired metrics, VBench-derived
EXPECTED_PROMPTS=128
RUN_VBENCH=${run_vbench}
INDEX_ROOT=${index_root}
EVAL_ROOT=${eval_root}
WORLD_MODEL_RESULTS_ROOT=${WORLD_MODEL_RESULTS_ROOT}
No manifests, metrics, or GPU jobs were started.
EOF
  exit 0
fi

python "${script_dir}/prepare_moviegen_manifest.py" \
  --baseline causal_forcing --prompts "${prompts}" --expected-prompts 128 \
  --source-root "b1=${causal_b1}" --source-root "b2=${causal_b2}" \
  --output-root "${index_root}/causal_forcing"
python "${script_dir}/prepare_moviegen_manifest.py" \
  --baseline longcat --prompts "${prompts}" --expected-prompts 128 \
  --source-root "b1=${longcat_b1}" --source-root "b2=${longcat_b2}" \
  --output-root "${index_root}/longcat"

record_stage() {
  local name="$1"
  shift
  if [[ -f "${status_root}/${name}.done" ]]; then
    echo "skip completed stage: ${name}"
    return 0
  fi
  printf 'running\n' > "${status_root}/${name}.status"
  if "$@" > "${status_root}/${name}.log" 2>&1; then
    printf 'done\n' > "${status_root}/${name}.status"
    touch "${status_root}/${name}.done"
    return 0
  fi
  printf 'failed\n' > "${status_root}/${name}.status"
  touch "${status_root}/${name}.failed"
  return 1
}

run_paired() {
  local baseline="$1"
  local root="$2"
  local output="$3"
  shift 3
  python "${REPO_ROOT}/experiments/paired_quality/run_forcing_paired_metrics.py" \
    --baseline "${baseline}" --root "${root}" --output-dir "${output}" \
    --expected-videos 128 --device cuda --bootstrap-resamples 2000 "$@"
}

run_vbench_for() {
  local baseline="$1"
  local root="$2"
  local output="$3"
  [[ "${run_vbench}" == 1 ]] || return 0
  PROMPT_FILE="${prompts}" VIDEO_ROOT="${root}" EXPECTED_VIDEOS=128 \
    OUTPUT_ROOT="${output}" bash "${REPO_ROOT}/integrations/evaluation/run_vbench.sh" "${baseline}"
  python "${REPO_ROOT}/temporalresidualkvquant/scripts/eval/aggregate_vbench_long.py" \
    --input-dir "${output}/scores" \
    --label "${baseline}_bf16" \
    --label "${baseline}_trq_int4" \
    --label "${baseline}_trq_int2" \
    --label "${baseline}_naive_int4" \
    --label "${baseline}_naive_int2" \
    --out "${output}/derived_summary.json" \
    --markdown "${output}/derived_summary.md"
}

causal_lane() {
  export CUDA_VISIBLE_DEVICES="${causal_gpu}"
  record_stage causal_paired run_paired causal_forcing \
    "${index_root}/causal_forcing" "${eval_root}/causal_forcing/paired_metrics"
  record_stage causal_vbench run_vbench_for causal_forcing \
    "${index_root}/causal_forcing" "${eval_root}/causal_forcing/vbench"
}

longcat_lane() {
  export CUDA_VISIBLE_DEVICES="${longcat_gpu}"
  record_stage longcat_paired run_paired longcat \
    "${index_root}/longcat" "${eval_root}/longcat/paired_metrics" --start-frame 13
  record_stage longcat_vbench run_vbench_for longcat \
    "${index_root}/longcat" "${eval_root}/longcat/vbench"
}

set +e
causal_lane &
causal_pid=$!
longcat_lane &
longcat_pid=$!
wait "${causal_pid}"; causal_status=$?
wait "${longcat_pid}"; longcat_status=$?
set -e

printf 'causal=%s longcat=%s\n' "${causal_status}" "${longcat_status}" |
  tee "${status_root}/summary.txt"
if [[ "${causal_status}" -eq 0 && "${longcat_status}" -eq 0 ]]; then
  touch "${status_root}/evaluation.done"
  echo "MovieGen128 B2 automatic evaluation complete: ${run_id}"
  exit 0
fi
touch "${status_root}/evaluation.partial"
exit 1

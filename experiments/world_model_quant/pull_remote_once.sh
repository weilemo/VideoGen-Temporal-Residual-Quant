#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
branch="${1:-codex/trq-online-causal-gates}"

cd "${repo_root}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "remote checkout changes detected; refusing to pull" >&2
  git status --short >&2
  exit 1
fi
current="$(git branch --show-current)"
if [[ "${current}" != "${branch}" ]]; then
  echo "current branch is ${current:-detached}; switch to ${branch} explicitly first" >&2
  exit 2
fi

git pull --ff-only origin "${branch}"
printf '%s\n' "updated ${repo_root} from origin/${branch}"

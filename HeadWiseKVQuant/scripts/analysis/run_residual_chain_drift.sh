#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS=drift exec bash "${script_dir}/run_trq_diagnostics.sh" "$@"

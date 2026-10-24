#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export KIROCREW_HOME="${KIROCREW_HOME:-$project_dir/.direct-home}"
export KIROCREW_TELEMETRY_DISABLED=1
export PATH="$project_dir/.venv/bin:$PATH"

if [[ ! -x "$project_dir/.venv/bin/kirocrew" ]]; then
    printf '%s\n' '项目环境未安装, 请先安装 Python 3.12 及项目依赖.' >&2
    exit 1
fi

exec timeout -k 10s "${KIROCREW_RUN_TIMEOUT:-7d}" "$project_dir/.venv/bin/kirocrew" gateway --no-open --no-tunnel "$@"
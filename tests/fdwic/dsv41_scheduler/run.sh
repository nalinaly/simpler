#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
backend=${1:-cpu}
if (($#)); then shift; fi
[[ "$backend" == cpu || "$backend" == a5 ]] || { echo 'usage: run.sh cpu|a5 [runner options]' >&2; exit 2; }
runner="$root/build/$backend/runner"
[[ -x "$runner" ]] || { echo "Run build.sh $backend first" >&2; exit 2; }
if [[ "$backend" == a5 ]]; then
    if command -v task-submit >/dev/null && [[ -z "${TASK_DEVICE:-}" ]]; then
        bash "$root/../../../.claude/skills/onboard-arch-precheck/check.sh" a5
        printf -v command 'bash %q a5' "$root/run.sh"
        for argument in "$@"; do printf -v command '%s %q' "$command" "$argument"; done
        task-submit --device auto --device-num 1 --run "$command"
        exit $?
    fi
    : "${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"
    exec "$runner" --binary "$root/build/a5/kernel.o" --device "${TASK_DEVICE:-0}" "$@"
fi
exec "$runner" "$@"

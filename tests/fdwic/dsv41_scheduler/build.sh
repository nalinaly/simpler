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
[[ "$backend" == cpu || "$backend" == a5 ]] || { echo 'usage: build.sh cpu|a5' >&2; exit 2; }
out="$root/build/$backend"
mkdir -p "$out"
start=$SECONDS
cxx=${CXX:-g++}
if [[ "$backend" == cpu ]]; then
    "$cxx" -std=c++17 -O2 -g -pthread -Wall -Wextra -Werror "$root/host.cpp" -o "$out/runner"
else
    : "${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"
    toolroot="$ASCEND_HOME_PATH/$(uname -m)-linux"
    pto=${PTO_ISA_ROOT:-$toolroot}
    [[ -f "$pto/include/pto/pto-inst.hpp" ]] || { echo "PTO headers missing under $pto" >&2; exit 2; }
    flags=(-c -O3 -g -x cce -std=c++17 --cce-aicore-only
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=false
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -I"$pto/include")
    "$ASCEND_HOME_PATH/bin/ccec" "${flags[@]}" --cce-aicore-arch=dav-c310-cube -DDSV_AIC \
        "$root/kernels/scheduler.cpp" -o "$out/aic.o" > "$out/aic.log" 2>&1 &
    aic_pid=$!
    "$ASCEND_HOME_PATH/bin/ccec" "${flags[@]}" --cce-aicore-arch=dav-c310-vec \
        "$root/kernels/scheduler.cpp" -o "$out/aiv.o" > "$out/aiv.log" 2>&1 &
    aiv_pid=$!
    result=0
    wait "$aic_pid" || result=1
    wait "$aiv_pid" || result=1
    if ((result)); then cat "$out/aic.log" "$out/aiv.log"; exit 1; fi
    "$ASCEND_HOME_PATH/bin/ld.lld" -m aicorelinux -Ttext=0 -static \
        --version-script="$root/kernels/exports.map" "$out/aic.o" "$out/aiv.o" -o "$out/kernel.o"
    "$cxx" -std=c++17 -O2 -g -pthread -Wall -Wextra -Wno-deprecated-declarations -DDSV_ONBOARD \
        -I"$toolroot/include" -I"$toolroot/pkg_inc" -I"$toolroot/pkg_inc/runtime" \
        -I"$toolroot/pkg_inc/runtime/runtime" "$root/host.cpp" \
        -L"$toolroot/lib64" -Wl,-rpath,"$toolroot/lib64" -lascendcl -lruntime -ldl -o "$out/runner"
fi
printf 'build_seconds=%s backend=%s\n' "$((SECONDS-start))" "$backend" | tee "$out/build-time.txt"
sha256sum "$out/runner" > "$out/artifacts.sha256"
if [[ "$backend" == a5 ]]; then sha256sum "$out/kernel.o" >> "$out/artifacts.sha256"; fi

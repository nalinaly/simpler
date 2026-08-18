#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Build-only helper for the standalone WithHostArgs/placeholder probe.
set -euo pipefail

PROBE_ROOT="$(cd "$(dirname "$0")" && pwd)"
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set}"
OUT_DIR="${1:-${TMPDIR:-/tmp}/gpt_pypto_host_args_probe}"
mkdir -p "${OUT_DIR}"

HCC_CXX="${HCC_CXX:-${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++}"
if [[ ! -x "${HCC_CXX}" ]]; then
  echo "HCC AICPU cross compiler not found at ${HCC_CXX}" >&2
  exit 2
fi

"${HCC_CXX}" -shared -fPIC -O2 -std=c++17 -Wall -Wextra -Werror -Wl,--build-id \
  -I"${PROBE_ROOT}/common" \
  -o "${OUT_DIR}/libhost_args_probe_aicpu.so" \
  "${PROBE_ROOT}/aicpu/host_args_probe_aicpu.cpp"

g++ -O2 -std=c++17 -Wall -Wextra -Werror \
  -I"${PROBE_ROOT}/common" \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/include" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime/runtime" \
  -L"${ASCEND_HOME_PATH}/lib64" \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
  -o "${OUT_DIR}/host_args_probe" \
  "${PROBE_ROOT}/host/host_args_probe.cpp" \
  -lascendcl -lruntime

g++ -O2 -std=c++17 -Wall -Wextra -Werror \
  -I"${PROBE_ROOT}/common" \
  -o "${OUT_DIR}/host_args_probe_self_test" \
  "${PROBE_ROOT}/host/host_args_probe_self_test.cpp" \
  "${PROBE_ROOT}/aicpu/host_args_probe_aicpu.cpp"

"${OUT_DIR}/host_args_probe_self_test"

echo "built ${OUT_DIR}/host_args_probe"

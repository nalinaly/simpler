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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CXX_BIN="${CXX:-g++}"
QK_CALLBACK_MANIFEST_NAME="qk_callback_artifacts.manifest"

BUILD_VARIANT="${1:-legacy}"
VARIANT_DEFINES=()
case "$BUILD_VARIANT" in
    legacy)
        if [[ $# -ne 0 ]]; then
            echo "Usage: $0 | $0 qk-callback callback-eager|callback-lazy" >&2
            exit 1
        fi
        BUILD_DIR="$ROOT_DIR/build/cpu"
        ;;
    qk-callback)
        if [[ $# -ne 2 ]]; then
            echo "Usage: $0 qk-callback callback-eager|callback-lazy" >&2
            exit 1
        fi
        QK_CALLBACK_SHAPE="$2"
        case "$QK_CALLBACK_SHAPE" in
            callback-eager) QK_CALLBACK_SHAPE_ID=1 ;;
            callback-lazy) QK_CALLBACK_SHAPE_ID=2 ;;
            *)
                echo "Unknown QK callback shape: $QK_CALLBACK_SHAPE" >&2
                exit 1
                ;;
        esac
        BUILD_DIR="$ROOT_DIR/build/cpu/qk-callback/$QK_CALLBACK_SHAPE"
        VARIANT_DEFINES=("-DPA_QK_CALLBACK_SHAPE_ID=$QK_CALLBACK_SHAPE_ID")
        ;;
    *)
        echo "Usage: $0 | $0 qk-callback callback-eager|callback-lazy" >&2
        exit 1
        ;;
esac

# CPU 后端只依赖 C++17、pthread 和本目录 common/，不需要 CANN。
# CXX 可显式指向用户目录下的 g++-15，未设置时沿用当前 PATH 中的 g++。

# CPU build 与设备 build 使用平行目录，便于 run.sh 根据 backend 做严格选择，
# 也避免把 host 回归二进制误当成 A5 产物。
mkdir -p "$BUILD_DIR"
if [[ "$BUILD_VARIANT" == "qk-callback" ]]; then
    if ! command -v sha256sum >/dev/null 2>&1; then
        echo "sha256sum is required to publish the QK callback manifest." >&2
        exit 1
    fi
    rm -f -- "$BUILD_DIR/$QK_CALLBACK_MANIFEST_NAME"
fi

echo "[BUILD] CPU scheduler executable"
# -pthread 同时提供编译期线程宏和链接期 pthread 支持；严格告警用于防止
# CPU 等价层因类型或原子接口变化而静默偏离设备端公共协议。
"$CXX_BIN" -O3 -std=c++17 -pthread -Wall -Wextra -Werror \
    "${VARIANT_DEFINES[@]}" \
    -I"$ROOT_DIR/common" \
    "$SCRIPT_DIR/main.cpp" \
    -o "$BUILD_DIR/pa_scheduler_cpu"

# PollBatch 是 common/ 中的设备/CPU 共用模板。这里用普通 C++17 编译器
# 直接实例化并执行边界自测；任一断言失败都会借助 set -e 阻止构建成功。
echo "[BUILD] atomic PollBatch boundary self-test"
"$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
    -DPA_BUILD_SWIMLANE=1 \
    -I"$ROOT_DIR/common" \
    "$ROOT_DIR/common/test_atomic_poll_batch.cpp" \
    -o "$BUILD_DIR/test_atomic_poll_batch"

echo "[TEST] atomic PollBatch boundary self-test"
"$BUILD_DIR/test_atomic_poll_batch"

if [[ "$BUILD_VARIANT" == "qk-callback" ]]; then
    # A callback artifact is not publishable merely because it links.  Run the
    # lightweight b1 real-compute path first so exact frontend/protocol counts
    # and the numerical output oracle gate every future callback rebuild.
    echo "[TEST] QK callback CPU b1 real-compute semantic oracle"
    "$BUILD_DIR/pa_scheduler_cpu" \
        --batches 1 --runs 1 --winner-workload real-compute --no-swimlane

    CALLBACK_ARTIFACTS=(pa_scheduler_cpu)
    if [[ ! -x "$BUILD_DIR/pa_scheduler_cpu" ]]; then
        echo "Cannot publish QK callback manifest; CPU executable is missing." >&2
        exit 1
    fi
    MANIFEST_PATH="$BUILD_DIR/$QK_CALLBACK_MANIFEST_NAME"
    MANIFEST_TMP="$(mktemp "$BUILD_DIR/.${QK_CALLBACK_MANIFEST_NAME}.tmp.XXXXXX")"
    cleanup_callback_manifest_tmp() {
        if [[ -n "${MANIFEST_TMP:-}" ]]; then
            rm -f -- "$MANIFEST_TMP"
        fi
    }
    trap cleanup_callback_manifest_tmp EXIT
    {
        printf '# schema=pa_scheduler_qk_callback_artifacts/v1\n'
        printf '# backend=cpu\n'
        printf '# shape=%s\n' "$QK_CALLBACK_SHAPE"
        printf '# shape_id=%u\n' "$QK_CALLBACK_SHAPE_ID"
        printf '# observation=inline-semantic\n'
        (cd "$BUILD_DIR" && sha256sum "${CALLBACK_ARTIFACTS[@]}")
    } > "$MANIFEST_TMP"
    mv -f -- "$MANIFEST_TMP" "$MANIFEST_PATH"
    MANIFEST_TMP=""
    trap - EXIT
    echo "[CHECK] QK callback artifact manifest published: $MANIFEST_PATH"
fi

# set -e 保证编译或链接失败时不会打印 complete，也不会在组合构建中继续
# 后续步骤；只有成功退出的构建才被本脚本声明为可运行产物。
echo "[BUILD] complete: $BUILD_DIR/pa_scheduler_cpu"

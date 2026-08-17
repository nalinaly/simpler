/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include "pto_runtime_c_api.h"

/**
 * Allocation-free operation table for one borrowed-stream L1 enqueue.
 *
 * The table deliberately contains only the six operations permitted inside
 * an L1 launch. In particular, synchronization, allocation, stream/event
 * creation, capture inspection, and model attachment cannot be expressed.
 * The opaque context normally points to a stack snapshot owned by the caller.
 */
struct L1LaunchSequenceOps {
    void *context{nullptr};
    int (*wait_event)(void *context, void *stream, void *event) noexcept {nullptr};
    int (*memset_handshake)(void *context, void *stream) noexcept {nullptr};
    int (*record_event)(void *context, void *event, void *stream) noexcept {nullptr};
    int (*launch_aicpu)(void *context, void *caller_stream) noexcept {nullptr};
    int (*launch_aicore)(void *context, void *hidden_stream) noexcept {nullptr};

    bool valid() const {
        return wait_event != nullptr && memset_handshake != nullptr && record_event != nullptr &&
               launch_aicpu != nullptr && launch_aicore != nullptr;
    }
};

struct L1LaunchSequenceHandles {
    void *caller_stream{nullptr};
    void *hidden_stream{nullptr};
    void *prepare_tail_event{nullptr};
    void *start_event{nullptr};
    void *aicore_done_event{nullptr};
    void *serial_tail_event{nullptr};
    bool wait_for_prepare_tail{true};
    bool wait_for_serial_tail{false};

    bool valid() const {
        return caller_stream != nullptr && hidden_stream != nullptr && prepare_tail_event != nullptr &&
               start_event != nullptr && aicore_done_event != nullptr && serial_tail_event != nullptr;
    }
};

/**
 * Enqueue the exact caller/hidden-stream fork-join protocol for one L1 op.
 *
 * Returns immediately on the first failed enqueue. The caller must poison the
 * context because a prefix may already be resident on either stream.
 */
inline int enqueue_l1_launch_sequence(const L1LaunchSequenceOps &ops, const L1LaunchSequenceHandles &handles) noexcept {
    if (!ops.valid() || !handles.valid()) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;

    int rc = 0;
    if (handles.wait_for_prepare_tail) {
        rc = ops.wait_event(ops.context, handles.caller_stream, handles.prepare_tail_event);
        if (rc != 0) return rc;
    }
    if (handles.wait_for_serial_tail) {
        rc = ops.wait_event(ops.context, handles.caller_stream, handles.serial_tail_event);
        if (rc != 0) return rc;
    }
    rc = ops.memset_handshake(ops.context, handles.caller_stream);
    if (rc != 0) return rc;
    rc = ops.record_event(ops.context, handles.start_event, handles.caller_stream);
    if (rc != 0) return rc;
    rc = ops.launch_aicpu(ops.context, handles.caller_stream);
    if (rc != 0) return rc;
    rc = ops.wait_event(ops.context, handles.hidden_stream, handles.start_event);
    if (rc != 0) return rc;
    rc = ops.launch_aicore(ops.context, handles.hidden_stream);
    if (rc != 0) return rc;
    rc = ops.record_event(ops.context, handles.aicore_done_event, handles.hidden_stream);
    if (rc != 0) return rc;
    rc = ops.wait_event(ops.context, handles.caller_stream, handles.aicore_done_event);
    if (rc != 0) return rc;
    return ops.record_event(ops.context, handles.serial_tail_event, handles.caller_stream);
}

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

#include "pto_orchestration_api.h"

#if defined(__CCE_AICORE__)
#include "callable.h"
#include "inner_kernel.h"
#include "pto_runtime2.h"
#include "runtime.h"

#include "dist_engine/common/state.h"
#include "dist_engine/common/atomic.h"
#include "dist_engine/common/trace.h"
#include "dist_engine/common/swimlane.h"
#include "dist_engine/common/worker_state.h"
#include "dist_engine/common/runtime_state.h"

extern "C" PTO_DEVICE_FUNC int32_t pto_call_linked_kernel_aic(int32_t func_id, __gm__ int64_t *args)
    __attribute__((weak));
extern "C" PTO_DEVICE_FUNC int32_t pto_call_linked_kernel_aiv(int32_t func_id, __gm__ int64_t *args)
    __attribute__((weak));

#ifndef SPIN_WAIT_HINT
#define SPIN_WAIT_HINT() ((void)0)
#endif

#include "dist_engine/aicore/primitive.h"
#include "dist_engine/aicore/log.h"
#include "dist_engine/aicore/tensor_map.h"
#include "dist_engine/aicore/core_state.h"
#include "dist_engine/aicore/submit_core.h"
#endif

#include "dist_engine/aicore/qk_split_submit_types.h"

#define PTO_QK_SPLIT_ALWAYS_INLINE PTO_DEVICE_FUNC __attribute__((always_inline)) inline

template <bool Lazy>
class FdwicPrivateAicQkBuilder {
public:
    PTO_QK_SPLIT_ALWAYS_INLINE FdwicPrivateAicQkBuilder(L0TaskArgs &args, bool won) : args_(args), won_(won) {}

    template <typename Thunk>
    PTO_QK_SPLIT_ALWAYS_INLINE void add_input(Thunk thunk) {
        if (!Lazy || won_) args_.add_input(thunk());
    }

    template <typename Thunk>
    PTO_QK_SPLIT_ALWAYS_INLINE void add_output(Thunk thunk) {
        args_.add_output(thunk());
    }

    template <typename Thunk>
    PTO_QK_SPLIT_ALWAYS_INLINE void add_scalar(Thunk thunk) {
        if (!Lazy || won_) args_.add_scalar(thunk());
    }

private:
    L0TaskArgs &args_;
    bool won_;
};

template <bool Lazy, typename BuildCallback>
PTO_QK_SPLIT_ALWAYS_INLINE TaskOutputTensors rt_submit_private_aic_qk_split_once(
    int32_t kernel_id, L0TaskArgs &scratch, BuildCallback build
) {
#if defined(__CCE_AICORE__)
    if (dist_is_fatal_query()) return TaskOutputTensors{};

    __gm__ DistCore *self = g_self;
    if (self == nullptr) {
        fdwic_trace_set_fatal();
        return TaskOutputTensors{};
    }

    const int32_t task_id = self->local_index++;
    TRACE_SPAN_BEGIN(submit_trace);
    TRACE_LAP_RESET(self);
    drain_block_won(self);
    drain_phase_b(self);
    TRACE_LAP(self, task_id, -1, TracePhase::EfDrain);

    if (task_id < 0 || task_id >= kFlagCap || kernel_id < 0 || kernel_id >= RUNTIME_MAX_FUNC_ID) {
        fdwic_trace_set_fatal(task_id);
        return TaskOutputTensors{};
    }

    TRACE_SPAN_BEGIN(claim_trace);
    const bool claim_attempted = self->role == CoreType::AIC;
    bool won = false;
    if (claim_attempted) {
        const int64_t old = fdwic_trace_atomic_fetch_max<int64_t>(
            task_id, FdwicAtomicSite::ClaimMax, g_dist.cube_cursor[task_id % kCursorShards].v,
            static_cast<int64_t>(task_id), /*result_used=*/true
        );
        won = task_id > old;
    }
    const int32_t claimed_kernel_id = won ? kernel_id : INVALID_KERNEL_ID;
    const uint32_t claim_flags =
        fdwic_atomic_swimlane_enabled() ?
            (won ? kFdwicClaimWon : 0U) | (claim_attempted ? kFdwicClaimAttempted : 0U) :
            static_cast<uint32_t>(won);
    TRACE_SPAN_END(claim_trace, self, task_id, claimed_kernel_id, TracePhase::Claim, claim_flags, 0);

    scratch.reset();
    FdwicPrivateAicQkBuilder<Lazy> builder(scratch, won);
    build(builder);

    const FdwicQkSplitSubmitTicket ticket{
        submit_trace,
        static_cast<uint32_t>(task_id),
        static_cast<int16_t>(kernel_id),
        static_cast<uint8_t>(won ? 1 : 0),
        0,
    };
#if defined(__DAV_CUBE__)
    return fdwic_qk_split_finish_aic(&ticket, &scratch);
#elif defined(__DAV_VEC__)
    return fdwic_qk_split_finish_aiv(&ticket, &scratch);
#endif
#else
    scratch.reset();
    FdwicPrivateAicQkBuilder<Lazy> builder(scratch, true);
    build(builder);
    return rt_submit_aic_task(kernel_id, scratch);
#endif
}

#undef PTO_QK_SPLIT_ALWAYS_INLINE

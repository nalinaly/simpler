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

#include "cce_aicore_intrinsics.h"
#include <pto/common/constants.hpp>
#include <pto/common/kernel_meta.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#define PA_DEVICE __aicore__ inline
#define PA_DEVICE_NOINLINE static __aicore__ __attribute__((noinline))
#define PA_LOOP_NOUNROLL _Pragma("clang loop unroll(disable)")
#define PA_GM __gm__
#include "pmu_probe.h"
#include "../common/winner_workload.h"
#include "../common/pa_scheduler_core.h"

#define PA_CCEC_OPS_DEFINE_REAL_WORKLOAD 1
#include "ccec_ops.h"
#undef PA_CCEC_OPS_DEFINE_REAL_WORKLOAD

using pa_scheduler_ccec::CcecOps;
#if defined(PA_BUILD_AIC)
// Match ordinary's per-Scalar state ownership without its Build/split-finish path.
[[block_local]] pa_scheduler::LocalStats pa_only_scheduler_stats_aic;
// 同一源码分别按 cube/vec 架构编译；metadata 声明每个物理 block 静态组合 1 个 AIC 与 2 个 AIV。
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void pa_scheduler_0_mix_aic(__gm__ pa_scheduler::SchedulerState *state) {
    // 八个 active AIC，连续 worker 0..7。
    const uint32_t worker_id = static_cast<uint32_t>(get_block_idx());
    pa_scheduler::RunOnlyScheduler<CcecOps>(
        state, worker_id, pa_scheduler::CoreRole::Aic, pa_only_scheduler_stats_aic);
}
#elif defined(PA_BUILD_AIV)
[[block_local]] pa_scheduler::LocalStats pa_only_scheduler_stats_aiv;
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void pa_scheduler_0_mix_aiv(__gm__ pa_scheduler::SchedulerState *state) {
    // mixed launch 仍创建两个 AIV context；subblock 1 不参与任何调度/计算。
    if (get_subblockid() != 0) return;
    const uint32_t vector_id = static_cast<uint32_t>(get_block_idx());
    const uint32_t worker_id = pa_scheduler::kAicWorkers + vector_id;
    pa_scheduler::RunOnlyScheduler<CcecOps>(
        state, worker_id, pa_scheduler::CoreRole::Aiv, pa_only_scheduler_stats_aiv);
}
#else
#error "Compile with PA_BUILD_AIC or PA_BUILD_AIV"
#endif

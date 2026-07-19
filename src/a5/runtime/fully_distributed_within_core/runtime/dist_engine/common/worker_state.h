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

#include "dist_engine/common/target.h"
#include "dist_engine/common/swimlane_types.h"

// g_dist / g_self storage. The AICPU build owns the BSS DistGlobal and
// initializes it in dist_engine_register. AICore builds attach to the published
// Runtime::dist.shared_addr at dist_core_main entry: CCEC uses GM pointers,
// while CPU sim uses an ordinary process pointer into the AICPU DSO's BSS.
// The `g_dist` macro keeps hot-path code identical.
#if defined(__CCE_AICORE__)
#define FDWIC_WORKER_STATE_NAME_IMPL(base, suffix) base##suffix
#define FDWIC_WORKER_STATE_NAME_EXPAND(base, suffix) FDWIC_WORKER_STATE_NAME_IMPL(base, suffix)
#if defined(__DAV_CUBE__)
#define FDWIC_WORKER_STATE_NAME(base) FDWIC_WORKER_STATE_NAME_EXPAND(base, _aic)
#elif defined(__DAV_VEC__)
#define FDWIC_WORKER_STATE_NAME(base) FDWIC_WORKER_STATE_NAME_EXPAND(base, _aiv)
#else
#error "CCEC worker state requires __DAV_CUBE__ or __DAV_VEC__"
#endif

#define g_dist_ptr FDWIC_WORKER_STATE_NAME(fdwic_g_dist_ptr)
#define g_self FDWIC_WORKER_STATE_NAME(fdwic_g_self)
#define g_ccec_runtime FDWIC_WORKER_STATE_NAME(fdwic_g_ccec_runtime)
#define g_ccec_core_idx FDWIC_WORKER_STATE_NAME(fdwic_g_ccec_core_idx)
#define g_ccec_core_type FDWIC_WORKER_STATE_NAME(fdwic_g_ccec_core_type)
#define g_ccec_aic_count FDWIC_WORKER_STATE_NAME(fdwic_g_ccec_aic_count)
#define g_ccec_aiv_count FDWIC_WORKER_STATE_NAME(fdwic_g_ccec_aiv_count)
#define g_ccec_ordinal FDWIC_WORKER_STATE_NAME(fdwic_g_ccec_ordinal)
#define g_ccec_valid_worker FDWIC_WORKER_STATE_NAME(fdwic_g_ccec_valid_worker)
#define g_fdwic_joint_submit_seen FDWIC_WORKER_STATE_NAME(fdwic_g_joint_submit_seen)
#define g_fdwic_swimlane_level FDWIC_WORKER_STATE_NAME(fdwic_g_swimlane_level)
#define g_fdwic_swimlane_header FDWIC_WORKER_STATE_NAME(fdwic_g_swimlane_header)
#define g_fdwic_swimlane_core FDWIC_WORKER_STATE_NAME(fdwic_g_swimlane_core)
#define g_fdwic_swimlane_records FDWIC_WORKER_STATE_NAME(fdwic_g_swimlane_records)
#define g_fdwic_swimlane_records_per_core FDWIC_WORKER_STATE_NAME(fdwic_g_swimlane_records_per_core)
#define g_fdwic_atomic_poll_burst FDWIC_WORKER_STATE_NAME(fdwic_g_atomic_poll_burst)
#define g_fdwic_atomic_calls FDWIC_WORKER_STATE_NAME(fdwic_g_atomic_calls)
#define g_fdwic_poll_calls FDWIC_WORKER_STATE_NAME(fdwic_g_poll_calls)
#define g_fdwic_poll_batch_records FDWIC_WORKER_STATE_NAME(fdwic_g_poll_batch_records)
#define g_fdwic_atomic_counter_overflow FDWIC_WORKER_STATE_NAME(fdwic_g_atomic_counter_overflow)

#if defined(FDWIC_DEFINE_CCEC_WORKER_STATE)
#define FDWIC_WORKER_STATE_STORAGE
#else
#define FDWIC_WORKER_STATE_STORAGE extern
#endif

extern "C" {
[[block_local]] FDWIC_WORKER_STATE_STORAGE __gm__ DistGlobal *g_dist_ptr;
[[block_local]] FDWIC_WORKER_STATE_STORAGE __gm__ DistCore *g_self;
[[block_local]] FDWIC_WORKER_STATE_STORAGE __gm__ Runtime *g_ccec_runtime;
[[block_local]] FDWIC_WORKER_STATE_STORAGE int32_t g_ccec_core_idx;
[[block_local]] FDWIC_WORKER_STATE_STORAGE int32_t g_ccec_core_type;
[[block_local]] FDWIC_WORKER_STATE_STORAGE int32_t g_ccec_aic_count;
[[block_local]] FDWIC_WORKER_STATE_STORAGE int32_t g_ccec_aiv_count;
[[block_local]] FDWIC_WORKER_STATE_STORAGE int32_t g_ccec_ordinal;
[[block_local]] FDWIC_WORKER_STATE_STORAGE bool g_ccec_valid_worker;
[[block_local]] FDWIC_WORKER_STATE_STORAGE bool g_fdwic_joint_submit_seen;
[[block_local]] FDWIC_WORKER_STATE_STORAGE uint32_t g_fdwic_swimlane_level;
[[block_local]] FDWIC_WORKER_STATE_STORAGE __gm__ FdwicSwimlaneHeader *g_fdwic_swimlane_header;
[[block_local]] FDWIC_WORKER_STATE_STORAGE __gm__ FdwicSwimlaneCoreState *g_fdwic_swimlane_core;
[[block_local]] FDWIC_WORKER_STATE_STORAGE __gm__ FdwicSwimlaneRecord *g_fdwic_swimlane_records;
[[block_local]] FDWIC_WORKER_STATE_STORAGE uint32_t g_fdwic_swimlane_records_per_core;
[[block_local]] FDWIC_WORKER_STATE_STORAGE FdwicAtomicPollBurst g_fdwic_atomic_poll_burst;
[[block_local]] FDWIC_WORKER_STATE_STORAGE uint32_t g_fdwic_atomic_calls;
[[block_local]] FDWIC_WORKER_STATE_STORAGE uint32_t g_fdwic_poll_calls;
[[block_local]] FDWIC_WORKER_STATE_STORAGE uint32_t g_fdwic_poll_batch_records;
[[block_local]] FDWIC_WORKER_STATE_STORAGE bool g_fdwic_atomic_counter_overflow;
}

#undef FDWIC_WORKER_STATE_STORAGE
#define g_dist (*g_dist_ptr)
#elif defined(__CPU_SIM)
static DistGlobal g_dist_fallback;
static DistGlobal *g_dist_ptr = nullptr;
thread_local DistCore *g_self = nullptr;
thread_local bool g_fdwic_joint_submit_seen = false;
thread_local uint32_t g_fdwic_swimlane_level = 0;
thread_local FdwicSwimlaneHeader *g_fdwic_swimlane_header = nullptr;
thread_local FdwicSwimlaneCoreState *g_fdwic_swimlane_core = nullptr;
thread_local FdwicSwimlaneRecord *g_fdwic_swimlane_records = nullptr;
thread_local uint32_t g_fdwic_swimlane_records_per_core = 0;
thread_local FdwicAtomicPollBurst g_fdwic_atomic_poll_burst = {};
thread_local uint32_t g_fdwic_atomic_calls = 0;
thread_local uint32_t g_fdwic_poll_calls = 0;
thread_local uint32_t g_fdwic_poll_batch_records = 0;
thread_local bool g_fdwic_atomic_counter_overflow = false;
#define g_dist (*g_dist_ptr)
#else
static DistGlobal g_dist_fallback;
static DistGlobal *g_dist_ptr = &g_dist_fallback;
thread_local DistCore *g_self = nullptr;
thread_local bool g_fdwic_joint_submit_seen = false;
thread_local uint32_t g_fdwic_swimlane_level = 0;
thread_local FdwicSwimlaneHeader *g_fdwic_swimlane_header = nullptr;
thread_local FdwicSwimlaneCoreState *g_fdwic_swimlane_core = nullptr;
thread_local FdwicSwimlaneRecord *g_fdwic_swimlane_records = nullptr;
thread_local uint32_t g_fdwic_swimlane_records_per_core = 0;
thread_local FdwicAtomicPollBurst g_fdwic_atomic_poll_burst = {};
thread_local uint32_t g_fdwic_atomic_calls = 0;
thread_local uint32_t g_fdwic_poll_calls = 0;
thread_local uint32_t g_fdwic_poll_batch_records = 0;
thread_local bool g_fdwic_atomic_counter_overflow = false;
#define g_dist (*g_dist_ptr)
#endif

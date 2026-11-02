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

#include "scheduler_graph_nodes.h"
#include "scheduler_graph_work.h"
#include "scheduler_memory.h"

struct SchedulerGraphWorkToken {
    uint64_t index{0};
    uint64_t epoch{0};
    uint64_t owner{0};
    uint64_t state_address{0};
};

enum class SchedulerGraphWorkCompletion : uint32_t { REJECTED, COMPLETED, LAST };

inline __aicore__ bool scheduler_graph_work_matches(
    const __gm__ SchedulerGraphWorkGroup &group, const SchedulerGraphWorkToken &token, uint64_t epoch
) {
    return token.epoch == epoch && token.owner != 0 && token.owner <= UINT32_MAX && token.index >= group.begin &&
           token.index - group.begin < group.count;
}

inline __aicore__ bool scheduler_graph_work_claim(
    const __gm__ SchedulerGraphWorkGroup &group, __gm__ SchedulerGraphWorkState &state, uint64_t index, uint32_t worker,
    uint64_t epoch, SchedulerGraphWorkToken *token
) {
    if (token == nullptr || worker == UINT32_MAX || index < group.begin || index - group.begin >= group.count)
        return false;
    const uint64_t owner = static_cast<uint64_t>(worker) + 1;
    if (scheduler_gm_compare_exchange(state.owner_state, UINT64_C(0), (owner << 2) | UINT64_C(1)) != 0) return false;
    *token = {index, epoch, owner, reinterpret_cast<uint64_t>(&state)};
    return true;
}

// The owner writes back its entire range before contributing. The state CAS
// makes every contribution unique even if a caller retries the same token.
inline __aicore__ SchedulerGraphWorkCompletion scheduler_graph_work_complete(
    __gm__ SchedulerGraphWorkGroup &group, __gm__ SchedulerGraphWorkState &state, const SchedulerGraphWorkToken &token,
    uint64_t epoch, uint64_t *remaining_before
) {
    if (remaining_before != nullptr) *remaining_before = 0;
    if (!scheduler_graph_work_matches(group, token, epoch) || token.state_address != reinterpret_cast<uint64_t>(&state))
        return SchedulerGraphWorkCompletion::REJECTED;
    scheduler_cache_barrier();
    const uint64_t expected = (token.owner << 2) | UINT64_C(1);
    if (scheduler_gm_compare_exchange(state.owner_state, expected, (token.owner << 2) | UINT64_C(2)) != expected)
        return SchedulerGraphWorkCompletion::REJECTED;
    const uint64_t prior = scheduler_gm_fetch_add(group.remaining, UINT64_MAX);
    if (remaining_before != nullptr) *remaining_before = prior;
    if (prior == 0 || prior > group.count) return SchedulerGraphWorkCompletion::REJECTED;
    return prior == 1 ? SchedulerGraphWorkCompletion::LAST : SchedulerGraphWorkCompletion::COMPLETED;
}

inline __aicore__ bool scheduler_graph_work_fail(
    const __gm__ SchedulerGraphWorkGroup &group, __gm__ SchedulerGraphWorkState &state,
    const SchedulerGraphWorkToken &token, uint64_t epoch
) {
    if (!scheduler_graph_work_matches(group, token, epoch) || token.state_address != reinterpret_cast<uint64_t>(&state))
        return false;
    const uint64_t expected = (token.owner << 2) | UINT64_C(1);
    return scheduler_gm_compare_exchange(state.owner_state, expected, (token.owner << 2) | UINT64_C(3)) == expected;
}

// Only the last successful work contributor calls this. A failed instance is
// terminal, and ownership makes repeated publication unable to drain the run.
inline __aicore__ bool scheduler_graph_work_publish(
    __gm__ SchedulerGraphInstanceControl &instance, __gm__ SchedulerGraphBuildControl &build,
    const SchedulerGraphWorkToken &token, uint64_t *remaining_before
) {
    if (remaining_before != nullptr) *remaining_before = 0;
    if (instance.run_epoch != build.run_epoch || !scheduler_graph_work_matches(instance.work, token, build.run_epoch) ||
        instance.instance_index >= build.instance_count ||
        scheduler_gm_fetch_add(instance.work.remaining, UINT64_C(0)) != 0)
        return false;
    if (scheduler_gm_compare_exchange(instance.materialize_owner, UINT64_C(0), token.owner) != 0) return false;
    if (scheduler_gm_compare_exchange(instance.materialize_state, UINT64_C(1), UINT64_C(2)) != 1) return false;
    const uint64_t prior = scheduler_gm_fetch_add(build.remaining_instances, UINT64_MAX);
    if (remaining_before != nullptr) *remaining_before = prior;
    return prior != 0 && prior <= build.instance_count;
}

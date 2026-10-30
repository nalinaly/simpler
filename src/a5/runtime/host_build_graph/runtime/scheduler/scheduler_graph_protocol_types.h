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

#include "scheduler_protocol_types.h"

// Each record has one writer and an isolated cache line. Graph-free and L0
// runs allocate none of these records; stamps share the native worker stream.
struct alignas(128) SchedulerGraphProtocolInstance {
    SchedulerProtocolStamp materialize_before;
    SchedulerProtocolStamp materialize_after;
    SchedulerProtocolStamp publish_before;
    SchedulerProtocolStamp publish_after;
    uint64_t run_epoch;
    uint64_t definition_key;
    uint64_t remaining_before;
    uint32_t instance_index;
    uint32_t parent_id;
    uint32_t worker_index;
    uint32_t child_count;
    uint32_t status;
};

struct alignas(128) SchedulerGraphProtocolContribution {
    SchedulerProtocolStamp before;
    SchedulerProtocolStamp after;
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t remaining_before;
    uint32_t worker_index;
    uint32_t instance_index;
};

struct alignas(128) SchedulerGraphProtocolWorker {
    SchedulerProtocolStamp before;
    SchedulerProtocolStamp after;
    uint64_t run_epoch;
    uint64_t remaining_observed;
    uint64_t poll_count;
    uint32_t worker_index;
    uint32_t instance_count;
    uint32_t status;
};

static_assert(sizeof(SchedulerGraphProtocolInstance) == 128);
static_assert(sizeof(SchedulerGraphProtocolContribution) == 128);
static_assert(sizeof(SchedulerGraphProtocolWorker) == 128);
static_assert(
    std::is_standard_layout_v<SchedulerGraphProtocolInstance> &&
    std::is_trivially_copyable_v<SchedulerGraphProtocolInstance>
);
static_assert(
    std::is_standard_layout_v<SchedulerGraphProtocolContribution> &&
    std::is_trivially_copyable_v<SchedulerGraphProtocolContribution>
);
static_assert(
    std::is_standard_layout_v<SchedulerGraphProtocolWorker> &&
    std::is_trivially_copyable_v<SchedulerGraphProtocolWorker>
);

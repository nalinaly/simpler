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

#include <cstdint>
#include <type_traits>

enum class SchedulerGraphWorkRegion : uint32_t { TASK_STORAGE, TENSOR_POOL, SCALAR_POOL, SERIAL_INSTANCE };

struct SchedulerGraphWorkRange {
    uint32_t begin;
    uint32_t count;
};

// Ranges name elements of one instance's region. The arrays are immutable for
// the run; ownership and completion belong to separate atomic control words.
struct SchedulerGraphWork {
    uint32_t instance_index;
    SchedulerGraphWorkRegion region;
    uint32_t first_range;
    uint32_t range_count;
    uint64_t write_bytes;
};

static_assert(sizeof(SchedulerGraphWorkRange) == 8);
static_assert(sizeof(SchedulerGraphWork) == 24);
static_assert(std::is_standard_layout_v<SchedulerGraphWork> && std::is_trivially_copyable_v<SchedulerGraphWork>);
static_assert(
    std::is_standard_layout_v<SchedulerGraphWorkRange> && std::is_trivially_copyable_v<SchedulerGraphWorkRange>
);

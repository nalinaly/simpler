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

#include <gtest/gtest.h>

#include "pto_runtime2.h"

TEST(HbgL1RuntimeArenaSizing, KeepsHistoricalDefaultForL2) {
    const PTO2RuntimeArenaSizing sizing = pto2_default_runtime_arena_sizing();
    EXPECT_EQ(sizing.ready_queue_capacity, PTO2_READY_QUEUE_SIZE);
    EXPECT_EQ(sizing.tensor_map_num_buckets, PTO2_TENSORMAP_NUM_BUCKETS);
    EXPECT_EQ(sizing.tensor_map_pool_size, PTO2_TENSORMAP_POOL_SIZE);
}

TEST(HbgL1RuntimeArenaSizing, DerivesBoundedCompactCapacities) {
    const PTO2RuntimeArenaSizing small = pto2_hbg_l1_runtime_arena_sizing(16);
    EXPECT_EQ(small.ready_queue_capacity, 64);
    EXPECT_EQ(small.tensor_map_num_buckets, 128);
    EXPECT_EQ(small.tensor_map_pool_size, 16 * CORE_MAX_TENSOR_ARGS);

    const PTO2RuntimeArenaSizing medium = pto2_hbg_l1_runtime_arena_sizing(64);
    EXPECT_EQ(medium.ready_queue_capacity, 64);
    EXPECT_EQ(medium.tensor_map_num_buckets, 512);
    EXPECT_EQ(medium.tensor_map_pool_size, 64 * CORE_MAX_TENSOR_ARGS);

    const PTO2RuntimeArenaSizing large = pto2_hbg_l1_runtime_arena_sizing(PTO2_TASK_WINDOW_SIZE);
    EXPECT_EQ(large.ready_queue_capacity, PTO2_READY_QUEUE_SIZE);
    EXPECT_EQ(large.tensor_map_num_buckets, PTO2_TENSORMAP_NUM_BUCKETS);
    EXPECT_EQ(large.tensor_map_pool_size, PTO2_TENSORMAP_POOL_SIZE);
}

TEST(HbgL1RuntimeArenaSizing, ShrinksOnlyExplicitL1Layout) {
    uint64_t task_windows[PTO2_MAX_RING_DEPTH] = {16};
    uint64_t heap_sizes[PTO2_MAX_RING_DEPTH] = {PTO2_HEAP_SIZE};

    DeviceArena default_arena;
    const PTO2RuntimeArenaLayout default_layout = runtime_reserve_layout(default_arena, task_windows, heap_sizes);

    DeviceArena compact_arena;
    const PTO2RuntimeArenaLayout compact_layout = runtime_reserve_layout(
        compact_arena, task_windows, heap_sizes, pto2_hbg_l1_runtime_arena_sizing(task_windows[0])
    );

    EXPECT_EQ(default_layout.sched.ready_queue_capacity, PTO2_READY_QUEUE_SIZE);
    EXPECT_EQ(default_layout.orch.tensor_map.pool_size, PTO2_TENSORMAP_POOL_SIZE);
    EXPECT_EQ(compact_layout.sched.ready_queue_capacity, 64);
    EXPECT_EQ(compact_layout.orch.tensor_map.pool_size, 16 * CORE_MAX_TENSOR_ARGS);
    EXPECT_LT(compact_layout.arena_size, default_layout.arena_size / 8);
}

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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

#include "dist_engine/aicpu/shared_tensor_map_init.h"

namespace {

#if PTO_FDWIC_SHARED_MAP

constexpr uint8_t kPayloadPattern = 0xa5;

bool bytes_keep_pattern(const void *object, size_t offset, size_t bytes) {
    const auto *data = static_cast<const uint8_t *>(object);
    for (size_t index = offset; index < offset + bytes; ++index) {
        if (data[index] != kPayloadPattern) return false;
    }
    return true;
}

TEST(FdwicSharedPaTensorMapState, ExactLayoutAppendsAfterTheFrozenPrivateTail) {
    EXPECT_EQ(kFdwicSharedPaBatches, 256U);
    EXPECT_EQ(kFdwicSharedPaTasksPerBatch, 5U);
    EXPECT_EQ(kFdwicSharedPaTaskCapacity, 1280U);
    EXPECT_EQ(kFdwicSharedOutputMaxPerTask, 8U);
    EXPECT_EQ(kFdwicSharedHeapShards, 8U);
    EXPECT_EQ(kFdwicSharedVectorCursorShards, 8U);
    EXPECT_EQ(kFdwicSharedHeapBytes, 256ULL << 20);
    EXPECT_EQ(kFdwicSharedHeapShardBytes, 32ULL << 20);

    EXPECT_EQ(sizeof(SharedOutputCell), 2048U);
    EXPECT_EQ(alignof(SharedOutputCell), kCacheLine);
    EXPECT_EQ(offsetof(SharedOutputCell, published), 0U);
    EXPECT_EQ(offsetof(SharedOutputCell, last_writer), 512U);
    EXPECT_EQ(offsetof(SharedOutputCell, tensors), 1024U);

    EXPECT_EQ(sizeof(SharedWriterHistoryRecord), 8U);
    EXPECT_EQ(sizeof(SharedWriterHistoryCell), 320U);
    EXPECT_EQ(alignof(SharedWriterHistoryCell), kCacheLine);
    EXPECT_EQ(offsetof(SharedWriterHistoryCell, entries), 16U);

    EXPECT_EQ(offsetof(SharedPaTensorMapState, shared_outputs), 0U);
    EXPECT_EQ(offsetof(SharedPaTensorMapState, shared_heap_cursor), 2621440U);
    EXPECT_EQ(offsetof(SharedPaTensorMapState, shared_heap_vend), 2621952U);
    EXPECT_EQ(offsetof(SharedPaTensorMapState, shared_vector_cursor), 2622016U);
    EXPECT_EQ(offsetof(SharedPaTensorMapState, writer_history), 2622528U);
    EXPECT_EQ(sizeof(SharedPaTensorMapState), 3032128U);
    EXPECT_EQ(alignof(SharedPaTensorMapState), kCacheLine);

    EXPECT_EQ(offsetof(DistTaskCell, deps_prepared), 16U);
    EXPECT_EQ(offsetof(DistGlobal, shared_pa), kFdwicSharedTensorMapOffset);
    EXPECT_EQ(sizeof(DistGlobal), kFdwicSharedTensorMapOffset + sizeof(SharedPaTensorMapState));
    EXPECT_LE(sizeof(DistGlobal), kDistEngineGlobalStateSize);
}

TEST(FdwicSharedPaTensorMapState, AicpuResetOnlyReinitializesPublishedControlAndHistoryHeaders) {
    auto state = std::make_unique<SharedPaTensorMapState>();
    std::memset(state.get(), kPayloadPattern, sizeof(*state));

    dist_shared_pa_tensor_map_reset(*state);

    for (uint32_t task = 0; task < kFdwicSharedPaTaskCapacity; ++task) {
        const SharedOutputCell &outputs = state->shared_outputs[task];
        for (uint32_t output = 0; output < kFdwicSharedOutputMaxPerTask; ++output) {
            EXPECT_EQ(outputs.published[output].v, -1) << "task=" << task << ", output=" << output;
            EXPECT_EQ(outputs.last_writer[output].v, -1) << "task=" << task << ", output=" << output;
        }
        EXPECT_TRUE(bytes_keep_pattern(
            &outputs, offsetof(SharedOutputCell, tensors),
            sizeof(SharedOutputCell) - offsetof(SharedOutputCell, tensors)
        )) << "task="
           << task;

        const SharedWriterHistoryCell &history = state->writer_history[task];
        EXPECT_EQ(history.magic, 0U) << "task=" << task;
        EXPECT_EQ(history.writer_task, -1) << "task=" << task;
        EXPECT_EQ(history.count, 0U) << "task=" << task;
        EXPECT_EQ(history.reserved, 0U) << "task=" << task;
        EXPECT_TRUE(bytes_keep_pattern(
            &history, offsetof(SharedWriterHistoryCell, entries),
            sizeof(SharedWriterHistoryCell) - offsetof(SharedWriterHistoryCell, entries)
        )) << "task="
           << task;
    }
    for (uint32_t shard = 0; shard < kFdwicSharedHeapShards; ++shard) {
        EXPECT_EQ(state->shared_heap_cursor[shard].v, 0) << "shard=" << shard;
    }
    EXPECT_EQ(state->shared_heap_vend.v, 0);
    for (uint32_t shard = 0; shard < kFdwicSharedVectorCursorShards; ++shard) {
        EXPECT_EQ(state->shared_vector_cursor[shard].v, -1) << "shard=" << shard;
    }

    state->shared_outputs[kFdwicSharedPaTaskCapacity - 1].published[7].v = 37;
    state->shared_outputs[kFdwicSharedPaTaskCapacity - 1].last_writer[7].v = 41;
    state->shared_heap_cursor[7].v = 4096;
    state->shared_heap_vend.v = 8192;
    state->shared_vector_cursor[7].v = 53;
    state->writer_history[kFdwicSharedPaTaskCapacity - 1].magic = kFdwicSharedWriterHistoryMagic;
    state->writer_history[kFdwicSharedPaTaskCapacity - 1].writer_task = 1279;
    state->writer_history[kFdwicSharedPaTaskCapacity - 1].count = 3;

    dist_shared_pa_tensor_map_reset(*state);

    EXPECT_EQ(state->shared_outputs[kFdwicSharedPaTaskCapacity - 1].published[7].v, -1);
    EXPECT_EQ(state->shared_outputs[kFdwicSharedPaTaskCapacity - 1].last_writer[7].v, -1);
    EXPECT_EQ(state->shared_heap_cursor[7].v, 0);
    EXPECT_EQ(state->shared_heap_vend.v, 0);
    EXPECT_EQ(state->shared_vector_cursor[7].v, -1);
    EXPECT_EQ(state->writer_history[kFdwicSharedPaTaskCapacity - 1].magic, 0U);
    EXPECT_EQ(state->writer_history[kFdwicSharedPaTaskCapacity - 1].writer_task, -1);
    EXPECT_EQ(state->writer_history[kFdwicSharedPaTaskCapacity - 1].count, 0U);
}

TEST(FdwicSharedPaOutputRef, PlainAndInvalidFormsAreUnambiguous) {
    static_assert(sizeof(FdwicOutputRef) == 16);
    static_assert(alignof(FdwicOutputRef) == 4);
    static_assert(std::is_trivially_copyable_v<FdwicOutputRef>);

    const FdwicOutputRef invalid = fdwic_invalid_output_ref();
    EXPECT_EQ(invalid.producer_task_id, -1);
    EXPECT_EQ(invalid.output_slot, -1);
    EXPECT_EQ(invalid.flags, 0);
    EXPECT_EQ(invalid.view_ndims, 0);
    EXPECT_EQ(invalid.view_shape0, 0U);
    EXPECT_EQ(invalid.view_offset0, 0U);
    EXPECT_FALSE(fdwic_plain_output_ref(invalid));

    const FdwicOutputRef plain{17, 3, 0, 0, 0, 0};
    EXPECT_TRUE(fdwic_plain_output_ref(plain));

    FdwicOutputRef unsupported = plain;
    unsupported.flags = 1;
    EXPECT_FALSE(fdwic_plain_output_ref(unsupported));
    unsupported = plain;
    unsupported.view_ndims = 1;
    EXPECT_FALSE(fdwic_plain_output_ref(unsupported));
    unsupported = plain;
    unsupported.view_shape0 = 8;
    EXPECT_FALSE(fdwic_plain_output_ref(unsupported));
    unsupported = plain;
    unsupported.view_offset0 = 4;
    EXPECT_FALSE(fdwic_plain_output_ref(unsupported));

    SharedTaskOutputs outputs;
    outputs.reset(17);
    EXPECT_TRUE(outputs.empty());
    EXPECT_TRUE(outputs.add_output_ref(17, 0));
    EXPECT_TRUE(outputs.add_output_ref(17, 1));
    EXPECT_FALSE(outputs.add_output_ref(18, 2));
    EXPECT_FALSE(outputs.add_output_ref(17, 3));
    EXPECT_EQ(outputs.producer_task_id, 17);
    EXPECT_EQ(outputs.size(), 2U);
}

#else

TEST(FdwicPrivateTensorMapState, SharedSidecarDoesNotMoveThePrivateLayout) {
    EXPECT_EQ(sizeof(DistGlobal), kFdwicSharedTensorMapOffset);
    EXPECT_LE(sizeof(DistGlobal), kDistEngineGlobalStateSize);
}

#endif

}  // namespace

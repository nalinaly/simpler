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

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "common/host_api.h"
#include "hbg_static_execution_slot.h"

namespace {

using simpler::hbg::HbgPreparedStaticExecutionSlot;
using simpler::hbg::HbgStaticExecutionSlotLayout;
using simpler::hbg::HbgStaticExecutionSlotStatus;
using simpler::hbg::prepare_hbg_static_execution_slot;

struct FakeStaticArena {
    int setup_calls{0};
    int acquire_calls{0};
    int freeze_calls{0};
    int setup_rc{0};
    int freeze_rc{0};
    void *gm_heap{reinterpret_cast<void *>(0x100000)};
    void *shared_memory{reinterpret_cast<void *>(0x200000)};
    void *runtime_arena{reinterpret_cast<void *>(0x300000)};
    size_t gm_heap_capacity{0};
    size_t shared_memory_capacity{0};
    size_t runtime_arena_capacity{0};
};

int fake_setup_static_arena(void *runner_ctx, uint32_t, size_t gm_heap, size_t shared_memory, size_t runtime_arena) {
    auto *arena = static_cast<FakeStaticArena *>(runner_ctx);
    ++arena->setup_calls;
    arena->gm_heap_capacity = gm_heap;
    arena->shared_memory_capacity = shared_memory;
    arena->runtime_arena_capacity = runtime_arena;
    return arena->setup_rc;
}

void *fake_acquire_gm_heap(void *runner_ctx, uint32_t) {
    auto *arena = static_cast<FakeStaticArena *>(runner_ctx);
    ++arena->acquire_calls;
    return arena->gm_heap;
}

void *fake_acquire_shared_memory(void *runner_ctx, uint32_t) {
    auto *arena = static_cast<FakeStaticArena *>(runner_ctx);
    ++arena->acquire_calls;
    return arena->shared_memory;
}

void *fake_acquire_runtime_arena(void *runner_ctx, uint32_t) {
    auto *arena = static_cast<FakeStaticArena *>(runner_ctx);
    ++arena->acquire_calls;
    return arena->runtime_arena;
}

int fake_freeze_static_arena(
    void *runner_ctx, uint32_t, const void *gm_heap, size_t gm_heap_capacity, const void *shared_memory,
    size_t shared_memory_capacity, const void *runtime_arena, size_t runtime_arena_capacity
) {
    auto *arena = static_cast<FakeStaticArena *>(runner_ctx);
    ++arena->freeze_calls;
    EXPECT_EQ(gm_heap, arena->gm_heap);
    EXPECT_EQ(gm_heap_capacity, arena->gm_heap_capacity);
    EXPECT_EQ(shared_memory, arena->shared_memory);
    EXPECT_EQ(shared_memory_capacity, arena->shared_memory_capacity);
    EXPECT_EQ(runtime_arena, arena->runtime_arena);
    EXPECT_EQ(runtime_arena_capacity, arena->runtime_arena_capacity);
    return arena->freeze_rc;
}

const HostApiOps &host_api_ops(bool with_freeze = true) {
    static const HostApiOps with_freeze_ops = []() {
        HostApiOps ops{};
        ops.setup_static_arena = fake_setup_static_arena;
        ops.acquire_pooled_gm_heap = fake_acquire_gm_heap;
        ops.acquire_pooled_gm_sm = fake_acquire_shared_memory;
        ops.acquire_pooled_runtime_arena = fake_acquire_runtime_arena;
        ops.freeze_static_arena = fake_freeze_static_arena;
        return ops;
    }();
    static const HostApiOps without_freeze_ops = []() {
        HostApiOps ops = with_freeze_ops;
        ops.freeze_static_arena = nullptr;
        return ops;
    }();
    return with_freeze ? with_freeze_ops : without_freeze_ops;
}

HbgStaticExecutionSlotLayout make_layout() {
    return HbgStaticExecutionSlotLayout{
        .gm_heap_capacity = 0x10000,
        .shared_memory_capacity = 0x20000,
        .runtime_arena_capacity = 0x30000,
        .runtime_offset = 0x800,
    };
}

class HbgStaticExecutionSlotTest : public ::testing::Test {
protected:
    FakeStaticArena arena_;
    HostApi api_{&arena_, 0, 0, &host_api_ops()};
};

}  // namespace

TEST_F(HbgStaticExecutionSlotTest, PublishesOnlyAfterExactPlatformFreeze) {
    HbgPreparedStaticExecutionSlot slot{};
    ASSERT_EQ(prepare_hbg_static_execution_slot(&api_, make_layout(), &slot), HbgStaticExecutionSlotStatus::Ok);

    EXPECT_EQ(arena_.setup_calls, 1);
    EXPECT_EQ(arena_.acquire_calls, 3);
    EXPECT_EQ(arena_.freeze_calls, 1);
    EXPECT_EQ(slot.binding.gm_heap_base, reinterpret_cast<uint64_t>(arena_.gm_heap));
    EXPECT_EQ(slot.binding.shared_memory_base, reinterpret_cast<uint64_t>(arena_.shared_memory));
    EXPECT_EQ(slot.binding.runtime_arena_base, reinterpret_cast<uint64_t>(arena_.runtime_arena));
    EXPECT_EQ(slot.binding.gm_heap_capacity, 0x10000u);
    EXPECT_EQ(slot.binding.shared_memory_capacity, 0x20000u);
    EXPECT_EQ(slot.binding.runtime_arena_capacity, 0x30000u);
    EXPECT_EQ(slot.binding.runtime_offset, 0x800u);
    EXPECT_EQ(slot.binding.slot_generation, 0u);
}

TEST_F(HbgStaticExecutionSlotTest, InvalidCapacityHasNoPlatformSideEffects) {
    HbgStaticExecutionSlotLayout layout = make_layout();
    layout.runtime_offset = layout.runtime_arena_capacity;
    HbgPreparedStaticExecutionSlot slot{};

    EXPECT_EQ(prepare_hbg_static_execution_slot(&api_, layout, &slot), HbgStaticExecutionSlotStatus::InvalidCapacity);
    EXPECT_EQ(arena_.setup_calls, 0);
    EXPECT_EQ(arena_.acquire_calls, 0);
    EXPECT_EQ(arena_.freeze_calls, 0);
}

TEST_F(HbgStaticExecutionSlotTest, MissingFreezeCapabilityHasNoPlatformSideEffects) {
    api_ = HostApi(&arena_, 0, 0, &host_api_ops(false));
    HbgPreparedStaticExecutionSlot slot{};

    EXPECT_EQ(
        prepare_hbg_static_execution_slot(&api_, make_layout(), &slot),
        HbgStaticExecutionSlotStatus::MissingHostCapability
    );
    EXPECT_EQ(arena_.setup_calls, 0);
    EXPECT_EQ(arena_.acquire_calls, 0);
}

TEST_F(HbgStaticExecutionSlotTest, SetupFailureDoesNotAcquireOrFreeze) {
    arena_.setup_rc = -7;
    HbgPreparedStaticExecutionSlot slot{};

    EXPECT_EQ(
        prepare_hbg_static_execution_slot(&api_, make_layout(), &slot), HbgStaticExecutionSlotStatus::SetupFailed
    );
    EXPECT_EQ(arena_.setup_calls, 1);
    EXPECT_EQ(arena_.acquire_calls, 0);
    EXPECT_EQ(arena_.freeze_calls, 0);
}

TEST_F(HbgStaticExecutionSlotTest, MissingArenaBaseIsNeverFrozen) {
    arena_.runtime_arena = nullptr;
    HbgPreparedStaticExecutionSlot slot{};

    EXPECT_EQ(
        prepare_hbg_static_execution_slot(&api_, make_layout(), &slot), HbgStaticExecutionSlotStatus::AcquireFailed
    );
    EXPECT_EQ(arena_.setup_calls, 1);
    EXPECT_EQ(arena_.acquire_calls, 3);
    EXPECT_EQ(arena_.freeze_calls, 0);
}

TEST_F(HbgStaticExecutionSlotTest, OverlappingDeviceWindowsAreNeverFrozen) {
    arena_.runtime_arena = reinterpret_cast<void *>(0x208000);
    HbgPreparedStaticExecutionSlot slot{};

    EXPECT_EQ(
        prepare_hbg_static_execution_slot(&api_, make_layout(), &slot),
        HbgStaticExecutionSlotStatus::InvalidDeviceLayout
    );
    EXPECT_EQ(arena_.setup_calls, 1);
    EXPECT_EQ(arena_.acquire_calls, 3);
    EXPECT_EQ(arena_.freeze_calls, 0);
}

TEST_F(HbgStaticExecutionSlotTest, FreezeFailureDoesNotPublishCandidateBinding) {
    arena_.freeze_rc = -9;
    HbgPreparedStaticExecutionSlot slot{};
    slot.binding.gm_heap_base = 0xdeadbeef;

    EXPECT_EQ(
        prepare_hbg_static_execution_slot(&api_, make_layout(), &slot), HbgStaticExecutionSlotStatus::FreezeFailed
    );
    EXPECT_EQ(arena_.freeze_calls, 1);
    EXPECT_EQ(slot.binding.gm_heap_base, 0xdeadbeefu);
}

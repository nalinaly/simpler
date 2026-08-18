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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "common/host_api.h"
#include "hbg_launch_blob.h"

namespace simpler::hbg {

/** Capacity request computed from one HBG runtime configuration. */
struct HbgStaticExecutionSlotLayout {
    uint64_t gm_heap_capacity{0};
    uint64_t shared_memory_capacity{0};
    uint64_t runtime_arena_capacity{0};
    uint64_t runtime_offset{0};
};

/**
 * Allocated and address-frozen working regions for one HBG L1 context.
 *
 * `binding.slot_generation` deliberately stays zero here. DeviceRunner owns
 * that generation and adds it only when it seals the complete execution-slot
 * registration together with outer Runtime, KernelArgs and binary ownership.
 */
struct HbgPreparedStaticExecutionSlot {
    HbgExecutionBinding binding{};
};

static_assert(
    std::is_standard_layout_v<HbgStaticExecutionSlotLayout> &&
        std::is_trivially_copyable_v<HbgStaticExecutionSlotLayout>,
    "HBG static execution-slot layout must be a POD"
);
static_assert(
    std::is_standard_layout_v<HbgPreparedStaticExecutionSlot> &&
        std::is_trivially_copyable_v<HbgPreparedStaticExecutionSlot>,
    "HBG prepared static execution slot must be a POD"
);

enum class HbgStaticExecutionSlotStatus : uint32_t {
    Ok = 0,
    NullArgument,
    MissingHostCapability,
    InvalidCapacity,
    SetupFailed,
    AcquireFailed,
    InvalidDeviceLayout,
    FreezeFailed,
};

inline bool hbg_capacity_fits_size_t(uint64_t capacity) noexcept {
    return capacity != 0 && capacity <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

/**
 * Allocate, validate and freeze the three mutable HBG working regions.
 *
 * The output is published only after the platform owner verifies that the
 * acquired bases and exact capacities still name its live arena bank. A
 * failed call may leave platform-owned allocations for context teardown, but
 * it never publishes a partially trusted slot to the caller.
 */
inline HbgStaticExecutionSlotStatus prepare_hbg_static_execution_slot(
    const HostApi *api, const HbgStaticExecutionSlotLayout &layout, HbgPreparedStaticExecutionSlot *out
) noexcept {
    if (api == nullptr || out == nullptr) return HbgStaticExecutionSlotStatus::NullArgument;
    if (!api->supports_freeze_static_arena()) return HbgStaticExecutionSlotStatus::MissingHostCapability;
    if (!hbg_capacity_fits_size_t(layout.gm_heap_capacity) ||
        !hbg_capacity_fits_size_t(layout.shared_memory_capacity) ||
        !hbg_capacity_fits_size_t(layout.runtime_arena_capacity) ||
        layout.runtime_offset >= layout.runtime_arena_capacity) {
        return HbgStaticExecutionSlotStatus::InvalidCapacity;
    }

    const size_t gm_heap_capacity = static_cast<size_t>(layout.gm_heap_capacity);
    const size_t shared_memory_capacity = static_cast<size_t>(layout.shared_memory_capacity);
    const size_t runtime_arena_capacity = static_cast<size_t>(layout.runtime_arena_capacity);
    if (api->setup_static_arena(gm_heap_capacity, shared_memory_capacity, runtime_arena_capacity) != 0) {
        return HbgStaticExecutionSlotStatus::SetupFailed;
    }

    void *gm_heap = api->acquire_pooled_gm_heap();
    void *shared_memory = api->acquire_pooled_gm_sm();
    void *runtime_arena = api->acquire_pooled_runtime_arena();
    if (gm_heap == nullptr || shared_memory == nullptr || runtime_arena == nullptr) {
        return HbgStaticExecutionSlotStatus::AcquireFailed;
    }

    HbgExecutionBinding binding{};
    binding.shared_memory_base = reinterpret_cast<uint64_t>(shared_memory);
    binding.shared_memory_capacity = layout.shared_memory_capacity;
    binding.runtime_arena_base = reinterpret_cast<uint64_t>(runtime_arena);
    binding.runtime_arena_capacity = layout.runtime_arena_capacity;
    binding.gm_heap_base = reinterpret_cast<uint64_t>(gm_heap);
    binding.gm_heap_capacity = layout.gm_heap_capacity;
    binding.runtime_offset = layout.runtime_offset;

    if (!hbg_valid_device_window(binding.gm_heap_base, binding.gm_heap_capacity) ||
        !hbg_valid_device_window(binding.shared_memory_base, binding.shared_memory_capacity) ||
        !hbg_valid_device_window(binding.runtime_arena_base, binding.runtime_arena_capacity) ||
        hbg_ranges_overlap(
            binding.gm_heap_base, binding.gm_heap_capacity, binding.shared_memory_base, binding.shared_memory_capacity
        ) ||
        hbg_ranges_overlap(
            binding.gm_heap_base, binding.gm_heap_capacity, binding.runtime_arena_base, binding.runtime_arena_capacity
        ) ||
        hbg_ranges_overlap(
            binding.shared_memory_base, binding.shared_memory_capacity, binding.runtime_arena_base,
            binding.runtime_arena_capacity
        )) {
        return HbgStaticExecutionSlotStatus::InvalidDeviceLayout;
    }

    if (api->freeze_static_arena(
            gm_heap, gm_heap_capacity, shared_memory, shared_memory_capacity, runtime_arena, runtime_arena_capacity
        ) != 0) {
        return HbgStaticExecutionSlotStatus::FreezeFailed;
    }

    out->binding = binding;
    return HbgStaticExecutionSlotStatus::Ok;
}

}  // namespace simpler::hbg

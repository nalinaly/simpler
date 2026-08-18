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

#include "hbg_l1_launch_control.h"
#include "hbg_launch_blob.h"

namespace simpler::hbg {

inline constexpr uint32_t HBG_EXECUTION_SLOT_MAGIC = 0x31534748U;  // "HGS1" in little-endian memory.
inline constexpr uint16_t HBG_EXECUTION_SLOT_ABI_MAJOR = 1;
inline constexpr uint16_t HBG_EXECUTION_SLOT_ABI_MINOR = 1;

enum HbgExecutionSlotFlags : uint32_t {
    HBG_EXECUTION_SLOT_CAPACITY_FROZEN = 1U << 0,
    HBG_EXECUTION_SLOT_SERIAL_ONLY = 1U << 1,
};

inline constexpr uint32_t HBG_EXECUTION_SLOT_REQUIRED_FLAGS =
    HBG_EXECUTION_SLOT_CAPACITY_FROZEN | HBG_EXECUTION_SLOT_SERIAL_ONLY;

/**
 * Prepare-time trust root for one context-owned mutable HBG execution slot.
 *
 * The launch blob carries a destination binding for self-description, but it
 * is not allowed to authenticate itself.  Host prepare builds this record from
 * allocations it owns, seals it once, and later transfers the same immutable
 * bytes to the HBG AICPU registry.  A restore must compare the task-owned blob
 * against this record before writing any working byte.
 */
struct alignas(8) HbgExecutionSlotRegistration {
    uint32_t magic{HBG_EXECUTION_SLOT_MAGIC};
    uint16_t abi_major{HBG_EXECUTION_SLOT_ABI_MAJOR};
    uint16_t abi_minor{HBG_EXECUTION_SLOT_ABI_MINOR};
    uint32_t struct_size{sizeof(HbgExecutionSlotRegistration)};
    int32_t device_id{-1};
    uint32_t flags{HBG_EXECUTION_SLOT_REQUIRED_FLAGS};
    uint32_t prelaunch_control_offset{0};
    uint64_t max_launch_blob_size{0};
    HbgExecutionBinding binding{};
    uint64_t outer_runtime_base{0};
    uint64_t outer_runtime_size{0};
    uint64_t device_kernel_args_base{0};
    uint64_t device_kernel_args_size{0};
    uint64_t binary_generation{0};
    uint64_t registration_hash{0};
};

/** Host-owned inputs used to construct one complete registration. */
struct HbgExecutionSlotRegistrationSpec {
    int32_t device_id{-1};
    uint32_t prelaunch_control_offset{0};
    uint64_t max_launch_blob_size{0};
    HbgExecutionBinding binding{};
    uint64_t outer_runtime_base{0};
    uint64_t outer_runtime_size{0};
    uint64_t device_kernel_args_base{0};
    uint64_t device_kernel_args_size{0};
    uint64_t binary_generation{0};
};

static_assert(
    std::is_standard_layout_v<HbgExecutionSlotRegistrationSpec> &&
        std::is_trivially_copyable_v<HbgExecutionSlotRegistrationSpec>,
    "HBG execution-slot registration spec must be a host-copyable POD"
);

static_assert(sizeof(HbgExecutionSlotRegistration) == 144, "HBG execution-slot registration ABI changed");
static_assert(
    offsetof(HbgExecutionSlotRegistration, max_launch_blob_size) == 24, "HBG execution-slot capacity offset changed"
);
static_assert(
    offsetof(HbgExecutionSlotRegistration, prelaunch_control_offset) == 20, "HBG prelaunch-control offset changed"
);
static_assert(offsetof(HbgExecutionSlotRegistration, binding) == 32, "HBG execution-slot binding offset changed");
static_assert(
    offsetof(HbgExecutionSlotRegistration, registration_hash) == 136, "HBG execution-slot hash offset changed"
);
static_assert(
    std::is_standard_layout_v<HbgExecutionSlotRegistration>, "HBG execution-slot registration must be standard-layout"
);
static_assert(
    std::is_trivially_copyable_v<HbgExecutionSlotRegistration>, "HBG execution-slot registration must be byte-copyable"
);

enum class HbgExecutionSlotStatus : uint32_t {
    Ok = 0,
    NullArgument,
    InvalidMagic,
    UnsupportedVersion,
    InvalidHeader,
    InvalidFlags,
    InvalidDevice,
    DeviceMismatch,
    InvalidGeneration,
    InvalidBinding,
    InvalidRuntimeWindow,
    InvalidLaunchControl,
    InvalidKernelArgsWindow,
    InvalidBinaryGeneration,
    InvalidPackageCapacity,
    AddressOverlap,
    HashMismatch,
};

inline bool hbg_execution_slot_windows_overlap(
    uint64_t lhs_base, uint64_t lhs_size, uint64_t rhs_base, uint64_t rhs_size
) noexcept {
    if (lhs_size == 0 || rhs_size == 0) return false;
    return hbg_ranges_overlap(lhs_base, lhs_size, rhs_base, rhs_size);
}

inline bool hbg_minimum_launch_blob_size(const HbgExecutionBinding &binding, uint64_t *out) noexcept {
    if (out == nullptr) return false;
    const uint64_t descriptor_bytes = 2U * sizeof(HbgLaunchRegion);
    uint64_t header_size = 0;
    if (!hbg_checked_add_u64(sizeof(HbgLaunchBlobHeader), descriptor_bytes, &header_size) ||
        !hbg_checked_add_u64(header_size, HBG_LAUNCH_BLOB_ALIGNMENT - 1, &header_size)) {
        return false;
    }
    header_size &= ~(static_cast<uint64_t>(HBG_LAUNCH_BLOB_ALIGNMENT) - 1);
    uint64_t aligned_shared_memory = 0;
    if (!hbg_checked_add_u64(
            binding.shared_memory_capacity, static_cast<uint64_t>(HBG_LAUNCH_BLOB_ALIGNMENT - 1), &aligned_shared_memory
        )) {
        return false;
    }
    aligned_shared_memory &= ~(static_cast<uint64_t>(HBG_LAUNCH_BLOB_ALIGNMENT) - 1);

    uint64_t size = 0;
    if (!hbg_checked_add_u64(header_size, aligned_shared_memory, &size) ||
        !hbg_checked_add_u64(size, binding.runtime_arena_capacity, &size)) {
        return false;
    }
    *out = size;
    return true;
}

inline HbgExecutionSlotStatus hbg_validate_execution_slot_registration_fields(
    const HbgExecutionSlotRegistration &registration, int32_t expected_device_id = -1
) noexcept {
    if (registration.magic != HBG_EXECUTION_SLOT_MAGIC) return HbgExecutionSlotStatus::InvalidMagic;
    if (registration.abi_major != HBG_EXECUTION_SLOT_ABI_MAJOR ||
        registration.abi_minor != HBG_EXECUTION_SLOT_ABI_MINOR) {
        return HbgExecutionSlotStatus::UnsupportedVersion;
    }
    if (registration.struct_size != sizeof(HbgExecutionSlotRegistration)) {
        return HbgExecutionSlotStatus::InvalidHeader;
    }
    if (registration.flags != HBG_EXECUTION_SLOT_REQUIRED_FLAGS) {
        return HbgExecutionSlotStatus::InvalidFlags;
    }
    if (registration.device_id < 0) return HbgExecutionSlotStatus::InvalidDevice;
    if (expected_device_id >= 0 && registration.device_id != expected_device_id) {
        return HbgExecutionSlotStatus::DeviceMismatch;
    }
    if (registration.binding.slot_generation == 0) return HbgExecutionSlotStatus::InvalidGeneration;

    const HbgExecutionBinding &binding = registration.binding;
    if (!hbg_valid_device_window(binding.shared_memory_base, binding.shared_memory_capacity) ||
        !hbg_valid_device_window(binding.runtime_arena_base, binding.runtime_arena_capacity) ||
        binding.runtime_offset >= binding.runtime_arena_capacity ||
        (binding.gm_heap_base == 0) != (binding.gm_heap_capacity == 0) ||
        (binding.gm_heap_base != 0 && !hbg_valid_device_window(binding.gm_heap_base, binding.gm_heap_capacity))) {
        return HbgExecutionSlotStatus::InvalidBinding;
    }
    if (!hbg_valid_device_window(registration.outer_runtime_base, registration.outer_runtime_size)) {
        return HbgExecutionSlotStatus::InvalidRuntimeWindow;
    }
    uint64_t prelaunch_control_end = 0;
    if (registration.prelaunch_control_offset % alignof(HbgL1LaunchControl) != 0 ||
        !hbg_checked_add_u64(
            registration.prelaunch_control_offset, sizeof(HbgL1LaunchControl), &prelaunch_control_end
        ) ||
        prelaunch_control_end > registration.outer_runtime_size) {
        return HbgExecutionSlotStatus::InvalidLaunchControl;
    }
    if (!hbg_valid_device_window(registration.device_kernel_args_base, registration.device_kernel_args_size)) {
        return HbgExecutionSlotStatus::InvalidKernelArgsWindow;
    }
    if (registration.binary_generation == 0) return HbgExecutionSlotStatus::InvalidBinaryGeneration;

    uint64_t minimum_blob_size = 0;
    if (!hbg_minimum_launch_blob_size(binding, &minimum_blob_size) ||
        registration.max_launch_blob_size < minimum_blob_size ||
        registration.max_launch_blob_size > std::numeric_limits<uint32_t>::max()) {
        return HbgExecutionSlotStatus::InvalidPackageCapacity;
    }

    struct Window {
        uint64_t base;
        uint64_t size;
    };
    const Window windows[] = {
        {binding.shared_memory_base, binding.shared_memory_capacity},
        {binding.runtime_arena_base, binding.runtime_arena_capacity},
        {binding.gm_heap_base, binding.gm_heap_capacity},
        {registration.outer_runtime_base, registration.outer_runtime_size},
        {registration.device_kernel_args_base, registration.device_kernel_args_size},
    };
    for (size_t index = 0; index < sizeof(windows) / sizeof(windows[0]); ++index) {
        for (size_t previous = 0; previous < index; ++previous) {
            if (hbg_execution_slot_windows_overlap(
                    windows[index].base, windows[index].size, windows[previous].base, windows[previous].size
                )) {
                return HbgExecutionSlotStatus::AddressOverlap;
            }
        }
    }
    return HbgExecutionSlotStatus::Ok;
}

inline uint64_t hbg_execution_slot_registration_hash(const HbgExecutionSlotRegistration &registration) noexcept {
    return common::utils::fnv1a_64(&registration, offsetof(HbgExecutionSlotRegistration, registration_hash));
}

/** Resolve the trusted device control line embedded in outer Runtime. */
inline HbgL1LaunchControl *hbg_l1_launch_control(const HbgExecutionSlotRegistration &registration) noexcept {
    if (hbg_validate_execution_slot_registration_fields(registration) != HbgExecutionSlotStatus::Ok ||
        registration.registration_hash != hbg_execution_slot_registration_hash(registration)) {
        return nullptr;
    }
    uint64_t address = 0;
    if (!hbg_checked_add_u64(registration.outer_runtime_base, registration.prelaunch_control_offset, &address)) {
        return nullptr;
    }
    return reinterpret_cast<HbgL1LaunchControl *>(address);
}

/**
 * Resolve the per-context cancellation trust root without trusting a failed
 * per-invocation registry read.
 *
 * The fallback address is latched by simpler_aicpu_init from prepare-time
 * HostArgs before registration/run tasks. It is used only when the immutable
 * registration is missing or fails validation; callers still own the cache
 * publish required after storing CANCEL.
 */
inline HbgL1LaunchControl *hbg_l1_launch_control_or_fallback(
    const HbgExecutionSlotRegistration *registration, uint64_t fallback_address
) noexcept {
    if (registration != nullptr) {
        HbgL1LaunchControl *registered = hbg_l1_launch_control(*registration);
        if (registered != nullptr) return registered;
    }
    return reinterpret_cast<HbgL1LaunchControl *>(fallback_address);
}

/** Seal a host-built registration without publishing a partially valid hash. */
inline HbgExecutionSlotStatus seal_hbg_execution_slot_registration(
    HbgExecutionSlotRegistration *registration, int32_t expected_device_id = -1
) noexcept {
    if (registration == nullptr) return HbgExecutionSlotStatus::NullArgument;
    const HbgExecutionSlotStatus status =
        hbg_validate_execution_slot_registration_fields(*registration, expected_device_id);
    if (status != HbgExecutionSlotStatus::Ok) return status;
    registration->registration_hash = hbg_execution_slot_registration_hash(*registration);
    return HbgExecutionSlotStatus::Ok;
}

inline HbgExecutionSlotStatus validate_hbg_execution_slot_registration(
    const HbgExecutionSlotRegistration *registration, int32_t expected_device_id = -1
) noexcept {
    if (registration == nullptr) return HbgExecutionSlotStatus::NullArgument;
    const HbgExecutionSlotStatus status =
        hbg_validate_execution_slot_registration_fields(*registration, expected_device_id);
    if (status != HbgExecutionSlotStatus::Ok) return status;
    if (registration->registration_hash != hbg_execution_slot_registration_hash(*registration)) {
        return HbgExecutionSlotStatus::HashMismatch;
    }
    return HbgExecutionSlotStatus::Ok;
}

/**
 * Build and seal a registration without publishing a partial owner.
 *
 * The caller computes every generation and capacity before entering this
 * helper. A failed validation leaves `out` byte-for-byte unchanged.
 */
inline HbgExecutionSlotStatus build_hbg_execution_slot_registration(
    const HbgExecutionSlotRegistrationSpec &spec, HbgExecutionSlotRegistration *out
) noexcept {
    if (out == nullptr) return HbgExecutionSlotStatus::NullArgument;

    HbgExecutionSlotRegistration candidate{};
    candidate.device_id = spec.device_id;
    candidate.prelaunch_control_offset = spec.prelaunch_control_offset;
    candidate.max_launch_blob_size = spec.max_launch_blob_size;
    candidate.binding = spec.binding;
    candidate.outer_runtime_base = spec.outer_runtime_base;
    candidate.outer_runtime_size = spec.outer_runtime_size;
    candidate.device_kernel_args_base = spec.device_kernel_args_base;
    candidate.device_kernel_args_size = spec.device_kernel_args_size;
    candidate.binary_generation = spec.binary_generation;

    const HbgExecutionSlotStatus status = seal_hbg_execution_slot_registration(&candidate, spec.device_id);
    if (status != HbgExecutionSlotStatus::Ok) return status;
    *out = candidate;
    return HbgExecutionSlotStatus::Ok;
}

inline HbgExecutionSlotStatus
validate_hbg_launch_blob_size_for_slot(const HbgExecutionSlotRegistration &registration, size_t blob_size) noexcept {
    if (blob_size == 0 || blob_size > registration.max_launch_blob_size) {
        return HbgExecutionSlotStatus::InvalidPackageCapacity;
    }
    return HbgExecutionSlotStatus::Ok;
}

}  // namespace simpler::hbg

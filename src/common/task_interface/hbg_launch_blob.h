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

#include "arg_direction.h"
#include "host_args_launch.h"
#include "utils/fnv1a_64.h"

namespace simpler::hbg {

inline constexpr uint32_t HBG_LAUNCH_BLOB_MAGIC = 0x31474248U;  // "HBG1" in little-endian memory.
inline constexpr uint16_t HBG_LAUNCH_BLOB_ABI_MAJOR = 1;
inline constexpr uint16_t HBG_LAUNCH_BLOB_ABI_MINOR = 1;
inline constexpr uint32_t HBG_LAUNCH_BLOB_MAX_REGIONS = 1024;
inline constexpr size_t HBG_LAUNCH_BLOB_ALIGNMENT = 8;

enum class HbgLaunchRegionKind : uint32_t {
    SharedMemoryImage = 1,
    RuntimeArenaImage = 2,
    GmHeapInitializer = 3,
};

enum HbgLaunchBlobFlags : uint32_t {
    HBG_LAUNCH_DESTINATION_BOUND = 1U << 0,
};

enum HbgLaunchRegionFlags : uint32_t {
    HBG_REGION_REQUIRED = 1U << 0,
    HBG_REGION_IMMUTABLE_SOURCE = 1U << 1,
};

/**
 * Stable device destinations selected by one HBG execution slot.
 *
 * The pristine payload is task/node-owned, while these writable destinations
 * remain context-owned.  Absolute bases make the serialized image explicitly
 * destination-bound; a replay must reject a stale slot generation rather than
 * copying an old graph into a newly committed arena with different addresses.
 */
struct alignas(8) HbgExecutionBinding {
    uint64_t shared_memory_base{0};
    uint64_t shared_memory_capacity{0};
    uint64_t runtime_arena_base{0};
    uint64_t runtime_arena_capacity{0};
    uint64_t gm_heap_base{0};
    uint64_t gm_heap_capacity{0};
    uint64_t runtime_offset{0};
    uint64_t slot_generation{0};
};

/**
 * Immutable identity of the invocation whose pristine graph image is carried
 * by one launch blob.
 *
 * The three hashes intentionally describe different ownership domains:
 * callable_hash identifies the compiled operator, argument_snapshot_hash
 * identifies the tensor/scalar values baked into this graph image, and
 * function_binding_hash identifies the resolved AICore function table.  A
 * consumer must compare the identity with the callable selected for this
 * launch before restoring any bytes into the mutable execution slot.
 */
struct alignas(8) HbgInvocationIdentity {
    uint64_t callable_hash{0};
    uint64_t argument_snapshot_hash{0};
    uint64_t function_binding_hash{0};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    // The scheduler consumes this value on every invocation. Keeping it in
    // the task-owned graph snapshot prevents a later host build from
    // overwriting one context-wide Runtime::host_total_tasks before an older
    // eager task or captured node executes.
    int32_t host_total_tasks{0};
    uint32_t reserved{0};
};

/** One immutable source span and its offset within a mutable execution slot. */
struct alignas(8) HbgLaunchRegion {
    HbgLaunchRegionKind kind{HbgLaunchRegionKind::SharedMemoryImage};
    uint32_t flags{0};
    uint64_t source_offset{0};  // Relative to inline_payload_addr / header_size.
    uint64_t size{0};
    uint64_t destination_offset{0};
    uint64_t reserved{0};
};

/**
 * Variable-length HBG task payload header.
 *
 * Layout is:
 *   HbgLaunchBlobHeader
 *   region_count * HbgLaunchRegion
 *   zero padding through header_size
 *   immutable inline payload bytes
 *
 * The host serializer leaves inline_payload_addr zero.  A future
 * aclrtLaunchKernelWithHostArgs bridge may use aclrtPlaceHolderInfo to patch
 * that field to the runtime-owned device copy at header_size.  Keeping all
 * region offsets relative means the canonical host plan is never modified by
 * that per-launch patch.
 */
struct alignas(8) HbgLaunchBlobHeader {
    uint32_t magic{HBG_LAUNCH_BLOB_MAGIC};
    uint16_t abi_major{HBG_LAUNCH_BLOB_ABI_MAJOR};
    uint16_t abi_minor{HBG_LAUNCH_BLOB_ABI_MINOR};
    uint32_t header_size{0};
    uint32_t total_size{0};
    uint32_t region_count{0};
    uint32_t flags{HBG_LAUNCH_DESTINATION_BOUND};
    uint64_t plan_generation{0};
    uint64_t plan_hash{0};
    uint64_t inline_payload_addr{0};
    uint64_t inline_payload_size{0};
    HbgExecutionBinding binding{};
    HbgInvocationIdentity identity{};
};

static_assert(sizeof(HbgExecutionBinding) == 64, "HBG binding ABI changed");
static_assert(sizeof(HbgInvocationIdentity) == 40, "HBG invocation identity ABI changed");
static_assert(sizeof(HbgLaunchRegion) == 40, "HBG region ABI changed");
static_assert(sizeof(HbgLaunchBlobHeader) == 160, "HBG launch header ABI changed");
static_assert(offsetof(HbgLaunchBlobHeader, inline_payload_addr) == 40, "HBG placeholder offset changed");
static_assert(offsetof(HbgLaunchBlobHeader, binding) == 56, "HBG execution binding offset changed");
static_assert(offsetof(HbgLaunchBlobHeader, identity) == 120, "HBG invocation identity offset changed");
static_assert(alignof(HbgLaunchBlobHeader) == 8, "HBG host args require only runtime-guaranteed alignment");
static_assert(std::is_standard_layout_v<HbgLaunchBlobHeader>, "HBG launch header must be standard-layout");
static_assert(std::is_trivially_copyable_v<HbgLaunchBlobHeader>, "HBG launch header must be byte-copyable");
static_assert(std::is_trivially_copyable_v<HbgInvocationIdentity>, "HBG invocation identity must be byte-copyable");
static_assert(std::is_trivially_copyable_v<HbgLaunchRegion>, "HBG regions must be byte-copyable");

enum class HbgLaunchBlobAddressMode : uint32_t {
    HostUnpatched = 0,
    DevicePatched = 1,
    Either = 2,
};

enum class HbgLaunchBlobStatus : uint32_t {
    Ok = 0,
    NullArgument,
    Misaligned,
    InvalidMagic,
    UnsupportedVersion,
    InvalidHeader,
    InvalidFlags,
    InvalidGeneration,
    InvalidBinding,
    InvalidIdentity,
    IdentityMismatch,
    InvalidRegion,
    OutOfBounds,
    Overlap,
    HashMismatch,
    AllocationFailure,
};

inline bool hbg_checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t *out) noexcept {
    if (out == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

inline bool
hbg_ranges_overlap(uint64_t lhs_offset, uint64_t lhs_size, uint64_t rhs_offset, uint64_t rhs_size) noexcept {
    uint64_t lhs_end = 0;
    uint64_t rhs_end = 0;
    if (!hbg_checked_add_u64(lhs_offset, lhs_size, &lhs_end) || !hbg_checked_add_u64(rhs_offset, rhs_size, &rhs_end)) {
        return true;
    }
    return lhs_offset < rhs_end && rhs_offset < lhs_end;
}

inline bool hbg_valid_device_window(uint64_t base, uint64_t capacity) noexcept {
    return base != 0 && capacity != 0 && capacity <= std::numeric_limits<uint64_t>::max() - base;
}

inline bool
hbg_execution_binding_matches(const HbgExecutionBinding &actual, const HbgExecutionBinding &expected) noexcept {
    return actual.shared_memory_base == expected.shared_memory_base &&
           actual.shared_memory_capacity == expected.shared_memory_capacity &&
           actual.runtime_arena_base == expected.runtime_arena_base &&
           actual.runtime_arena_capacity == expected.runtime_arena_capacity &&
           actual.gm_heap_base == expected.gm_heap_base && actual.gm_heap_capacity == expected.gm_heap_capacity &&
           actual.runtime_offset == expected.runtime_offset;
}

inline bool
hbg_invocation_identity_matches(const HbgInvocationIdentity &actual, const HbgInvocationIdentity &expected) noexcept {
    return actual.callable_hash == expected.callable_hash &&
           actual.argument_snapshot_hash == expected.argument_snapshot_hash &&
           actual.function_binding_hash == expected.function_binding_hash &&
           actual.tensor_count == expected.tensor_count && actual.scalar_count == expected.scalar_count &&
           actual.host_total_tasks == expected.host_total_tasks && actual.reserved == expected.reserved;
}

inline bool hbg_valid_invocation_identity(const HbgInvocationIdentity &identity) noexcept {
    return identity.callable_hash != 0 && identity.function_binding_hash != 0 &&
           identity.tensor_count <= static_cast<uint32_t>(CHIP_MAX_TENSOR_ARGS) &&
           identity.scalar_count <= static_cast<uint32_t>(CHIP_MAX_SCALAR_ARGS) && identity.host_total_tasks >= 0 &&
           identity.reserved == 0;
}

inline uint64_t hbg_destination_base(const HbgExecutionBinding &binding, HbgLaunchRegionKind kind) noexcept {
    switch (kind) {
    case HbgLaunchRegionKind::SharedMemoryImage:
        return binding.shared_memory_base;
    case HbgLaunchRegionKind::RuntimeArenaImage:
        return binding.runtime_arena_base;
    case HbgLaunchRegionKind::GmHeapInitializer:
        return binding.gm_heap_base;
    }
    return 0;
}

inline uint64_t hbg_destination_capacity(const HbgExecutionBinding &binding, HbgLaunchRegionKind kind) noexcept {
    switch (kind) {
    case HbgLaunchRegionKind::SharedMemoryImage:
        return binding.shared_memory_capacity;
    case HbgLaunchRegionKind::RuntimeArenaImage:
        return binding.runtime_arena_capacity;
    case HbgLaunchRegionKind::GmHeapInitializer:
        return binding.gm_heap_capacity;
    }
    return 0;
}

inline const HbgLaunchRegion *hbg_launch_regions(const HbgLaunchBlobHeader *header) noexcept {
    return reinterpret_cast<const HbgLaunchRegion *>(reinterpret_cast<const uint8_t *>(header) + sizeof(*header));
}

inline const uint8_t *hbg_inline_payload(const HbgLaunchBlobHeader *header) noexcept {
    return reinterpret_cast<const uint8_t *>(header) + header->header_size;
}

inline uint64_t hbg_plan_hash(
    const HbgInvocationIdentity &identity, const HbgLaunchRegion *regions, uint32_t region_count,
    const uint8_t *payload, uint64_t payload_size
) noexcept {
    uint64_t hash = common::utils::fnv1a_64(&identity, sizeof(identity));
    hash = common::utils::fnv1a_64_append(hash, regions, static_cast<size_t>(region_count) * sizeof(*regions));
    return common::utils::fnv1a_64_append(hash, payload, static_cast<size_t>(payload_size));
}

/**
 * Validate an untrusted variable-length graph package before any restore.
 *
 * Full SM and runtime-arena images are mandatory and must cover their whole
 * frozen capacities.  That rule prevents replay from inheriting consumed
 * scheduler state that happened not to be overwritten by a smaller image.
 */
inline HbgLaunchBlobStatus validate_hbg_launch_blob(
    const void *blob, size_t blob_size, HbgLaunchBlobAddressMode address_mode,
    const HbgExecutionBinding *expected_binding = nullptr, const HbgInvocationIdentity *expected_identity = nullptr
) noexcept {
    if (blob == nullptr) return HbgLaunchBlobStatus::NullArgument;
    if (reinterpret_cast<uintptr_t>(blob) % alignof(HbgLaunchBlobHeader) != 0) {
        return HbgLaunchBlobStatus::Misaligned;
    }
    if (blob_size < sizeof(HbgLaunchBlobHeader)) return HbgLaunchBlobStatus::InvalidHeader;

    const auto *header = static_cast<const HbgLaunchBlobHeader *>(blob);
    if (header->magic != HBG_LAUNCH_BLOB_MAGIC) return HbgLaunchBlobStatus::InvalidMagic;
    if (header->abi_major != HBG_LAUNCH_BLOB_ABI_MAJOR || header->abi_minor != HBG_LAUNCH_BLOB_ABI_MINOR) {
        return HbgLaunchBlobStatus::UnsupportedVersion;
    }
    if (header->region_count < 2 || header->region_count > HBG_LAUNCH_BLOB_MAX_REGIONS) {
        return HbgLaunchBlobStatus::InvalidHeader;
    }

    const uint64_t descriptor_bytes = static_cast<uint64_t>(header->region_count) * sizeof(HbgLaunchRegion);
    uint64_t minimum_header_size = 0;
    uint64_t canonical_header_size = 0;
    if (!hbg_checked_add_u64(sizeof(HbgLaunchBlobHeader), descriptor_bytes, &minimum_header_size) ||
        !hbg_checked_add_u64(minimum_header_size, HBG_LAUNCH_BLOB_ALIGNMENT - 1, &canonical_header_size)) {
        return HbgLaunchBlobStatus::InvalidHeader;
    }
    canonical_header_size &= ~(static_cast<uint64_t>(HBG_LAUNCH_BLOB_ALIGNMENT) - 1);
    if (header->header_size != canonical_header_size || header->total_size != blob_size ||
        header->total_size < header->header_size ||
        header->inline_payload_size != static_cast<uint64_t>(header->total_size - header->header_size)) {
        return HbgLaunchBlobStatus::InvalidHeader;
    }
    if ((header->flags & ~HBG_LAUNCH_DESTINATION_BOUND) != 0 || (header->flags & HBG_LAUNCH_DESTINATION_BOUND) == 0) {
        return HbgLaunchBlobStatus::InvalidFlags;
    }
    if (header->plan_generation == 0 || header->binding.slot_generation == 0) {
        return HbgLaunchBlobStatus::InvalidGeneration;
    }
    const auto *header_bytes = static_cast<const uint8_t *>(blob);
    for (uint64_t offset = minimum_header_size; offset < canonical_header_size; ++offset) {
        if (header_bytes[offset] != 0) return HbgLaunchBlobStatus::InvalidHeader;
    }

    uint64_t expected_payload_addr = 0;
    if (!hbg_checked_add_u64(reinterpret_cast<uint64_t>(blob), header->header_size, &expected_payload_addr)) {
        return HbgLaunchBlobStatus::OutOfBounds;
    }
    if ((address_mode == HbgLaunchBlobAddressMode::HostUnpatched && header->inline_payload_addr != 0) ||
        (address_mode == HbgLaunchBlobAddressMode::DevicePatched &&
         header->inline_payload_addr != expected_payload_addr) ||
        (address_mode == HbgLaunchBlobAddressMode::Either && header->inline_payload_addr != 0 &&
         header->inline_payload_addr != expected_payload_addr)) {
        return HbgLaunchBlobStatus::InvalidHeader;
    }

    const HbgExecutionBinding &binding = header->binding;
    if (!hbg_valid_device_window(binding.shared_memory_base, binding.shared_memory_capacity) ||
        !hbg_valid_device_window(binding.runtime_arena_base, binding.runtime_arena_capacity) ||
        binding.runtime_offset >= binding.runtime_arena_capacity ||
        (binding.gm_heap_base == 0) != (binding.gm_heap_capacity == 0) ||
        (binding.gm_heap_base != 0 && !hbg_valid_device_window(binding.gm_heap_base, binding.gm_heap_capacity))) {
        return HbgLaunchBlobStatus::InvalidBinding;
    }
    if (expected_binding != nullptr) {
        if (binding.slot_generation != expected_binding->slot_generation) {
            return HbgLaunchBlobStatus::InvalidGeneration;
        }
        if (!hbg_execution_binding_matches(binding, *expected_binding)) {
            return HbgLaunchBlobStatus::InvalidBinding;
        }
    }

    const HbgInvocationIdentity &identity = header->identity;
    if (!hbg_valid_invocation_identity(identity)) {
        return HbgLaunchBlobStatus::InvalidIdentity;
    }
    if (expected_identity != nullptr && !hbg_invocation_identity_matches(identity, *expected_identity)) {
        return HbgLaunchBlobStatus::IdentityMismatch;
    }

    const HbgLaunchRegion *regions = hbg_launch_regions(header);
    bool saw_shared_memory = false;
    bool saw_runtime_arena = false;
    for (uint32_t i = 0; i < header->region_count; ++i) {
        const HbgLaunchRegion &region = regions[i];
        if ((region.flags & ~(HBG_REGION_REQUIRED | HBG_REGION_IMMUTABLE_SOURCE)) != 0 ||
            (region.flags & (HBG_REGION_REQUIRED | HBG_REGION_IMMUTABLE_SOURCE)) !=
                (HBG_REGION_REQUIRED | HBG_REGION_IMMUTABLE_SOURCE) ||
            region.reserved != 0 || region.size == 0 || region.source_offset % HBG_LAUNCH_BLOB_ALIGNMENT != 0) {
            return HbgLaunchBlobStatus::InvalidRegion;
        }

        const uint64_t destination_capacity = hbg_destination_capacity(binding, region.kind);
        const uint64_t destination_base = hbg_destination_base(binding, region.kind);
        uint64_t source_end = 0;
        uint64_t destination_end = 0;
        if (destination_base == 0 || destination_capacity == 0 ||
            !hbg_checked_add_u64(region.source_offset, region.size, &source_end) ||
            source_end > header->inline_payload_size ||
            !hbg_checked_add_u64(region.destination_offset, region.size, &destination_end) ||
            destination_end > destination_capacity) {
            return HbgLaunchBlobStatus::OutOfBounds;
        }

        switch (region.kind) {
        case HbgLaunchRegionKind::SharedMemoryImage:
            if (saw_shared_memory || region.destination_offset != 0 || region.size != binding.shared_memory_capacity) {
                return HbgLaunchBlobStatus::InvalidRegion;
            }
            saw_shared_memory = true;
            break;
        case HbgLaunchRegionKind::RuntimeArenaImage:
            if (saw_runtime_arena || region.destination_offset != 0 || region.size != binding.runtime_arena_capacity) {
                return HbgLaunchBlobStatus::InvalidRegion;
            }
            saw_runtime_arena = true;
            break;
        case HbgLaunchRegionKind::GmHeapInitializer:
            break;
        default:
            return HbgLaunchBlobStatus::InvalidRegion;
        }

        for (uint32_t previous = 0; previous < i; ++previous) {
            const HbgLaunchRegion &other = regions[previous];
            if (hbg_ranges_overlap(region.source_offset, region.size, other.source_offset, other.size) ||
                (region.kind == other.kind &&
                 hbg_ranges_overlap(region.destination_offset, region.size, other.destination_offset, other.size))) {
                return HbgLaunchBlobStatus::Overlap;
            }
        }
    }
    if (!saw_shared_memory || !saw_runtime_arena) return HbgLaunchBlobStatus::InvalidRegion;

    const uint8_t *payload = hbg_inline_payload(header);
    if (header->plan_hash !=
        hbg_plan_hash(identity, regions, header->region_count, payload, header->inline_payload_size)) {
        return HbgLaunchBlobStatus::HashMismatch;
    }
    return HbgLaunchBlobStatus::Ok;
}

/**
 * Prepare the single inline-payload placeholder consumed by the HBG launch.
 *
 * The canonical blob is validated in its unpatched state and is not modified.
 * The caller must give CANN a fresh writable copy because the runtime is
 * allowed to patch the pointer field while constructing its task-owned args.
 */
inline HbgLaunchBlobStatus make_hbg_launch_placeholder(
    const void *blob, size_t blob_size, host_args::HostArgsPlaceholder *out,
    const HbgExecutionBinding *expected_binding = nullptr, const HbgInvocationIdentity *expected_identity = nullptr
) noexcept {
    if (out == nullptr) return HbgLaunchBlobStatus::NullArgument;
    const HbgLaunchBlobStatus blob_status = validate_hbg_launch_blob(
        blob, blob_size, HbgLaunchBlobAddressMode::HostUnpatched, expected_binding, expected_identity
    );
    if (blob_status != HbgLaunchBlobStatus::Ok) return blob_status;

    const auto *header = static_cast<const HbgLaunchBlobHeader *>(blob);
    const host_args::HostArgsPlaceholder candidate{
        static_cast<uint32_t>(offsetof(HbgLaunchBlobHeader, inline_payload_addr)), header->header_size
    };
    if (host_args::validate_host_args_launch_layout(blob, blob_size, &candidate, 1) !=
        host_args::HostArgsLaunchStatus::Ok) {
        return HbgLaunchBlobStatus::InvalidHeader;
    }
    *out = candidate;
    return HbgLaunchBlobStatus::Ok;
}

}  // namespace simpler::hbg

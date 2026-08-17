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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "arg_direction.h"
#include "callable_protocol.h"
#include "utils/fnv1a_64.h"

namespace simpler::hbg {

inline constexpr uint32_t HBG_CALLABLE_REGISTRATION_MAGIC = 0x31434748U;  // "HGC1" in little-endian memory.
inline constexpr uint16_t HBG_CALLABLE_REGISTRATION_ABI_MAJOR = 1;
inline constexpr uint16_t HBG_CALLABLE_REGISTRATION_ABI_MINOR = 0;
inline constexpr uint32_t HBG_CALLABLE_REGISTRATION_SIZE = 56;

/**
 * Immutable callable trust root installed in the HBG AICPU binary.
 *
 * `callable_id` is context-global, while every callable owns an independent
 * func-id namespace.  The function-binding hash therefore authenticates one
 * callable-local 1024-entry function table rather than a context-global table.
 * The registry is append-only for the lifetime of the loaded AICPU binary.
 */
struct alignas(8) HbgCallableRegistration {
    uint32_t magic{HBG_CALLABLE_REGISTRATION_MAGIC};
    uint16_t abi_major{HBG_CALLABLE_REGISTRATION_ABI_MAJOR};
    uint16_t abi_minor{HBG_CALLABLE_REGISTRATION_ABI_MINOR};
    uint32_t struct_size{HBG_CALLABLE_REGISTRATION_SIZE};
    int32_t callable_id{-1};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    uint32_t reserved0{0};
    uint32_t reserved1{0};
    uint64_t callable_hash{0};
    uint64_t function_binding_hash{0};
    uint64_t registration_hash{0};
};

static_assert(sizeof(HbgCallableRegistration) == HBG_CALLABLE_REGISTRATION_SIZE, "HBG callable ABI changed");
static_assert(offsetof(HbgCallableRegistration, callable_hash) == 32, "HBG callable hash offset changed");
static_assert(offsetof(HbgCallableRegistration, registration_hash) == 48, "HBG registration hash offset changed");
static_assert(std::is_standard_layout_v<HbgCallableRegistration>, "HBG callable registration must be standard-layout");
static_assert(std::is_trivially_copyable_v<HbgCallableRegistration>, "HBG callable registration must be byte-copyable");

enum class HbgCallableStatus : uint32_t {
    Ok = 0,
    NullArgument,
    InvalidMagic,
    UnsupportedVersion,
    InvalidStructSize,
    InvalidCallableId,
    InvalidArgumentCount,
    InvalidReserved,
    InvalidCallableHash,
    InvalidFunctionBindingHash,
    HashMismatch,
};

inline uint64_t hbg_callable_registration_hash(const HbgCallableRegistration &registration) noexcept {
    uint64_t hash = common::utils::fnv1a_64(&registration.magic, sizeof(registration.magic));
    hash = common::utils::fnv1a_64_append(hash, &registration.abi_major, sizeof(registration.abi_major));
    hash = common::utils::fnv1a_64_append(hash, &registration.abi_minor, sizeof(registration.abi_minor));
    hash = common::utils::fnv1a_64_append(hash, &registration.struct_size, sizeof(registration.struct_size));
    hash = common::utils::fnv1a_64_append(hash, &registration.callable_id, sizeof(registration.callable_id));
    hash = common::utils::fnv1a_64_append(hash, &registration.tensor_count, sizeof(registration.tensor_count));
    hash = common::utils::fnv1a_64_append(hash, &registration.scalar_count, sizeof(registration.scalar_count));
    hash = common::utils::fnv1a_64_append(hash, &registration.reserved0, sizeof(registration.reserved0));
    hash = common::utils::fnv1a_64_append(hash, &registration.reserved1, sizeof(registration.reserved1));
    hash = common::utils::fnv1a_64_append(hash, &registration.callable_hash, sizeof(registration.callable_hash));
    return common::utils::fnv1a_64_append(
        hash, &registration.function_binding_hash, sizeof(registration.function_binding_hash)
    );
}

inline HbgCallableStatus
validate_hbg_callable_registration(const HbgCallableRegistration *registration, bool verify_hash = true) noexcept {
    if (registration == nullptr) return HbgCallableStatus::NullArgument;
    if (registration->magic != HBG_CALLABLE_REGISTRATION_MAGIC) return HbgCallableStatus::InvalidMagic;
    if (registration->abi_major != HBG_CALLABLE_REGISTRATION_ABI_MAJOR ||
        registration->abi_minor != HBG_CALLABLE_REGISTRATION_ABI_MINOR) {
        return HbgCallableStatus::UnsupportedVersion;
    }
    if (registration->struct_size != sizeof(HbgCallableRegistration)) return HbgCallableStatus::InvalidStructSize;
    if (registration->callable_id < 0 || registration->callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        return HbgCallableStatus::InvalidCallableId;
    }
    if (registration->tensor_count > static_cast<uint32_t>(CHIP_MAX_TENSOR_ARGS) ||
        registration->scalar_count > static_cast<uint32_t>(CHIP_MAX_SCALAR_ARGS)) {
        return HbgCallableStatus::InvalidArgumentCount;
    }
    if (registration->reserved0 != 0 || registration->reserved1 != 0) return HbgCallableStatus::InvalidReserved;
    if (registration->callable_hash == 0) return HbgCallableStatus::InvalidCallableHash;
    if (registration->function_binding_hash == 0) return HbgCallableStatus::InvalidFunctionBindingHash;
    if (verify_hash && registration->registration_hash != hbg_callable_registration_hash(*registration)) {
        return HbgCallableStatus::HashMismatch;
    }
    return HbgCallableStatus::Ok;
}

inline HbgCallableStatus seal_hbg_callable_registration(HbgCallableRegistration *registration) noexcept {
    const HbgCallableStatus status = validate_hbg_callable_registration(registration, false);
    if (status != HbgCallableStatus::Ok) return status;
    registration->registration_hash = hbg_callable_registration_hash(*registration);
    return HbgCallableStatus::Ok;
}

enum class HbgCallableRegistryPhase : uint32_t {
    Empty = 0,
    Publishing = 1,
    Ready = 2,
};

enum class HbgCallableRegistryStatus : uint32_t {
    Published = 0,
    Acquired,
    AlreadyRegistered,
    NullArgument,
    CallableRejected,
    InvalidCallableId,
    NotReady,
    Publishing,
    Conflict,
    CorruptState,
};

struct alignas(8) HbgCallableRegistryEntry {
    std::atomic<HbgCallableRegistryPhase> phase{HbgCallableRegistryPhase::Empty};
    HbgCallableRegistration registration{};
};

struct alignas(8) HbgCallableRegistry {
    HbgCallableRegistryEntry entries[MAX_REGISTERED_CALLABLE_IDS]{};
};

inline HbgCallableRegistryStatus
publish_hbg_callable_registration(HbgCallableRegistry *registry, const HbgCallableRegistration *registration) noexcept {
    if (registry == nullptr || registration == nullptr) return HbgCallableRegistryStatus::NullArgument;

    const HbgCallableRegistration candidate = *registration;
    if (validate_hbg_callable_registration(&candidate) != HbgCallableStatus::Ok) {
        return HbgCallableRegistryStatus::CallableRejected;
    }
    HbgCallableRegistryEntry &entry = registry->entries[candidate.callable_id];
    for (;;) {
        HbgCallableRegistryPhase phase = entry.phase.load(std::memory_order_acquire);
        if (phase == HbgCallableRegistryPhase::Ready) {
            return std::memcmp(&entry.registration, &candidate, sizeof(candidate)) == 0 ?
                       HbgCallableRegistryStatus::AlreadyRegistered :
                       HbgCallableRegistryStatus::Conflict;
        }
        if (phase == HbgCallableRegistryPhase::Publishing) return HbgCallableRegistryStatus::Publishing;
        if (phase != HbgCallableRegistryPhase::Empty) return HbgCallableRegistryStatus::CorruptState;
        if (!entry.phase.compare_exchange_strong(
                phase, HbgCallableRegistryPhase::Publishing, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            continue;
        }
        entry.registration = candidate;
        entry.phase.store(HbgCallableRegistryPhase::Ready, std::memory_order_release);
        return HbgCallableRegistryStatus::Published;
    }
}

inline HbgCallableRegistryStatus acquire_hbg_callable_registration(
    const HbgCallableRegistry *registry, int32_t callable_id, HbgCallableRegistration *out
) noexcept {
    if (registry == nullptr || out == nullptr) return HbgCallableRegistryStatus::NullArgument;
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        return HbgCallableRegistryStatus::InvalidCallableId;
    }

    const HbgCallableRegistryEntry &entry = registry->entries[callable_id];
    const HbgCallableRegistryPhase phase = entry.phase.load(std::memory_order_acquire);
    if (phase == HbgCallableRegistryPhase::Empty) return HbgCallableRegistryStatus::NotReady;
    if (phase == HbgCallableRegistryPhase::Publishing) return HbgCallableRegistryStatus::Publishing;
    if (phase != HbgCallableRegistryPhase::Ready) return HbgCallableRegistryStatus::CorruptState;

    const HbgCallableRegistration candidate = entry.registration;
    if (candidate.callable_id != callable_id ||
        validate_hbg_callable_registration(&candidate) != HbgCallableStatus::Ok) {
        return HbgCallableRegistryStatus::CorruptState;
    }
    *out = candidate;
    return HbgCallableRegistryStatus::Acquired;
}

}  // namespace simpler::hbg

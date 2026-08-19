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

#include "hbg_execution_slot_registry.h"

namespace simpler::hbg {

inline constexpr uint32_t HBG_CONTEXT_REGISTRY_MAGIC = 0x52474248U;  // "HBGR"
inline constexpr uint16_t HBG_CONTEXT_REGISTRY_ABI_MAJOR = 1;
inline constexpr uint16_t HBG_CONTEXT_REGISTRY_ABI_MINOR = 1;

/**
 * One borrowed-L1 context's complete HBG device registry.
 *
 * The DeviceRunner allocates and initializes this object in device memory
 * during prepare and owns it until a successful explicit close.  The resident
 * AICPU DSO keeps only the current context's address; no registration or
 * generation state survives in DSO storage.  Under the v1 single-live-context
 * contract, the ordered init task publishes that address before any register
 * or invocation task can consume it.
 */
struct alignas(64) HbgContextRegistry {
    uint32_t magic{HBG_CONTEXT_REGISTRY_MAGIC};
    uint16_t abi_major{HBG_CONTEXT_REGISTRY_ABI_MAJOR};
    uint16_t abi_minor{HBG_CONTEXT_REGISTRY_ABI_MINOR};
    uint32_t struct_size{0};
    uint32_t reserved{0};
    uint64_t context_generation{0};
    HbgExecutionSlotRegistry execution_slot{};
};

static_assert(alignof(HbgContextRegistry) == 64, "HBG context registry must remain cache-line aligned");
static_assert(sizeof(HbgContextRegistry) % 64 == 0, "HBG context registry size must cover complete cache lines");
static_assert(
    offsetof(HbgContextRegistry, execution_slot) % alignof(HbgExecutionSlotRegistry) == 0,
    "HBG execution-slot registry alignment changed"
);
inline bool valid_hbg_context_registry_header(const HbgContextRegistry *registry) noexcept {
    return registry != nullptr && registry->magic == HBG_CONTEXT_REGISTRY_MAGIC &&
           registry->abi_major == HBG_CONTEXT_REGISTRY_ABI_MAJOR &&
           registry->abi_minor == HBG_CONTEXT_REGISTRY_ABI_MINOR && registry->struct_size == sizeof(*registry) &&
           registry->reserved == 0 && registry->context_generation != 0;
}

inline bool initialize_hbg_context_registry(HbgContextRegistry *registry, uint64_t context_generation) noexcept {
    if (registry == nullptr || context_generation == 0) return false;
    registry->magic = HBG_CONTEXT_REGISTRY_MAGIC;
    registry->abi_major = HBG_CONTEXT_REGISTRY_ABI_MAJOR;
    registry->abi_minor = HBG_CONTEXT_REGISTRY_ABI_MINOR;
    registry->struct_size = sizeof(*registry);
    registry->reserved = 0;
    registry->context_generation = context_generation;
    reset_hbg_execution_slot_registry(&registry->execution_slot);
    return true;
}

}  // namespace simpler::hbg

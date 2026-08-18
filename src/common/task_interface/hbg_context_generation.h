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
#include <cstdint>

#include "hbg_callable_registry.h"
#include "hbg_execution_slot_registry.h"

namespace simpler::hbg {

enum class HbgContextGenerationStatus : uint32_t {
    Began = 0,
    AlreadyCurrent,
    NullArgument,
    InvalidGeneration,
    StaleGeneration,
};

// Legacy helper retained for the currently out-of-scope A5 runtime. A2/A3 L1
// uses HbgContextRegistry and stores no registry or generation in resident DSO
// state.
inline HbgContextGenerationStatus begin_hbg_context_generation(
    std::atomic<uint64_t> *current_generation, HbgExecutionSlotRegistry *execution_slot_registry,
    HbgCallableRegistry *callable_registry, uint64_t requested_generation
) noexcept {
    if (current_generation == nullptr || execution_slot_registry == nullptr || callable_registry == nullptr) {
        return HbgContextGenerationStatus::NullArgument;
    }
    if (requested_generation == 0) return HbgContextGenerationStatus::InvalidGeneration;

    const uint64_t current = current_generation->load(std::memory_order_acquire);
    if (current == requested_generation) return HbgContextGenerationStatus::AlreadyCurrent;
    if (current != 0 && requested_generation < current) return HbgContextGenerationStatus::StaleGeneration;

    reset_hbg_execution_slot_registry(execution_slot_registry);
    reset_hbg_callable_registry(callable_registry);
    current_generation->store(requested_generation, std::memory_order_release);
    return HbgContextGenerationStatus::Began;
}

}  // namespace simpler::hbg

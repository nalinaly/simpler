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
#include <cstring>

#include "hbg_execution_slot.h"

namespace simpler::hbg {

/**
 * Publication state for the HBG execution-slot registration resident in the
 * AICPU runtime DSO.
 *
 * A registration is immutable within one borrowed-L1 context generation.
 * CANN may keep the inner AICPU DSO resident after the host unloads its ACL
 * binary handle, so the next externally-quiesced context explicitly resets
 * this registry from its ordered init task before publishing a new trust root.
 */
enum class HbgExecutionSlotRegistryPhase : uint32_t {
    Empty = 0,
    Publishing = 1,
    Ready = 2,
};

enum class HbgExecutionSlotRegistryStatus : uint32_t {
    Published = 0,
    Acquired,
    AlreadyRegistered,
    NullArgument,
    SlotRejected,
    NotReady,
    Publishing,
    Conflict,
    CorruptState,
};

/**
 * Device-process owner for exactly one immutable execution-slot trust root.
 *
 * A writer validates a local byte snapshot before claiming Publishing, copies
 * the complete registration, then release-publishes Ready.  Readers acquire
 * Ready before copying and revalidating.  Once Ready, no field is modified;
 * an identical duplicate is idempotent and every differing duplicate fails
 * closed.
 */
struct alignas(8) HbgExecutionSlotRegistry {
    std::atomic<HbgExecutionSlotRegistryPhase> phase{HbgExecutionSlotRegistryPhase::Empty};
    HbgExecutionSlotRegistration registration{};
};

inline void reset_hbg_execution_slot_registry(HbgExecutionSlotRegistry *registry) noexcept {
    if (registry == nullptr) return;
    // Publishing is a fail-closed transient for an accidental concurrent
    // reader. The borrowed-L1 contract additionally requires the previous
    // context to be externally quiescent before its host owner closes it.
    registry->phase.store(HbgExecutionSlotRegistryPhase::Publishing, std::memory_order_release);
    registry->registration = HbgExecutionSlotRegistration{};
    registry->phase.store(HbgExecutionSlotRegistryPhase::Empty, std::memory_order_release);
}

inline HbgExecutionSlotRegistryStatus publish_hbg_execution_slot_registration(
    HbgExecutionSlotRegistry *registry, const HbgExecutionSlotRegistration *registration, int32_t expected_device_id
) noexcept {
    if (registry == nullptr || registration == nullptr) return HbgExecutionSlotRegistryStatus::NullArgument;

    const HbgExecutionSlotRegistration candidate = *registration;
    if (validate_hbg_execution_slot_registration(&candidate, expected_device_id) != HbgExecutionSlotStatus::Ok) {
        return HbgExecutionSlotRegistryStatus::SlotRejected;
    }

    for (;;) {
        HbgExecutionSlotRegistryPhase phase = registry->phase.load(std::memory_order_acquire);
        if (phase == HbgExecutionSlotRegistryPhase::Ready) {
            return std::memcmp(&registry->registration, &candidate, sizeof(candidate)) == 0 ?
                       HbgExecutionSlotRegistryStatus::AlreadyRegistered :
                       HbgExecutionSlotRegistryStatus::Conflict;
        }
        if (phase == HbgExecutionSlotRegistryPhase::Publishing) {
            return HbgExecutionSlotRegistryStatus::Publishing;
        }
        if (phase != HbgExecutionSlotRegistryPhase::Empty) {
            return HbgExecutionSlotRegistryStatus::CorruptState;
        }

        if (!registry->phase.compare_exchange_strong(
                phase, HbgExecutionSlotRegistryPhase::Publishing, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            continue;
        }
        registry->registration = candidate;
        registry->phase.store(HbgExecutionSlotRegistryPhase::Ready, std::memory_order_release);
        return HbgExecutionSlotRegistryStatus::Published;
    }
}

inline HbgExecutionSlotRegistryStatus acquire_hbg_execution_slot_registration(
    const HbgExecutionSlotRegistry *registry, int32_t expected_device_id, HbgExecutionSlotRegistration *out
) noexcept {
    if (registry == nullptr || out == nullptr) return HbgExecutionSlotRegistryStatus::NullArgument;

    const HbgExecutionSlotRegistryPhase phase = registry->phase.load(std::memory_order_acquire);
    if (phase == HbgExecutionSlotRegistryPhase::Empty) return HbgExecutionSlotRegistryStatus::NotReady;
    if (phase == HbgExecutionSlotRegistryPhase::Publishing) return HbgExecutionSlotRegistryStatus::Publishing;
    if (phase != HbgExecutionSlotRegistryPhase::Ready) return HbgExecutionSlotRegistryStatus::CorruptState;

    const HbgExecutionSlotRegistration candidate = registry->registration;
    if (validate_hbg_execution_slot_registration(&candidate, expected_device_id) != HbgExecutionSlotStatus::Ok) {
        return HbgExecutionSlotRegistryStatus::CorruptState;
    }
    *out = candidate;
    return HbgExecutionSlotRegistryStatus::Acquired;
}

}  // namespace simpler::hbg

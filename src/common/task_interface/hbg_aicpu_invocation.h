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
#include <cstring>
#include <type_traits>

#include "hbg_callable_registry.h"
#include "hbg_execution_slot.h"

namespace simpler::hbg {

/**
 * Per-AICPU-thread view of one runtime-owned HBG task package.
 *
 * The variable-length blob remains owned by CANN. Fixed headers and both
 * prepare-time trust roots are copied into aligned local storage before typed
 * access. Only the HBG boot leader performs full payload validation and
 * restore; other AICPU threads carry the same immutable view through the
 * platform affinity gate.
 */
struct alignas(8) HbgAicpuInvocationView {
    const void *blob{nullptr};
    uint64_t blob_size{0};
    HbgLaunchBlobHeader header{};
    HbgExecutionSlotRegistration slot{};
    HbgCallableRegistration callable{};
};

static_assert(
    std::is_standard_layout_v<HbgAicpuInvocationView> && std::is_trivially_copyable_v<HbgAicpuInvocationView>,
    "HBG AICPU invocation view must remain stack-copyable"
);

enum class HbgAicpuInvocationStatus : uint32_t {
    Ok = 0,
    NullArgument,
    SlotRejected,
    CallableRejected,
    InvalidHeader,
    InvalidPackageSize,
    InvalidPatchedAddress,
    BindingMismatch,
    CallableMismatch,
};

/**
 * Validate the fixed prefix without scanning the variable payload.
 *
 * This function is intentionally cheaper than validate_hbg_launch_blob(). It
 * establishes enough identity to select trusted KernelArgs and pass through
 * the platform affinity gate. The unique boot leader later validates the full
 * descriptor table, immutable payload, plan hash, and every restore span.
 */
inline HbgAicpuInvocationStatus make_hbg_aicpu_invocation_view(
    const void *blob, const HbgExecutionSlotRegistration &slot, const HbgCallableRegistration &callable,
    HbgAicpuInvocationView *out
) noexcept {
    if (blob == nullptr || out == nullptr) return HbgAicpuInvocationStatus::NullArgument;
    if (validate_hbg_execution_slot_registration(&slot) != HbgExecutionSlotStatus::Ok) {
        return HbgAicpuInvocationStatus::SlotRejected;
    }
    if (validate_hbg_callable_registration(&callable) != HbgCallableStatus::Ok) {
        return HbgAicpuInvocationStatus::CallableRejected;
    }

    HbgLaunchBlobHeader header{};
    std::memcpy(&header, blob, sizeof(header));
    if (header.magic != HBG_LAUNCH_BLOB_MAGIC || header.abi_major != HBG_LAUNCH_BLOB_ABI_MAJOR ||
        header.abi_minor != HBG_LAUNCH_BLOB_ABI_MINOR || header.region_count < 2 ||
        header.region_count > HBG_LAUNCH_BLOB_MAX_REGIONS || header.header_size < sizeof(header) ||
        header.header_size > header.total_size ||
        header.inline_payload_size != static_cast<uint64_t>(header.total_size - header.header_size) ||
        header.plan_generation == 0 || !hbg_valid_invocation_identity(header.identity)) {
        return HbgAicpuInvocationStatus::InvalidHeader;
    }
    if (validate_hbg_launch_blob_size_for_slot(slot, header.total_size) != HbgExecutionSlotStatus::Ok) {
        return HbgAicpuInvocationStatus::InvalidPackageSize;
    }

    uint64_t expected_payload = 0;
    if (!hbg_checked_add_u64(reinterpret_cast<uint64_t>(blob), header.header_size, &expected_payload) ||
        header.inline_payload_addr != expected_payload) {
        return HbgAicpuInvocationStatus::InvalidPatchedAddress;
    }
    if (header.binding.slot_generation != slot.binding.slot_generation ||
        !hbg_execution_binding_matches(header.binding, slot.binding)) {
        return HbgAicpuInvocationStatus::BindingMismatch;
    }
    if (header.identity.callable_id != callable.callable_id ||
        header.identity.callable_hash != callable.callable_hash ||
        header.identity.function_binding_hash != callable.function_binding_hash ||
        header.identity.tensor_count != callable.tensor_count ||
        header.identity.scalar_count != callable.scalar_count) {
        return HbgAicpuInvocationStatus::CallableMismatch;
    }

    HbgAicpuInvocationView candidate{};
    candidate.blob = blob;
    candidate.blob_size = header.total_size;
    candidate.header = header;
    candidate.slot = slot;
    candidate.callable = callable;
    *out = candidate;
    return HbgAicpuInvocationStatus::Ok;
}

}  // namespace simpler::hbg

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

#include "hbg_execution_slot.h"

namespace simpler::hbg {

/**
 * Published only after every pristine span reached the mutable working slot.
 * A scheduler must never treat partially copied bytes as a committed epoch.
 */
struct alignas(8) HbgRestoreCommit {
    uint64_t slot_generation{0};
    uint64_t plan_generation{0};
    uint64_t plan_hash{0};
    HbgInvocationIdentity identity{};
};

static_assert(sizeof(HbgRestoreCommit) == 56, "HBG restore commit ABI changed");
static_assert(std::is_trivially_copyable_v<HbgRestoreCommit>, "HBG restore commit must be byte-copyable");

enum class HbgRestoreStatus : uint32_t {
    Ok = 0,
    InvalidArguments,
    SlotRejected,
    BlobRejected,
    AddressOverflow,
    CopyFailed,
    PublishFailed,
};

struct HbgRestoreResult {
    HbgRestoreStatus status{HbgRestoreStatus::InvalidArguments};
    HbgExecutionSlotStatus slot_status{HbgExecutionSlotStatus::NullArgument};
    HbgLaunchBlobStatus blob_status{HbgLaunchBlobStatus::NullArgument};
    uint32_t region_index{std::numeric_limits<uint32_t>::max()};
    int callback_error{0};
};

using HbgRestoreCopyFn =
    int (*)(void *context, HbgLaunchRegionKind kind, void *destination, const void *source, size_t size) noexcept;
using HbgRestorePublishFn =
    int (*)(void *context, HbgLaunchRegionKind kind, const void *destination, size_t size) noexcept;

struct HbgRestoreOps {
    void *context{nullptr};
    HbgRestoreCopyFn copy{nullptr};
    HbgRestorePublishFn publish{nullptr};
};

/**
 * Restore one task-owned pristine image into a trusted mutable execution slot.
 *
 * The expected binding and invocation identity come from prepare-time state;
 * they must not be derived from the untrusted blob itself.  The blob must
 * already be in its runtime-owned, device-patched address state.  The output
 * commit is transactional: failure leaves the caller's prior commit untouched,
 * even though a failed copy callback may have partially written the slot.  The
 * caller must keep that slot non-dispatchable until a later full restore or
 * teardown.
 */
inline HbgRestoreResult restore_hbg_launch_blob(
    const void *blob, size_t blob_size, const HbgExecutionSlotRegistration &slot_registration,
    const HbgInvocationIdentity &expected_identity, const HbgRestoreOps &ops, HbgRestoreCommit *out_commit
) noexcept {
    HbgRestoreResult result;
    if (blob == nullptr || out_commit == nullptr || ops.copy == nullptr || ops.publish == nullptr) {
        result.status = HbgRestoreStatus::InvalidArguments;
        return result;
    }

    result.slot_status = validate_hbg_execution_slot_registration(&slot_registration);
    if (result.slot_status != HbgExecutionSlotStatus::Ok) {
        result.status = HbgRestoreStatus::SlotRejected;
        return result;
    }
    result.slot_status = validate_hbg_launch_blob_size_for_slot(slot_registration, blob_size);
    if (result.slot_status != HbgExecutionSlotStatus::Ok) {
        result.status = HbgRestoreStatus::SlotRejected;
        return result;
    }

    result.blob_status = validate_hbg_launch_blob(
        blob, blob_size, HbgLaunchBlobAddressMode::DevicePatched, &slot_registration.binding, &expected_identity
    );
    if (result.blob_status != HbgLaunchBlobStatus::Ok) {
        result.status = HbgRestoreStatus::BlobRejected;
        return result;
    }

    const auto *header = static_cast<const HbgLaunchBlobHeader *>(blob);
    const HbgLaunchRegion *regions = hbg_launch_regions(header);
    for (uint32_t index = 0; index < header->region_count; ++index) {
        result.region_index = index;
        const HbgLaunchRegion &region = regions[index];
        uint64_t source_address = 0;
        uint64_t destination_address = 0;
        if (!hbg_checked_add_u64(header->inline_payload_addr, region.source_offset, &source_address) ||
            !hbg_checked_add_u64(
                hbg_destination_base(header->binding, region.kind), region.destination_offset, &destination_address
            )) {
            result.status = HbgRestoreStatus::AddressOverflow;
            return result;
        }

        void *destination = reinterpret_cast<void *>(static_cast<uintptr_t>(destination_address));
        const void *source = reinterpret_cast<const void *>(static_cast<uintptr_t>(source_address));
        const size_t region_size = static_cast<size_t>(region.size);
        const int copy_rc = ops.copy(ops.context, region.kind, destination, source, region_size);
        if (copy_rc != 0) {
            result.status = HbgRestoreStatus::CopyFailed;
            result.callback_error = copy_rc;
            return result;
        }
        const int publish_rc = ops.publish(ops.context, region.kind, destination, region_size);
        if (publish_rc != 0) {
            result.status = HbgRestoreStatus::PublishFailed;
            result.callback_error = publish_rc;
            return result;
        }
    }

    const HbgRestoreCommit candidate{
        slot_registration.binding.slot_generation, header->plan_generation, header->plan_hash, header->identity
    };
    *out_commit = candidate;
    result.status = HbgRestoreStatus::Ok;
    result.slot_status = HbgExecutionSlotStatus::Ok;
    result.blob_status = HbgLaunchBlobStatus::Ok;
    result.region_index = std::numeric_limits<uint32_t>::max();
    result.callback_error = 0;
    return result;
}

}  // namespace simpler::hbg

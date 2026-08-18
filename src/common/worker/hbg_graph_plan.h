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
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "hbg_launch_blob_builder.h"

namespace simpler::hbg {

class HbgGraphPlan;

HbgLaunchBlobStatus build_hbg_graph_plan(
    const HbgExecutionBinding &binding, const HbgInvocationIdentity &identity, uint64_t plan_generation,
    const std::vector<HbgHostRegionInput> &inputs, std::unique_ptr<const HbgGraphPlan> *out
) noexcept;

/**
 * Immutable, host-owned canonical image produced by one HBG build.
 *
 * This object is deliberately distinct from the writable HostArgs vector
 * passed to CANN. It owns a deep copy of every pristine region, exposes no
 * mutable byte view, and can serialize any number of fresh launch snapshots.
 * CANN may patch a serialized snapshot's placeholder field without modifying
 * this plan or another eager/captured task's bytes.
 */
class HbgGraphPlan final {
public:
    HbgGraphPlan(const HbgGraphPlan &) = delete;
    HbgGraphPlan &operator=(const HbgGraphPlan &) = delete;
    HbgGraphPlan(HbgGraphPlan &&) = delete;
    HbgGraphPlan &operator=(HbgGraphPlan &&) = delete;

    const HbgExecutionBinding &binding() const noexcept { return header().binding; }
    const HbgInvocationIdentity &identity() const noexcept { return header().identity; }
    uint64_t plan_generation() const noexcept { return header().plan_generation; }
    uint64_t plan_hash() const noexcept { return header().plan_hash; }
    size_t serialized_size() const noexcept { return canonical_blob_.size(); }
    size_t region_count() const noexcept { return header().region_count; }

    /** Deep-serialize one independent writable HostArgs snapshot. */
    HbgLaunchBlobStatus serialize(std::vector<uint8_t> *out) const noexcept {
        if (out == nullptr) return HbgLaunchBlobStatus::NullArgument;

        std::vector<uint8_t> candidate;
        try {
            candidate = canonical_blob_;
        } catch (...) {
            return HbgLaunchBlobStatus::AllocationFailure;
        }
        const HbgLaunchBlobStatus status = validate_hbg_launch_blob(
            candidate.data(), candidate.size(), HbgLaunchBlobAddressMode::HostUnpatched, &header().binding,
            &header().identity
        );
        if (status != HbgLaunchBlobStatus::Ok) return status;
        *out = std::move(candidate);
        return HbgLaunchBlobStatus::Ok;
    }

private:
    friend HbgLaunchBlobStatus build_hbg_graph_plan(
        const HbgExecutionBinding &, const HbgInvocationIdentity &, uint64_t, const std::vector<HbgHostRegionInput> &,
        std::unique_ptr<const HbgGraphPlan> *
    ) noexcept;

    HbgGraphPlan() = default;
    const HbgLaunchBlobHeader &header() const noexcept {
        return *reinterpret_cast<const HbgLaunchBlobHeader *>(canonical_blob_.data());
    }

    // Canonical, validated HostUnpatched representation. It is private so no
    // placeholder patch or caller mutation can reach the plan itself.
    std::vector<uint8_t> canonical_blob_;
};

/**
 * Build and validate an immutable canonical plan transactionally.
 *
 * The prior output owner is preserved on every failure. Validation is run on
 * a serialized candidate after all source bytes have been deep-copied, so a
 * successful plan is guaranteed to be able to create a canonical launch blob
 * without retaining any caller buffer.
 */
inline HbgLaunchBlobStatus build_hbg_graph_plan(
    const HbgExecutionBinding &binding, const HbgInvocationIdentity &identity, uint64_t plan_generation,
    const std::vector<HbgHostRegionInput> &inputs, std::unique_ptr<const HbgGraphPlan> *out
) noexcept {
    if (out == nullptr) return HbgLaunchBlobStatus::NullArgument;

    std::vector<uint8_t> canonical_blob;
    const HbgLaunchBlobStatus status =
        build_hbg_launch_blob(binding, identity, plan_generation, inputs, &canonical_blob);
    if (status != HbgLaunchBlobStatus::Ok) return status;

    std::unique_ptr<HbgGraphPlan> candidate(new (std::nothrow) HbgGraphPlan());
    if (candidate == nullptr) return HbgLaunchBlobStatus::AllocationFailure;
    candidate->canonical_blob_ = std::move(canonical_blob);

    *out = std::unique_ptr<const HbgGraphPlan>(candidate.release());
    return HbgLaunchBlobStatus::Ok;
}

}  // namespace simpler::hbg

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

#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "hbg_argument_snapshot.h"
#include "hbg_graph_plan.h"

namespace simpler::hbg {

/**
 * One context-local HBG plan entry for one callable.
 *
 * The cache is intentionally single-entry: stable eager/capture addresses hit,
 * while changing addresses or scalar values replace the prior plan without an
 * unbounded per-invocation host-memory owner. Each launch still serializes a
 * writable snapshot for CANN, so captured nodes never share mutable HostArgs.
 */
class HbgGraphPlanCache final {
public:
    HbgGraphPlanCache() = default;
    HbgGraphPlanCache(const HbgGraphPlanCache &) = delete;
    HbgGraphPlanCache &operator=(const HbgGraphPlanCache &) = delete;
    HbgGraphPlanCache(HbgGraphPlanCache &&) = delete;
    HbgGraphPlanCache &operator=(HbgGraphPlanCache &&) = delete;

    const HbgGraphPlan *lookup(const ChipStorageTaskArgs &args, uint64_t argument_snapshot_hash) const noexcept {
        if (argument_snapshot_hash == 0 || argument_snapshot_hash_ != argument_snapshot_hash ||
            argument_snapshot_ == nullptr || plan_ == nullptr ||
            !hbg_argument_snapshots_equal(*argument_snapshot_, args)) {
            return nullptr;
        }
        return plan_.get();
    }

    HbgLaunchBlobStatus replace(
        const ChipStorageTaskArgs &args, uint64_t argument_snapshot_hash, std::unique_ptr<const HbgGraphPlan> plan
    ) noexcept {
        if (plan == nullptr) return HbgLaunchBlobStatus::NullArgument;
        if (argument_snapshot_hash == 0 || hbg_argument_snapshot_hash(args) != argument_snapshot_hash) {
            return HbgLaunchBlobStatus::InvalidIdentity;
        }
        const HbgInvocationIdentity &identity = plan->identity();
        if (identity.argument_snapshot_hash != argument_snapshot_hash ||
            identity.tensor_count != static_cast<uint32_t>(args.tensor_count()) ||
            identity.scalar_count != static_cast<uint32_t>(args.scalar_count())) {
            return HbgLaunchBlobStatus::IdentityMismatch;
        }

        std::vector<uint8_t> direct_launch_package;
        if (plan->has_direct_launch_package()) {
            const HbgLaunchBlobStatus status = plan->serialize_direct_launch_package(&direct_launch_package);
            if (status != HbgLaunchBlobStatus::Ok) return status;
        }
        std::unique_ptr<ChipStorageTaskArgs> argument_snapshot(new (std::nothrow) ChipStorageTaskArgs(args));
        if (argument_snapshot == nullptr) return HbgLaunchBlobStatus::AllocationFailure;

        argument_snapshot_ = std::move(argument_snapshot);
        plan_ = std::move(plan);
        direct_launch_package_ = std::move(direct_launch_package);
        argument_snapshot_hash_ = argument_snapshot_hash;
        return HbgLaunchBlobStatus::Ok;
    }

    const std::vector<uint8_t> &direct_launch_package() const noexcept { return direct_launch_package_; }

private:
    uint64_t argument_snapshot_hash_{0};
    std::unique_ptr<ChipStorageTaskArgs> argument_snapshot_;
    std::unique_ptr<const HbgGraphPlan> plan_;
    std::vector<uint8_t> direct_launch_package_;
};

}  // namespace simpler::hbg

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

#ifndef SIMPLER_TASK_INTERFACE_ORCHESTRATION_REQUIREMENTS_H_
#define SIMPLER_TASK_INTERFACE_ORCHESTRATION_REQUIREMENTS_H_

#include <cstdint>

namespace simpler {
namespace orchestration {

/**
 * Optional symbol emitted by new PyPTO orchestration shared objects.
 *
 * Keeping this as a separate scalar-returning symbol preserves the historical
 * PTO2OrchestrationConfig return ABI.  L2 may load an older SO without it;
 * borrowed-resource HBG L1 must reject such an SO because absence cannot prove
 * that host graph construction is independent of tensor contents.
 */
inline constexpr const char *REQUIREMENTS_V1_SYMBOL = "pypto_orchestration_requirements_v1";

inline constexpr uint64_t REQUIREMENT_TENSOR_DATA_READ = UINT64_C(1) << 0;
inline constexpr uint64_t REQUIREMENT_TENSOR_DATA_WRITE = UINT64_C(1) << 1;
inline constexpr uint64_t REQUIREMENTS_V1_KNOWN_MASK = REQUIREMENT_TENSOR_DATA_READ | REQUIREMENT_TENSOR_DATA_WRITE;
inline constexpr uint64_t REQUIREMENTS_V1_HOST_TENSOR_DATA_MASK = REQUIREMENTS_V1_KNOWN_MASK;

using RequirementsV1Function = uint64_t (*)(void);

enum class HbgL1RequirementsStatus : uint8_t {
    Ok = 0,
    MetadataUnavailable,
    UnknownRequirement,
    HostTensorDataRequired,
};

/**
 * Validate one orchestration SO before HBG L1 executes host graph build.
 *
 * HBG L1 borrows external device tensors and cannot synchronize the caller
 * stream merely so host orchestration can inspect their contents.  Every known
 * tensor-data requirement is therefore unsupported in v1.  Unknown metadata
 * and unknown future bits also fail closed; legacy HBG L2 remains free to use
 * its staged host views and does not call this validator.
 */
inline constexpr HbgL1RequirementsStatus validate_hbg_l1_requirements(bool metadata_available, uint64_t requirements) {
    if (!metadata_available) return HbgL1RequirementsStatus::MetadataUnavailable;
    if ((requirements & ~REQUIREMENTS_V1_KNOWN_MASK) != 0) {
        return HbgL1RequirementsStatus::UnknownRequirement;
    }
    if ((requirements & REQUIREMENTS_V1_HOST_TENSOR_DATA_MASK) != 0) {
        return HbgL1RequirementsStatus::HostTensorDataRequired;
    }
    return HbgL1RequirementsStatus::Ok;
}

}  // namespace orchestration
}  // namespace simpler

#endif  // SIMPLER_TASK_INTERFACE_ORCHESTRATION_REQUIREMENTS_H_

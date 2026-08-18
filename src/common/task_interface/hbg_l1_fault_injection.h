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

namespace simpler::hbg {

/**
 * Internal, task-local HBG L1 fault injection stages.
 *
 * A test request is carried by the runtime-owned HostArgs snapshot.  It is not
 * resident process state, so one injected launch cannot leak into the next
 * invocation or an ACLGraph node built from another package.  Production
 * launch blobs leave both the flag and marker zero.
 */
enum class HbgL1FaultStage : uint32_t {
    None = 0,
    RestoreCopy = 1,
    RestorePublish = 2,
    AfterSchedulerInit = 3,
    BeforeClassify = 4,
    BeforeDispatch = 5,
    Shutdown = 6,
    RuntimeDestroy = 7,
    SchedulerInit = 8,
    SchedulerAssign = 9,
    SchedulerDispatch = 10,
    PlatformBridge = 11,
    AffinityInputs = 12,
    KernelArgsRuntime = 13,
};

inline constexpr uint32_t HBG_L1_FAULT_MARKER_MAGIC = 0x544C3146U;  // "F1LT" in little-endian memory.
inline constexpr int32_t HBG_L1_FAULT_ERROR_BASE = -1700;

inline constexpr bool hbg_l1_valid_fault_stage(HbgL1FaultStage stage) noexcept {
    return stage >= HbgL1FaultStage::RestoreCopy && stage <= HbgL1FaultStage::KernelArgsRuntime;
}

inline constexpr uint64_t hbg_l1_encode_fault_marker(HbgL1FaultStage stage) noexcept {
    return hbg_l1_valid_fault_stage(stage) ?
               (static_cast<uint64_t>(HBG_L1_FAULT_MARKER_MAGIC) << 32U) | static_cast<uint32_t>(stage) :
               0;
}

inline constexpr HbgL1FaultStage hbg_l1_decode_fault_marker(uint64_t marker) noexcept {
    if (static_cast<uint32_t>(marker >> 32U) != HBG_L1_FAULT_MARKER_MAGIC) return HbgL1FaultStage::None;
    const auto stage = static_cast<HbgL1FaultStage>(static_cast<uint32_t>(marker));
    return hbg_l1_valid_fault_stage(stage) ? stage : HbgL1FaultStage::None;
}

inline constexpr int32_t hbg_l1_fault_error(HbgL1FaultStage stage) noexcept {
    return hbg_l1_valid_fault_stage(stage) ? HBG_L1_FAULT_ERROR_BASE - static_cast<int32_t>(stage) : 0;
}

}  // namespace simpler::hbg

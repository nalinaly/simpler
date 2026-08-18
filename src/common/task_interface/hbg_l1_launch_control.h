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
#include <type_traits>

namespace simpler::hbg {

inline constexpr uint32_t HBG_L1_PRELAUNCH_CONTINUE = 0;
inline constexpr uint32_t HBG_L1_PRELAUNCH_CANCEL = 1;

/**
 * Context-owned control line used only before the HBG scheduler generation
 * exists.
 *
 * Per-core Handshake::aicpu_ready cannot safely carry this signal: an AICPU
 * validation failure may happen before the corresponding AICore has published
 * its report, and that later whole-cache-line report could overwrite an early
 * per-core CANCEL.  This independent cache line has one writer value, is
 * cleared on the caller stream before every launch, and remains CANCEL until
 * the complete hidden-stream AICore launch exits.
 */
struct alignas(64) HbgL1LaunchControl {
    volatile uint32_t prelaunch_state{HBG_L1_PRELAUNCH_CONTINUE};
    uint8_t reserved[60]{};
};

static_assert(sizeof(HbgL1LaunchControl) == 64, "HBG L1 launch control must occupy one cache line");
static_assert(alignof(HbgL1LaunchControl) == 64, "HBG L1 launch control must be cache-line aligned");
static_assert(std::is_standard_layout_v<HbgL1LaunchControl>, "HBG L1 launch control must be standard-layout");
static_assert(std::is_trivially_copyable_v<HbgL1LaunchControl>, "HBG L1 launch control must be byte-copyable");

inline bool hbg_l1_prelaunch_cancelled(const HbgL1LaunchControl *control) noexcept {
    return control != nullptr && control->prelaunch_state == HBG_L1_PRELAUNCH_CANCEL;
}

}  // namespace simpler::hbg

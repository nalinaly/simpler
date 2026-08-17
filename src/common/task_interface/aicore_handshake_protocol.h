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

// Pre-register-window AICPU -> AICore control. The successful path continues
// to use DATA_MAIN_BASE != 0 as its only release signal; value 1 is reserved so
// this error-only extension does not reintroduce an AICPU-ready round trip.
inline constexpr uint32_t AICORE_PRE_WINDOW_WAIT = 0;
inline constexpr uint32_t AICORE_PRE_WINDOW_LEGACY_PROCEED = 1;
inline constexpr uint32_t AICORE_PRE_WINDOW_CANCEL = 2;

// The normal register window opens in a few microseconds. Check the GM cancel
// line only occasionally while that MMIO register is still zero. CANCEL stays
// published until the entire AICore launch completes, so a sparse poll cannot
// miss it.
inline constexpr uint32_t AICORE_PRE_WINDOW_CANCEL_POLL_INTERVAL = 256;
inline constexpr uint32_t AICORE_PRE_WINDOW_CANCEL_POLL_MASK = AICORE_PRE_WINDOW_CANCEL_POLL_INTERVAL - 1;

inline constexpr bool aicore_register_index_valid(uint32_t physical_core_id, uint32_t register_address_count) noexcept {
    return physical_core_id < register_address_count;
}

inline constexpr bool aicore_register_mapping_invalid(
    uint32_t physical_core_id, uint32_t max_physical_core_count, uint64_t register_address
) noexcept {
    return !aicore_register_index_valid(physical_core_id, max_physical_core_count) || register_address == 0;
}

static_assert(
    (AICORE_PRE_WINDOW_CANCEL_POLL_INTERVAL & AICORE_PRE_WINDOW_CANCEL_POLL_MASK) == 0,
    "pre-window cancel poll interval must be a power of two"
);

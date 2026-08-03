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

// Private BlockWon historically exposes four physical slots.  Experiments may
// reduce the active prefix to one or two slots, while BlockWon keeps storage
// for all four so an experimental AICore image remains layout-compatible with
// the baseline AICPU-owned DistGlobal allocation.
#ifndef PTO_FDWIC_PRIVATE_WON_SLOT_COUNT
#define PTO_FDWIC_PRIVATE_WON_SLOT_COUNT 4
#endif

constexpr int32_t kWonSlotStorageCapacity = 4;
constexpr int32_t kWonSlotCount = PTO_FDWIC_PRIVATE_WON_SLOT_COUNT;

static_assert(
    kWonSlotCount == 1 || kWonSlotCount == 2 || kWonSlotCount == kWonSlotStorageCapacity,
    "private WonSlot count must be 1, 2, or 4"
);

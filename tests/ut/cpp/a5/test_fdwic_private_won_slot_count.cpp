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

#include <gtest/gtest.h>

#include <cstddef>

#include "dist_engine/common/state.h"

#ifndef PTO_FDWIC_TEST_EXPECTED_WON_SLOT_COUNT
#error "PTO_FDWIC_TEST_EXPECTED_WON_SLOT_COUNT must select the test variant"
#endif

TEST(FdwicPrivateWonSlotCount, PreservesBlockWonAndRingSlotAbi) {
    EXPECT_EQ(kWonSlotCount, PTO_FDWIC_TEST_EXPECTED_WON_SLOT_COUNT);
    EXPECT_EQ(kPrivateSlots, 4);
    EXPECT_EQ(kWonReserve, 2);
    EXPECT_EQ(sizeof(BlockWon), sizeof(WonSlot) * kWonSlotStorageCapacity + 2 * 64);
    EXPECT_EQ(
        offsetof(BlockWon, any_pub),
        sizeof(WonSlot) * kWonSlotStorageCapacity + 64
    );
}

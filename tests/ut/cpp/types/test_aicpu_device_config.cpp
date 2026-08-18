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

#include "aicpu/aicpu_device_config.h"

TEST(AicpuDeviceConfig, RetainsHbgPrelaunchControlAcrossInvocationTasks) {
    set_hbg_l1_prelaunch_control_addr(0);
    EXPECT_EQ(get_hbg_l1_prelaunch_control_addr(), 0u);

    constexpr unsigned long long control_address = 0x12345678000ULL;
    set_hbg_l1_prelaunch_control_addr(control_address);
    EXPECT_EQ(get_hbg_l1_prelaunch_control_addr(), control_address);

    set_hbg_l1_prelaunch_control_addr(0);
}

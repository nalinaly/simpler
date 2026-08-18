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

#include "aicpu/platform_aicpu_affinity.h"

TEST(PlatformAicpuAffinityConfig, AcceptsOnlyShapesThatCanPopulateEveryRole) {
    EXPECT_TRUE(platform_aicpu_affinity_config_valid(1, 1));
    EXPECT_TRUE(platform_aicpu_affinity_config_valid(MAX_GATE_THREADS, MAX_GATE_THREADS));
    EXPECT_TRUE(platform_aicpu_affinity_config_valid(4, 8));

    EXPECT_FALSE(platform_aicpu_affinity_config_valid(0, 1));
    EXPECT_FALSE(platform_aicpu_affinity_config_valid(1, 0));
    EXPECT_FALSE(platform_aicpu_affinity_config_valid(-1, 1));
    EXPECT_FALSE(platform_aicpu_affinity_config_valid(1, -1));
    EXPECT_FALSE(platform_aicpu_affinity_config_valid(MAX_GATE_THREADS + 1, MAX_GATE_THREADS + 1));
    EXPECT_FALSE(platform_aicpu_affinity_config_valid(MAX_GATE_THREADS, MAX_GATE_THREADS + 1));
    EXPECT_FALSE(platform_aicpu_affinity_config_valid(5, 4));
}

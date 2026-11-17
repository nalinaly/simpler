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
#include "completion_counter_plan.h"
#include <gtest/gtest.h>

TEST(HbgCompletionCounterPlan, ReusesAcrossControlNodesAndPreservesOldThresholds) {
    SchedulerCompletionCounterPlan plan;
    ASSERT_TRUE(scheduler_plan_completion_counters({{}, {0}, {1}}, {0, 1, 2}, {192, 0, 128}, &plan));
    ASSERT_EQ(plan.chains.size(), 1u);
    EXPECT_EQ(plan.thresholds[0], 192u);
    EXPECT_EQ(plan.thresholds[2], 320u);
    EXPECT_EQ(plan.chains[0].limit, 320u);
    EXPECT_EQ(plan.task_chains[1], UINT32_MAX);
}

TEST(HbgCompletionCounterPlan, IndependentBranchesDoNotShareCounter) {
    SchedulerCompletionCounterPlan plan;
    ASSERT_TRUE(scheduler_plan_completion_counters({{}, {0}, {0}, {1, 2}}, {0, 1, 2, 3}, {2, 4, 6, 8}, &plan));
    EXPECT_EQ(plan.chains.size(), 2u);
    EXPECT_NE(plan.task_chains[1], plan.task_chains[2]);
    EXPECT_EQ(plan.task_chains[3], plan.task_chains[2]);
}

TEST(HbgCompletionCounterPlan, SplitsBeforeRegisterOverflow) {
    SchedulerCompletionCounterPlan plan;
    ASSERT_TRUE(scheduler_plan_completion_counters({{}, {0}}, {0, 1}, {UINT32_MAX - 1u, 2}, &plan));
    EXPECT_EQ(plan.chains.size(), 2u);
    EXPECT_EQ(plan.thresholds[1], 2u);
}

TEST(HbgCompletionCounterPlan, RejectsInvalidDependencyOrder) {
    SchedulerCompletionCounterPlan plan;
    EXPECT_FALSE(scheduler_plan_completion_counters({{}, {0}}, {1, 0}, {2, 2}, &plan));
    EXPECT_FALSE(scheduler_plan_completion_counters({{1}, {0}}, {0, 1}, {2, 2}, &plan));
    EXPECT_FALSE(scheduler_plan_completion_counters({{}, {}}, {0, 0}, {2, 2}, &plan));
}

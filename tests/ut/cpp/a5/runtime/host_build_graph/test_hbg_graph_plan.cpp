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

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "scheduler_graph_plan.h"

namespace {

using Kind = SchedulerGraphPlanNodeKind;
using Shape = SchedulerGraphPlanShape;

TEST(HbgGraphPlan, OrdinaryReferencesAndPublicIdentitiesRemainUnchanged) {
    auto plan = SchedulerGraphPlan::build(std::vector<Shape>(7));
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->node_count(), 7);
    EXPECT_EQ(plan->instance_count(), 0);
    for (int32_t ref = 0; ref < 7; ++ref) {
        auto node = plan->node_at(ref);
        ASSERT_TRUE(node.has_value());
        EXPECT_EQ(node->kind, Kind::ORDINARY);
        EXPECT_EQ(node->task_id, TaskId::make_global(ref));
        EXPECT_EQ(node->outer_id, ref);
        EXPECT_EQ(node->child_index, -1);
        EXPECT_EQ(plan->completion_ref(ref), ref);
        EXPECT_FALSE(plan->root_gate(ref).has_value());
    }
}

TEST(HbgGraphPlan, GraphExpansionPreservesParentQualifiedChildIdentities) {
    auto plan = SchedulerGraphPlan::build({{}, {true, 2}, {}, {true, 2}, {}});
    ASSERT_TRUE(plan.has_value());
    ASSERT_EQ(plan->node_count(), 11);
    ASSERT_EQ(plan->instance_count(), 2);
    const std::vector<Kind> expected{
        Kind::ORDINARY, Kind::ENTER, Kind::CHILD, Kind::CHILD, Kind::EXIT,     Kind::ORDINARY,
        Kind::ENTER,    Kind::CHILD, Kind::CHILD, Kind::EXIT,  Kind::ORDINARY,
    };
    for (int32_t ref = 0; ref < plan->node_count(); ++ref) {
        SCOPED_TRACE(ref);
        ASSERT_TRUE(plan->node_at(ref));
        EXPECT_EQ(plan->node_at(ref)->kind, expected[ref]);
    }
    EXPECT_EQ(plan->node_at(2)->task_id, TaskId::make_sub_task(1, 0));
    EXPECT_EQ(plan->node_at(7)->task_id, TaskId::make_sub_task(3, 0));
    EXPECT_NE(plan->node_at(2)->task_id, plan->node_at(7)->task_id);
    EXPECT_EQ(plan->node_at(1)->task_id, TaskId::make_global(1));
    EXPECT_EQ(plan->node_at(4)->task_id, TaskId::make_global(1));
    EXPECT_EQ(plan->completion_ref(1), 4);
    EXPECT_EQ(plan->completion_ref(3), 9);
    EXPECT_EQ(plan->root_gate(2), 1);
    EXPECT_EQ(plan->root_gate(7), 6);
}

TEST(HbgGraphPlan, ExternalConsumersWaitForExitRatherThanEnterOrOneChild) {
    auto plan = SchedulerGraphPlan::build({{}, {true, 3}, {}, {true, 2}, {}});
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->dependency_ref(1, 0), 0);
    EXPECT_EQ(plan->dependency_ref(6, 1), 5);
    EXPECT_EQ(plan->dependency_ref(7, 1), 5);
    EXPECT_EQ(plan->dependency_ref(11, 3), 10);
    EXPECT_FALSE(plan->dependency_ref(1, 1));
    EXPECT_FALSE(plan->dependency_ref(6, 2));
    EXPECT_FALSE(plan->dependency_ref(5, 1));
    EXPECT_FALSE(plan->root_gate(5));
}

TEST(HbgGraphPlan, InternalCsrIndicesTranslateOnlyWithinTheirOwnInstance) {
    auto plan = SchedulerGraphPlan::build({{true, 3}, {true, 3}});
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->dependency_ref(3, 0), 1);
    EXPECT_EQ(plan->dependency_ref(3, 1), 2);
    EXPECT_EQ(plan->dependency_ref(8, 0), 6);
    EXPECT_EQ(plan->dependency_ref(8, 1), 7);
    EXPECT_FALSE(plan->dependency_ref(3, 2));
    EXPECT_FALSE(plan->dependency_ref(3, 3));
    EXPECT_FALSE(plan->dependency_ref(6, 0));
    EXPECT_FALSE(plan->dependency_ref(8, -1));
}

TEST(HbgGraphPlan, LargeBodyDoesNotCreateAnOversizedExitFanin) {
    auto plan = SchedulerGraphPlan::build({{true, SUB_TASK_MAX_NUM}});
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->node_count(), SUB_TASK_MAX_NUM + 2);
    const int32_t exit_ref = plan->completion_ref(0).value();
    EXPECT_EQ(plan->node_at(exit_ref)->kind, Kind::EXIT);
    for (int32_t child = 0; child < SUB_TASK_MAX_NUM; ++child) {
        EXPECT_EQ(plan->node_at(child + 1)->task_id, TaskId::make_sub_task(0, child));
        EXPECT_EQ(plan->root_gate(child + 1), 0);
        EXPECT_FALSE(plan->dependency_ref(exit_ref, child));
    }
}

TEST(HbgGraphPlan, RejectsMalformedShapesBeforeAllocatingExpandedStorage) {
    EXPECT_FALSE(SchedulerGraphPlan::build({{true, 0}}));
    EXPECT_FALSE(SchedulerGraphPlan::build({{true, -1}}));
    EXPECT_FALSE(SchedulerGraphPlan::build({{true, SUB_TASK_MAX_NUM + 1}}));
    EXPECT_FALSE(SchedulerGraphPlan::build({{false, 1}}));
    EXPECT_FALSE(SchedulerGraphPlan::build({{true, std::numeric_limits<int32_t>::max()}}));
}

TEST(HbgGraphPlan, EmptyRunHasNoSyntheticWork) {
    auto plan = SchedulerGraphPlan::build({});
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->node_count(), 0);
    EXPECT_EQ(plan->instance_count(), 0);
    EXPECT_FALSE(plan->node_at(0));
    EXPECT_FALSE(plan->completion_ref(0));
}

TEST(HbgGraphPlan, InvalidReferencesCannotAliasValidNodes) {
    auto plan = SchedulerGraphPlan::build({{}, {true, 2}});
    ASSERT_TRUE(plan);
    for (int64_t ref : std::array<int64_t, 3>{-1, 5, INT64_MAX}) {
        EXPECT_FALSE(plan->node_at(ref));
        EXPECT_FALSE(plan->root_gate(ref));
        EXPECT_FALSE(plan->dependency_ref(ref, 0));
    }
    EXPECT_FALSE(plan->completion_ref(-1));
    EXPECT_FALSE(plan->completion_ref(2));
    EXPECT_FALSE(plan->completion_ref(INT64_MAX));
}

TEST(HbgGraphPlan, AllProjectedDependenciesPrecedeConsumers) {
    auto plan = SchedulerGraphPlan::build({{}, {true, 7}, {}, {true, 3}, {true, 1}, {}});
    ASSERT_TRUE(plan);
    for (int32_t consumer = 0; consumer < plan->node_count(); ++consumer) {
        SCOPED_TRACE(consumer);
        auto node = plan->node_at(consumer).value();
        if (node.kind == Kind::CHILD) {
            ASSERT_LT(plan->root_gate(consumer).value(), consumer);
            for (int32_t pred = 0; pred < node.child_index; ++pred) {
                auto mapped = plan->dependency_ref(consumer, pred);
                ASSERT_TRUE(mapped);
                EXPECT_LT(mapped.value(), consumer);
                EXPECT_EQ(plan->node_at(mapped.value())->outer_id, node.outer_id);
            }
        } else if (node.kind != Kind::EXIT) {
            for (int32_t pred = 0; pred < node.outer_id; ++pred) {
                auto mapped = plan->dependency_ref(consumer, pred);
                ASSERT_TRUE(mapped);
                EXPECT_LT(mapped.value(), consumer);
                EXPECT_EQ(mapped, plan->completion_ref(pred));
            }
        }
    }
}

}  // namespace

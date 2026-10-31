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
#include <cstddef>
#include <cstdint>

#include "scheduler/scheduler_graph_definition.h"

namespace {

struct Body {
    GraphDefinition definition{};
    std::array<SubTaskDefinition, 2> children{};
    std::array<uint64_t, 2> scalars{13, 29};
    std::array<GraphScalarInheritance, 2> sources{
        GraphScalarInheritance::self_value(), GraphScalarInheritance::from_boundary(3)
    };
};

struct DefinitionObject {
    GraphDefinitionHeader header{};
    Body body{};

    DefinitionObject() {
        header.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
        header.full_key = 17;
        header.definition_bytes = sizeof(Body);
        auto &def = body.definition;
        def.full_key = header.full_key;
        def.total_bytes = sizeof(Body);
        def.task_count = body.children.size();
        def.scalar_arg_count = body.scalars.size();
        def.off_sub_tasks = offsetof(Body, children);
        def.off_scalars = offsetof(Body, scalars);
        def.off_scalar_inheritance = offsetof(Body, sources);
        body.children[1].kernel_id[1] = 9;
    }

    bool
    open(SchedulerGraphDefinitionView *view, uint64_t expected_key = 17, uint64_t bytes = sizeof(DefinitionObject)) {
        return scheduler_graph_definition_open(
            reinterpret_cast<uint64_t>(this), bytes, offsetof(DefinitionObject, body), expected_key, view
        );
    }
};

TEST(HbgGraphDefinitionView, UsesTheRecordedWireImageWithoutAnotherDefinitionCopy) {
    DefinitionObject object;
    SchedulerGraphDefinitionView view{};
    ASSERT_TRUE(object.open(&view));
    const auto &def = object.body.definition;
    EXPECT_EQ(
        scheduler_graph_definition_array<SubTaskDefinition>(view, def.off_sub_tasks, def.task_count),
        object.body.children.data()
    );
    SubTaskDefinition child{};
    ASSERT_TRUE(scheduler_graph_definition_read(view, def.off_sub_tasks, def.task_count, 1, &child));
    EXPECT_EQ(child.kernel_id[1], 9);
    GraphScalarInheritance source{};
    ASSERT_TRUE(scheduler_graph_definition_read(view, def.off_scalar_inheritance, def.scalar_arg_count, 1, &source));
    EXPECT_TRUE(source.inherited());
    EXPECT_EQ(source.boundary_index(), 3u);
}

TEST(HbgGraphDefinitionView, RejectsMismatchedFramingBeforeAnySectionAccess) {
    DefinitionObject object;
    SchedulerGraphDefinitionView view{};
    EXPECT_FALSE(object.open(&view, 18));
    EXPECT_FALSE(object.open(&view, 17, sizeof(object) - 1));
    object.header.magic = 0;
    EXPECT_FALSE(object.open(&view));
    object.header.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
    object.header.definition_bytes--;
    EXPECT_FALSE(object.open(&view));
    object.header.definition_bytes++;
    object.body.definition.full_key++;
    EXPECT_FALSE(object.open(&view));
}

TEST(HbgGraphDefinitionView, RejectsBadAlignmentAbsentOffsetsAndAddressOverflow) {
    DefinitionObject object;
    SchedulerGraphDefinitionView view{};
    const auto address = reinterpret_cast<uint64_t>(&object);
    EXPECT_FALSE(
        scheduler_graph_definition_open(address + 1, sizeof(object), sizeof(GraphDefinitionHeader), 17, &view)
    );
    EXPECT_FALSE(scheduler_graph_definition_open(address, sizeof(object), 0, 17, &view));
    EXPECT_FALSE(scheduler_graph_definition_open(UINT64_MAX - 15, 32, sizeof(GraphDefinitionHeader), 17, &view));
    EXPECT_FALSE(scheduler_graph_definition_open(0, sizeof(object), sizeof(GraphDefinitionHeader), 17, &view));
    EXPECT_FALSE(object.open(nullptr));
}

TEST(HbgGraphDefinitionView, BoundsEveryArrayInsideItsOwnDefinition) {
    std::array<DefinitionObject, 2> objects;
    SchedulerGraphDefinitionView view{};
    ASSERT_TRUE(objects[0].open(&view, 17, sizeof(objects)));
    const uint32_t offset = objects[0].body.definition.off_scalars;
    EXPECT_EQ(scheduler_graph_definition_array<uint64_t>(view, offset, -1), nullptr);
    EXPECT_EQ(scheduler_graph_definition_array<uint64_t>(view, offset + 1, 1), nullptr);
    EXPECT_EQ(scheduler_graph_definition_array<uint64_t>(view, sizeof(Body), 1), nullptr);
    EXPECT_EQ(scheduler_graph_definition_array<uint64_t>(view, UINT32_MAX, INT32_MAX), nullptr);
    EXPECT_EQ(scheduler_graph_definition_array<uint64_t>(view, 0, 1), nullptr);
    uint64_t value = 99;
    EXPECT_FALSE(scheduler_graph_definition_read(view, offset, 2, 2, &value));
    EXPECT_FALSE(scheduler_graph_definition_read(view, offset, 2, -1, &value));
    EXPECT_FALSE(scheduler_graph_definition_read(view, offset, 2, 0, static_cast<uint64_t *>(nullptr)));
    EXPECT_EQ(value, 99u);
    ASSERT_TRUE(scheduler_graph_definition_read(view, offset, 2, 1, &value));
    EXPECT_EQ(value, 29u);
}

}  // namespace

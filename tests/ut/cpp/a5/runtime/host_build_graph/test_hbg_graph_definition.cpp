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

#include "host_build_graph/graph_definition.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

TEST(HbgGraphDefinition, SharedWireTypesKeepTheRecordingLayout) {
    EXPECT_EQ(sizeof(GraphDefinitionHeader), 32u);
    EXPECT_EQ(GRAPH_DEFINITION_OBJECT_ALIGN, 16u);
    EXPECT_EQ(sizeof(GraphDefinition), 104u);
    EXPECT_EQ(sizeof(SubTaskDefinition), 80u);
    EXPECT_EQ(sizeof(GraphScalarInheritance), 4u);
    EXPECT_EQ(offsetof(SubTaskDefinition, tensor_count), 24u);
    EXPECT_EQ(offsetof(SubTaskDefinition, dump_metadata), 48u);
    EXPECT_EQ(offsetof(GraphDefinition, off_sub_tasks), 80u);
    EXPECT_TRUE(std::is_trivially_copyable_v<GraphDefinition>);
    EXPECT_TRUE(std::is_standard_layout_v<SubTaskDefinition>);
}

TEST(HbgGraphDefinition, ScalarSourcesDistinguishOwnedValuesFromBoundaryZero) {
    const auto own = GraphScalarInheritance::self_value();
    const auto first = GraphScalarInheritance::from_boundary(0);
    const auto last = GraphScalarInheritance::from_boundary(UINT16_MAX);
    EXPECT_FALSE(own.inherited());
    EXPECT_TRUE(first.inherited());
    EXPECT_EQ(first.boundary_index(), 0u);
    EXPECT_TRUE(last.inherited());
    EXPECT_EQ(last.boundary_index(), UINT16_MAX);
}

TEST(HbgGraphDefinition, SectionReadersRejectNegativeCountsMisalignmentAndOverflow) {
    struct Image {
        GraphDefinition definition{};
        uint64_t scalar{29};
    } image;
    image.definition.total_bytes = sizeof(image);
    const uint32_t offset = offsetof(Image, scalar);
    EXPECT_EQ(graph_definition_array<uint64_t>(image.definition, offset, 1), &image.scalar);
    EXPECT_EQ(graph_definition_array<uint64_t>(image.definition, offset, -1), nullptr);
    EXPECT_EQ(graph_definition_array<uint64_t>(image.definition, offset + 1, 1), nullptr);
    EXPECT_EQ(graph_definition_array<uint64_t>(image.definition, offset, 2), nullptr);
    EXPECT_EQ(graph_definition_array<uint64_t>(image.definition, UINT32_MAX, INT32_MAX), nullptr);
    EXPECT_EQ(graph_definition_array<uint64_t>(image.definition, 0, 1), nullptr);
}

}  // namespace

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

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "hbg_graph_test_support.h"
#include "host_build_graph/graph_image_view.h"

namespace {

TEST(HbgGraphMaterialize, RebindsBoundaryHeapAndScalarSourcesIntoOriginalStorage) {
    MaterializeFixture f;
    ASSERT_EQ(f.materialize(), SchedulerGraphResult::OK);
    EXPECT_EQ(f.output.tasks[0].task.task_id, TaskId::make_sub_task(5, 0));
    EXPECT_EQ(f.output.tasks[1].task.task_id, TaskId::make_sub_task(5, 1));
    EXPECT_EQ(f.output.tasks[0].task.kernel_id[1], 7);
    EXPECT_EQ(f.output.tasks[1].task.packed_buffer_base, f.heap.data() + 64);
    EXPECT_EQ(f.output.tasks[0].payload.tensor_data(), f.output.tensors.data());
    EXPECT_EQ(f.output.tasks[1].payload.scalar_data(), f.output.scalars.data() + 1);
    EXPECT_EQ(f.output.tensors[0].buffer.addr, 0x100000u);
    EXPECT_EQ(f.output.tensors[0].buffer.size, 1024u);
    EXPECT_EQ(f.output.tensors[0].start_offset, 7u);
    EXPECT_EQ(f.output.tensors[0].address_space, AddressSpace::DEVICE);
    EXPECT_EQ(f.output.tensors[1].buffer.addr, reinterpret_cast<uint64_t>(f.heap.data() + 16));
    EXPECT_EQ(f.output.tensors[1].start_offset, 2u);
    EXPECT_EQ(f.output.scalars[0], 97u);
    EXPECT_EQ(f.output.scalars[1], 29u);
    EXPECT_EQ(f.output.tasks[0].payload.fanin_count, 0);
    EXPECT_EQ(f.output.tasks[0].payload.predicate.op, PredicateOp::NONE);
}

TEST(HbgGraphMaterialize, RepeatedMaterializationRefreshesInvocationValues) {
    MaterializeFixture f;
    ASSERT_EQ(f.materialize(), SchedulerGraphResult::OK);
    f.boundary[0].buffer.addr = 0x200000;
    f.boundary[0].start_offset = 11;
    f.scalar[0] = 123;
    ASSERT_EQ(f.materialize(), SchedulerGraphResult::OK);
    EXPECT_EQ(f.output.tensors[0].buffer.addr, 0x200000u);
    EXPECT_EQ(f.output.tensors[0].start_offset, 13u);
    EXPECT_EQ(f.output.scalars[0], 123u);
    EXPECT_EQ(f.output.scalars[1], 29u);
}

TEST(HbgGraphMaterialize, RejectsBoundaryAndPoolMismatches) {
    MaterializeFixture f;
    f.instance.boundary_scalar_count = 0;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    f.instance.boundary_scalar_count = 1;
    f.object.body.sources[0] = GraphScalarInheritance::from_boundary(1);
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    f.object.body.sources[0] = GraphScalarInheritance::self_value();
    f.object.body.tensors[0].owner_task_id = TaskId::make_param(1);
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    f.object.body.tensors[0].owner_task_id = TaskId::make_param(0);
    f.object.body.tasks[1].tensor_offset = 2;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
}

TEST(HbgGraphMaterialize, RejectsTruncationAndHeapOverflow) {
    MaterializeFixture f;
    --f.instance.task_storage_bytes;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    ++f.instance.task_storage_bytes;
    f.object.body.task_offsets[1] = 128;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    f.object.body.task_offsets[1] = 64;
    f.object.header.magic = 0;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
}

TEST(HbgGraphMaterialize, PredicateCannotSilentlyBecomeAnUnconditionalTask) {
    MaterializeFixture f;
    f.object.body.tasks[0].predicate_slot = 1;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::UNSUPPORTED_SHAPE);
}

TEST(HbgGraphMaterialize, MatchesLegacyParameterBindingForTheSameDefinition) {
    MaterializeFixture f;
    const auto &definition = f.object.body.definition;
    const uint64_t bytes = definition.required_heap + definition.execution_storage_bytes;
    std::unique_ptr<void, decltype(&std::free)> heap(std::aligned_alloc(64, (bytes + 63) & ~UINT64_C(63)), &std::free);
    ASSERT_NE(heap, nullptr);
    ChipTaskStorage outer{};
    outer.task.task_id = TaskId::make_global(5);
    outer.task.packed_buffer_base = heap.get();
    outer.task.packed_buffer_end = static_cast<uint8_t *>(heap.get()) + bytes;
    outer.task.graph_definition_offset = offsetof(MaterializeFixture::Object, body);
    outer.slot.task_kind = TaskKind::GRAPH;
    outer.payload.tensor_count = outer.payload.scalar_count = 1;
    outer.payload.bind_regions(f.boundary.data(), f.scalar.data(), nullptr);
    GraphImageView image{reinterpret_cast<const std::byte *>(&f.object), sizeof(f.object)};
    auto *execution = graph_execution_localize(outer.slot, image);
    ASSERT_NE(execution, nullptr);
    ASSERT_EQ(
        graph_execution_materialize_slice(outer.slot, *execution, image, 2), GraphMaterializeResult::MATERIALIZED
    );
    ASSERT_EQ(f.materialize(), SchedulerGraphResult::OK);
    for (int32_t i = 0; i < 2; ++i) {
        const auto &legacy = execution->task_at(i);
        const auto &native = f.output.tasks[i];
        EXPECT_EQ(native.task.task_id, legacy.task.task_id);
        for (uint32_t lane = 0; lane < 3; ++lane)
            EXPECT_EQ(native.task.kernel_id[lane], legacy.task.kernel_id[lane]);
        EXPECT_EQ(native.payload.tensor_count, legacy.payload.tensor_count);
        EXPECT_EQ(native.payload.scalar_count, legacy.payload.scalar_count);
        EXPECT_EQ(native.payload.scalar_data()[0], legacy.payload.scalar_data()[0]);
        const auto &a = native.payload.tensor_data()[0];
        const auto &b = legacy.payload.tensor_data()[0];
        const uint64_t a_base = i == 0 ? 0 : reinterpret_cast<uint64_t>(f.heap.data());
        const uint64_t b_base = i == 0 ? 0 : reinterpret_cast<uint64_t>(heap.get());
        EXPECT_EQ(a.buffer.addr - a_base, b.buffer.addr - b_base);
        EXPECT_EQ(a.buffer.size, b.buffer.size);
        EXPECT_EQ(a.start_offset, b.start_offset);
        EXPECT_EQ(a.owner_task_id, b.owner_task_id);
        EXPECT_EQ(a.address_space, b.address_space);
        EXPECT_EQ(a.extent_elem_cache, b.extent_elem_cache);
        EXPECT_EQ(a.shapes[0], b.shapes[0]);
        EXPECT_EQ(a.strides[0], b.strides[0]);
    }
}

}  // namespace

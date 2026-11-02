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

#include <algorithm>
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
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
}

TEST(HbgGraphMaterialize, PredicateBindsBoundaryViewAndInternalHeapPerInvocation) {
    MaterializeFixture f;
    f.enable_predicates();
    ASSERT_EQ(f.materialize(), SchedulerGraphResult::OK);
    const auto &boundary = f.output.tasks[0].payload.predicate;
    const auto &internal = f.output.tasks[1].payload.predicate;
    EXPECT_EQ(boundary.addr, 0x100000u + (5 + 2 + 1) * 4);
    EXPECT_EQ(boundary.target, 13);
    EXPECT_EQ(boundary.elem_size, 4);
    EXPECT_EQ(boundary.op, PredicateOp::EQ);
    EXPECT_EQ(internal.addr, reinterpret_cast<uint64_t>(f.heap.data()) + 16 + (2 + 1) * 4);
    EXPECT_EQ(internal.target, 14);
    f.boundary[0].buffer.addr = 0x200000;
    f.boundary[0].start_offset = 11;
    ASSERT_EQ(f.materialize(), SchedulerGraphResult::OK);
    EXPECT_EQ(boundary.addr, 0x200000u + (11 + 2 + 1) * 4);
    EXPECT_EQ(internal.addr, reinterpret_cast<uint64_t>(f.heap.data()) + 28);
}

TEST(HbgGraphMaterialize, PredicateRejectsSelfReadInvalidOperatorAndAddressOverflow) {
    MaterializeFixture f;
    f.enable_predicates();
    auto &predicate = f.object.body.predicates[0];
    predicate.operand.owner_task_id = TaskId::make_sub_task(0, 0);
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    predicate.operand.owner_task_id = TaskId::make_param(0);
    predicate.op = 7;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    predicate.op = static_cast<uint8_t>(PredicateOp::EQ);
    predicate.elem_offset = predicate.operand.extent_elem_cache;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    predicate.elem_offset = 1;
    predicate.elem_size = 3;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    predicate.elem_size = 4;
    f.boundary[0].buffer.addr = UINT64_MAX - 8;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    f.boundary[0].buffer.addr = 0x100000;
    f.boundary[0].buffer.size = 16;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    f.boundary[0].buffer.size = 1024;
    f.boundary[0].start_offset = UINT64_MAX / 4;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
}

TEST(HbgGraphMaterialize, MatchesLegacyParameterBindingForTheSameDefinition) {
    MaterializeFixture f;
    f.enable_predicates();
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
        EXPECT_EQ(native.payload.predicate.addr - a_base, legacy.payload.predicate.addr - b_base);
        EXPECT_EQ(native.payload.predicate.target, legacy.payload.predicate.target);
        EXPECT_EQ(native.payload.predicate.elem_size, legacy.payload.predicate.elem_size);
        EXPECT_EQ(native.payload.predicate.op, legacy.payload.predicate.op);
    }
}

SchedulerGraphView work_graph(const MaterializeFixture &fixture) {
    return {0, 0, fixture.nodes.size(), fixture.nodes.size() - 1, reinterpret_cast<uint64_t>(fixture.nodes.data())};
}

using StorageBytes = std::array<uint8_t, sizeof(MaterializeFixture::Storage)>;

StorageBytes storage_bytes(const MaterializeFixture &fixture) {
    StorageBytes result{};
    std::copy_n(reinterpret_cast<const uint8_t *>(&fixture.output), result.size(), result.begin());
    return result;
}

void poison_storage(MaterializeFixture *fixture) {
    std::fill_n(reinterpret_cast<uint8_t *>(&fixture->output), sizeof(fixture->output), 0xa5);
}

TEST(HbgGraphMaterialize, RegionWorkMatchesSerialBytesInEveryConstructionOrder) {
    MaterializeFixture f;
    f.enable_predicates();
    poison_storage(&f);
    ASSERT_EQ(f.materialize(), SchedulerGraphResult::OK);
    const auto expected = storage_bytes(f);
    const std::array<SchedulerGraphWorkRange, 3> ranges{{{0, 1}, {1, 1}, {0, 2}}};
    const std::array<SchedulerGraphWork, 4> work{{
        {0, SchedulerGraphWorkRegion::TASK_STORAGE, 0, 1, SCHEDULER_GRAPH_TASK_STORAGE_STRIDE},
        {0, SchedulerGraphWorkRegion::TASK_STORAGE, 1, 1, SCHEDULER_GRAPH_TASK_STORAGE_STRIDE},
        {0, SchedulerGraphWorkRegion::TENSOR_POOL, 2, 1, 2 * TASKPAYLOAD_TENSOR_STRIDE},
        {0, SchedulerGraphWorkRegion::SCALAR_POOL, 2, 1, 2 * sizeof(uint64_t)},
    }};
    std::array<uint32_t, 4> order{0, 1, 2, 3};
    do {
        poison_storage(&f);
        for (uint32_t index : order) {
            int32_t failed_child = 99;
            ASSERT_EQ(
                scheduler_graph_materialize_work(
                    f.instance, work_graph(f), work[index], ranges.data(), ranges.size(), &failed_child
                ),
                SchedulerGraphResult::OK
            );
            EXPECT_EQ(failed_child, -1);
        }
        EXPECT_EQ(storage_bytes(f), expected);
    } while (std::next_permutation(order.begin(), order.end()));
}

TEST(HbgGraphMaterialize, SharedSlotsDoNotValidateOrWriteUnusedPoolEntries) {
    MaterializeFixture f;
    f.object.body.tasks[1].tensor_offset = 0;
    f.object.body.tasks[1].scalar_offset = 0;
    f.object.body.tensors[1].owner_task_id = TaskId::make_param(99);
    f.object.body.sources[1] = GraphScalarInheritance::from_boundary(99);
    poison_storage(&f);
    ASSERT_EQ(f.materialize(), SchedulerGraphResult::OK);
    const auto expected = storage_bytes(f);
    poison_storage(&f);
    const std::array<SchedulerGraphWorkRange, 2> ranges{{{0, 2}, {0, 1}}};
    const std::array<SchedulerGraphWork, 3> work{{
        {0, SchedulerGraphWorkRegion::SCALAR_POOL, 1, 1, sizeof(uint64_t)},
        {0, SchedulerGraphWorkRegion::TENSOR_POOL, 1, 1, TASKPAYLOAD_TENSOR_STRIDE},
        {0, SchedulerGraphWorkRegion::TASK_STORAGE, 0, 1, 2 * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE},
    }};
    for (const auto &item : work)
        ASSERT_EQ(
            scheduler_graph_materialize_work(f.instance, work_graph(f), item, ranges.data(), ranges.size()),
            SchedulerGraphResult::OK
        );
    EXPECT_EQ(storage_bytes(f), expected);
}

TEST(HbgGraphMaterialize, ParameterWorkReportsTheOriginalChildOnBindingFailure) {
    MaterializeFixture f;
    const SchedulerGraphWorkRange range{1, 1};
    const SchedulerGraphWork scalar{0, SchedulerGraphWorkRegion::SCALAR_POOL, 0, 1, sizeof(uint64_t)};
    f.object.body.sources[1] = GraphScalarInheritance::from_boundary(99);
    int32_t failed_child = -1;
    EXPECT_EQ(
        scheduler_graph_materialize_work(f.instance, work_graph(f), scalar, &range, 1, &failed_child),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );
    EXPECT_EQ(failed_child, 1);
    EXPECT_EQ(f.output.scalars[1], 0u);
    f.object.body.sources[1] = GraphScalarInheritance::self_value();
    f.object.body.tensors[1].owner_task_id = TaskId::make_param(99);
    const SchedulerGraphWork tensor{0, SchedulerGraphWorkRegion::TENSOR_POOL, 0, 1, TASKPAYLOAD_TENSOR_STRIDE};
    EXPECT_EQ(
        scheduler_graph_materialize_work(f.instance, work_graph(f), tensor, &range, 1, &failed_child),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );
    EXPECT_EQ(failed_child, 1);
}

TEST(HbgGraphMaterialize, InvalidWorkCannotWriteOutsideItsValidatedRanges) {
    MaterializeFixture f;
    poison_storage(&f);
    const auto initial = storage_bytes(f);
    std::array<SchedulerGraphWorkRange, 2> ranges{{{0, 1}, {0, 1}}};
    SchedulerGraphWork work{0, SchedulerGraphWorkRegion::SCALAR_POOL, 0, 2, 2 * sizeof(uint64_t)};
    auto execute = [&] {
        return scheduler_graph_materialize_work(f.instance, work_graph(f), work, ranges.data(), ranges.size());
    };
    EXPECT_EQ(execute(), SchedulerGraphResult::INVALID_ARGUMENTS);
    ranges[1] = {1, 2};
    EXPECT_EQ(execute(), SchedulerGraphResult::INVALID_ARGUMENTS);
    ranges[1] = {1, 1};
    work.write_bytes = 8;
    EXPECT_EQ(execute(), SchedulerGraphResult::INVALID_ARGUMENTS);
    work.write_bytes = 16;
    work.instance_index = 1;
    EXPECT_EQ(execute(), SchedulerGraphResult::INVALID_ARGUMENTS);
    work.instance_index = 0;
    work.first_range = UINT32_MAX;
    EXPECT_EQ(execute(), SchedulerGraphResult::INVALID_ARGUMENTS);
    EXPECT_EQ(storage_bytes(f), initial);
}

TEST(HbgGraphMaterialize, SerialWorkPreservesWholeInstanceBinding) {
    MaterializeFixture f;
    const SchedulerGraphWork work{0, SchedulerGraphWorkRegion::SERIAL_INSTANCE, 0, 0, 0};
    ASSERT_EQ(scheduler_graph_materialize_work(f.instance, work_graph(f), work, nullptr, 0), SchedulerGraphResult::OK);
    EXPECT_EQ(f.output.scalars[0], 97u);
    EXPECT_EQ(f.output.scalars[1], 29u);
    EXPECT_EQ(f.output.tasks[1].task.task_id, TaskId::make_sub_task(5, 1));
}

TEST(HbgGraphMaterialize, WritebackSpanIncludesBothSidesOfAnUnalignedRange) {
    SchedulerGraphWritebackSpan span{};
    ASSERT_TRUE(scheduler_graph_writeback_span(0x1038, 16, &span));
    EXPECT_EQ(span.first, 0x1000u);
    EXPECT_EQ(span.count, 2u);
    ASSERT_TRUE(scheduler_graph_writeback_span(0x1008, 64, &span));
    EXPECT_EQ(span.first, 0x1000u);
    EXPECT_EQ(span.count, 2u);
    ASSERT_TRUE(scheduler_graph_writeback_span(0x1000, 64, &span));
    EXPECT_EQ(span.count, 1u);
    ASSERT_TRUE(scheduler_graph_writeback_span(0, 0, &span));
    EXPECT_EQ(span.count, 0u);
    EXPECT_FALSE(scheduler_graph_writeback_span(UINT64_MAX - 7, 16, &span));
    EXPECT_FALSE(scheduler_graph_writeback_span(0x1000, 8, nullptr));
}

TEST(HbgGraphMaterialize, PoolAddressOverflowIsRejectedBeforeAnyConstruction) {
    MaterializeFixture f;
    poison_storage(&f);
    const auto initial = storage_bytes(f);
    f.instance.scalar_pool_address = UINT64_MAX - 7;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    EXPECT_EQ(storage_bytes(f), initial);
    f.instance.scalar_pool_address = reinterpret_cast<uint64_t>(f.output.scalars.data());
    f.instance.tensor_pool_address = UINT64_MAX - 63;
    EXPECT_EQ(f.materialize(), SchedulerGraphResult::INVALID_ARGUMENTS);
    EXPECT_EQ(storage_bytes(f), initial);
}

}  // namespace

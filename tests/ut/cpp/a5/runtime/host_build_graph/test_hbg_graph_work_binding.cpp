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
#include <atomic>
#include <cstdlib>
#include <memory>
#include <sstream>

#include "scheduler_graph_binding.h"
#include "scheduler_graph_work_export.h"

namespace {

struct WorkBindingFixture {
    static constexpr int32_t kChildren = 64;
    struct Body {
        GraphDefinition definition{};
        std::array<SubTaskDefinition, kChildren> tasks{};
        std::array<uint64_t, kChildren> offsets{};
        std::array<int32_t, kChildren + 1> rows{};
        std::array<uint16_t, kChildren> roots{};
        std::array<simpler::hbg::TensorData, kChildren> tensors{};
        std::array<uint64_t, kChildren> scalars{};
        std::array<GraphScalarInheritance, kChildren> inheritance{};
    };
    struct alignas(GRAPH_DEFINITION_OBJECT_ALIGN) Object {
        GraphDefinitionHeader header{};
        Body body{};
    };
    std::array<Object, 2> objects{};
    struct Image {
        std::array<ChipTaskStorage, 2> tasks{};
        std::array<simpler::hbg::Tensor, 2> tensors{};
        std::array<uint64_t, 2> scalars{};
    } image;
    SharedMemoryTaskHeader header{};
    std::array<std::atomic<ChipTaskState>, 2> states{};
    std::optional<SchedulerGraphBinding> binding;
    AicoreSchedulerLayout layout{};
    std::unique_ptr<void, decltype(&std::free)> storage{nullptr, &std::free};
    static constexpr uint64_t kDevice = 0x20000000;

    WorkBindingFixture() {
        for (auto &object : objects) {
            object.header.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
            object.header.graph_key = 17;
            object.header.definition_bytes = sizeof(Body);
            auto &definition = object.body.definition;
            definition.graph_key = 17;
            definition.total_bytes = sizeof(Body);
            definition.boundary_tensor_count = definition.boundary_scalar_count = 1;
            definition.off_sub_tasks = offsetof(Body, tasks);
            definition.off_sub_task_offsets = offsetof(Body, offsets);
            definition.off_fanin_offsets = definition.off_fanout_offsets = offsetof(Body, rows);
            definition.off_root_indices = offsetof(Body, roots);
            definition.off_tensors = offsetof(Body, tensors);
            definition.off_scalars = offsetof(Body, scalars);
            definition.off_scalar_inheritance = offsetof(Body, inheritance);
            set_children(object, kChildren);
            for (int32_t child = 0; child < kChildren; ++child) {
                auto &task = object.body.tasks[child];
                task.kernel_id[0] = task.kernel_id[2] = INVALID_KERNEL_ID;
                task.kernel_id[1] = 7;
                task.active_mask = LaneMask::aiv0();
                task.logical_block_num = task.total_required_subtasks = 1;
                task.tensor_count = task.scalar_count = 1;
                task.tensor_offset = task.scalar_offset = child;
                object.body.roots[child] = child;
                object.body.tensors[child].owner_task_id = TaskId::make_param(0);
                object.body.scalars[child] = child;
                object.body.inheritance[child] = GraphScalarInheritance::self_value();
            }
        }
        header.task_storage = image.tasks.data();
        header.task_states = states.data();
        header.total_tasks = 2;
        for (uint32_t parent = 0; parent < 2; ++parent) {
            auto &task = image.tasks[parent];
            task.task.task_id = TaskId::make_global(parent);
            task.task.graph_definition_offset = offsetof(Object, body);
            task.task.packed_buffer_base = reinterpret_cast<void *>(UINT64_C(0x40000000) + parent * 65536);
            task.task.packed_buffer_end = reinterpret_cast<void *>(
                reinterpret_cast<uint64_t>(task.task.packed_buffer_base) +
                objects[0].body.definition.execution_storage_bytes
            );
            task.slot.task_kind = TaskKind::GRAPH;
            task.payload.tensor_count = task.payload.scalar_count = 1;
            task.payload.bind_regions(image.tensors.data() + parent, image.scalars.data() + parent, nullptr);
            states[parent].store(CHIP_TASK_PENDING);
        }
    }

    static void set_children(Object &object, int32_t count) {
        auto &definition = object.body.definition;
        definition.task_count = definition.root_count = definition.tensor_arg_count = definition.scalar_arg_count =
            count;
        size_t bytes = 0;
        graph_execution_storage_bytes(count, count, count, 0, &bytes);
        definition.execution_storage_bytes = static_cast<uint32_t>(bytes);
    }

    bool prepare(uint32_t workers = 96, bool profiling = false) {
        binding = SchedulerGraphBinding::prepare(
            header, 2, reinterpret_cast<const std::byte *>(objects.data()), sizeof(objects), workers
        );
        if (!binding) return false;
        const auto count = static_cast<uint64_t>(binding->plan().node_count());
        const SchedulerSpmdRequirements spmd{96, 32, 0, 0};
        return scheduler_plan_layout(count, 0, count, &layout, false, count - 4, spmd, profiling) &&
               binding->reserve(&layout);
    }

    bool bind() {
        storage.reset(std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size));
        if (!storage || !scheduler_init_data_from_layout(storage.get(), layout, kDevice)) return false;
        sm_layout::SegmentOffsets segments{};
        return binding->bind(storage.get(), kDevice, layout, &image, 0x10000000, segments, 7, 0);
    }
};

TEST(HbgGraphWorkBinding, SharedDefinitionReusesRangesAndOwnsSeparateWorkCounters) {
    WorkBindingFixture f;
    ASSERT_TRUE(f.prepare());
    ASSERT_TRUE(f.bind());
    const auto &plan = f.layout.graph;
    ASSERT_GT(plan.work_count, 2u);
    EXPECT_EQ(plan.protocol_work_offset, 0u);
    const auto *items = scheduler_state_at<SchedulerGraphWork>(f.storage.get(), plan.work_offset);
    const auto *states = scheduler_state_at<SchedulerGraphWorkState>(f.storage.get(), plan.work_states_offset);
    const auto *instances =
        scheduler_state_at<SchedulerGraphInstanceControl>(f.storage.get(), plan.instance_controls_offset);
    const auto *build = scheduler_state_at<SchedulerGraphBuildControl>(f.storage.get(), plan.build_control_offset);
    EXPECT_EQ(build->work_count, plan.work_count);
    EXPECT_EQ(build->work_ranges_address, WorkBindingFixture::kDevice + plan.work_ranges_offset);
    EXPECT_EQ(build->work_states_address % 128, 0u);
    const uint32_t count = instances[0].work.count;
    ASSERT_EQ(count * 2u, plan.work_count);
    EXPECT_EQ(instances[0].work.begin, 0u);
    EXPECT_EQ(instances[1].work.begin, count);
    EXPECT_EQ(instances[1].work.remaining, count);
    EXPECT_EQ(instances[1].remaining_children, WorkBindingFixture::kChildren);
    for (uint32_t index = 0; index < count; ++index) {
        EXPECT_EQ(items[index].instance_index, 0u);
        EXPECT_EQ(items[index + count].instance_index, 1u);
        EXPECT_EQ(items[index].first_range, items[index + count].first_range);
        EXPECT_EQ(items[index].range_count, items[index + count].range_count);
        EXPECT_EQ(states[index].owner_state, 0u);
        EXPECT_EQ(states[index + count].owner_state, 0u);
    }
}

TEST(HbgGraphWorkBinding, SmallAndSingleWorkerRunsAllocateNoWorkStorage) {
    WorkBindingFixture f;
    ASSERT_TRUE(f.prepare(1));
    EXPECT_EQ(f.layout.graph.work_count, 0u);
    WorkBindingFixture::set_children(f.objects[0], 4);
    ASSERT_TRUE(f.prepare(96));
    const uint64_t parallel_size = f.layout.total_size;
    EXPECT_EQ(f.layout.graph.work_count, 0u);
    EXPECT_EQ(f.layout.graph.work_offset, 0u);
    EXPECT_EQ(f.layout.graph.work_ranges_offset, 0u);
    EXPECT_EQ(f.layout.graph.work_states_offset, 0u);
    ASSERT_TRUE(f.prepare(1));
    EXPECT_EQ(f.layout.total_size, parallel_size);
}

TEST(HbgGraphWorkBinding, MixedRunUsesOneSerialWorkForItsSmallInstance) {
    WorkBindingFixture f;
    WorkBindingFixture::set_children(f.objects[1], 4);
    f.image.tasks[1].task.graph_definition_offset =
        sizeof(WorkBindingFixture::Object) + offsetof(WorkBindingFixture::Object, body);
    ASSERT_TRUE(f.prepare());
    ASSERT_TRUE(f.bind());
    const auto &plan = f.layout.graph;
    const auto *items = scheduler_state_at<SchedulerGraphWork>(f.storage.get(), plan.work_offset);
    const auto *instances =
        scheduler_state_at<SchedulerGraphInstanceControl>(f.storage.get(), plan.instance_controls_offset);
    EXPECT_EQ(instances[1].work.count, 1u);
    EXPECT_EQ(items[instances[1].work.begin].region, SchedulerGraphWorkRegion::SERIAL_INSTANCE);
}

TEST(HbgGraphWorkBinding, UnplannableOffsetsKeepNativeDeviceValidation) {
    WorkBindingFixture f;
    f.objects[0].body.tasks[3].scalar_offset = WorkBindingFixture::kChildren + 1;
    f.image.tasks[1].task.graph_definition_offset =
        sizeof(WorkBindingFixture::Object) + offsetof(WorkBindingFixture::Object, body);
    ASSERT_TRUE(f.prepare());
    ASSERT_TRUE(f.binding->supported());
    ASSERT_TRUE(f.bind());
    const auto *items = scheduler_state_at<SchedulerGraphWork>(f.storage.get(), f.layout.graph.work_offset);
    EXPECT_EQ(items[0].instance_index, 0u);
    EXPECT_EQ(items[0].region, SchedulerGraphWorkRegion::SERIAL_INSTANCE);
    EXPECT_GT(f.layout.graph.work_count, 2u);
}

TEST(HbgGraphWorkBinding, AliasedInstanceWritebackLinesAreRejected) {
    WorkBindingFixture f;
    f.image.tasks[1].task.packed_buffer_base = f.image.tasks[0].task.packed_buffer_base;
    f.image.tasks[1].task.packed_buffer_end = f.image.tasks[0].task.packed_buffer_end;
    ASSERT_TRUE(f.prepare());
    EXPECT_FALSE(f.bind());
}

TEST(HbgGraphWorkBinding, DiagnosticStorageIsAllocatedOnlyWhenRequested) {
    WorkBindingFixture f;
    ASSERT_TRUE(f.prepare(96, false));
    const uint64_t plain_size = f.layout.total_size;
    EXPECT_EQ(f.layout.graph.protocol_work_offset, 0u);
    ASSERT_TRUE(f.prepare(96, true));
    ASSERT_TRUE(f.bind());
    EXPECT_GT(f.layout.total_size, plain_size);
    EXPECT_NE(f.layout.graph.protocol_work_offset, 0u);
    const auto *build =
        scheduler_state_at<SchedulerGraphBuildControl>(f.storage.get(), f.layout.graph.build_control_offset);
    EXPECT_EQ(build->protocol_work_address, WorkBindingFixture::kDevice + f.layout.graph.protocol_work_offset);
}

TEST(HbgGraphWorkBinding, FrozenPlanCannotBeReplacedByMutatedDeviceDescriptors) {
    WorkBindingFixture f;
    ASSERT_TRUE(f.prepare(96, true));
    const auto capture = f.binding->freeze_protocol(std::vector<SchedulerTaskMetadata>(f.binding->plan().node_count()));
    ASSERT_TRUE(f.bind());
    ASSERT_EQ(capture.manifest.materialize_work.size(), f.layout.graph.work_count);
    ASSERT_EQ(capture.manifest.materialize_ranges.size(), f.layout.graph.work_range_count);
    auto export_work = [&] {
        std::ostringstream text;
        {
            scheduler_protocol_export::JsonObject root(text);
            const scheduler_protocol_export::Image image(f.storage.get(), f.layout.total_size);
            scheduler_protocol_export::graph_work_plan(root, image, f.layout.graph, capture.manifest);
        }
        return text.str();
    };
    EXPECT_NE(export_work().find("\"work_plan\""), std::string::npos);
    auto *work = scheduler_state_at<SchedulerGraphWork>(f.storage.get(), f.layout.graph.work_offset);
    ++work[0].write_bytes;
    EXPECT_THROW(export_work(), std::invalid_argument);
    --work[0].write_bytes;
    auto *ranges = scheduler_state_at<SchedulerGraphWorkRange>(f.storage.get(), f.layout.graph.work_ranges_offset);
    ++ranges[0].count;
    EXPECT_THROW(export_work(), std::invalid_argument);
}

}  // namespace

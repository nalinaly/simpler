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

#include "hbg_graph_test_support.h"
#include "scheduler_graph_binding.h"
#include "scheduler_protocol_export.h"

namespace {

struct BindingFixture {
    MaterializeFixture definition;
    struct Image {
        std::array<ChipTaskStorage, 2> tasks{};
        std::array<simpler::hbg::Tensor, 2> tensors{};
        std::array<uint64_t, 2> scalars{};
        std::array<int32_t, 1> fanins{0};
    } image;
    SharedMemoryTaskHeader header{};
    std::array<std::atomic<ChipTaskState>, 2> states{};

    BindingFixture() {
        header.task_storage = image.tasks.data();
        header.task_states = states.data();
        header.total_tasks = 2;
        for (int32_t parent = 0; parent < 2; ++parent) {
            auto &task = image.tasks[parent];
            task.task.task_id = TaskId::make_global(parent);
            task.task.graph_definition_offset = offsetof(MaterializeFixture::Object, body);
            task.task.packed_buffer_base = reinterpret_cast<void *>(UINT64_C(0x40000000) + parent * 4096);
            task.task.packed_buffer_end = reinterpret_cast<void *>(
                reinterpret_cast<uint64_t>(task.task.packed_buffer_base) +
                definition.object.body.definition.required_heap +
                definition.object.body.definition.execution_storage_bytes
            );
            task.slot.task_kind = TaskKind::GRAPH;
            task.payload.tensor_count = task.payload.scalar_count = 1;
            task.payload.fanin_count = parent;
            task.payload.bind_regions(
                image.tensors.data() + parent, image.scalars.data() + parent,
                parent == 1 ? image.fanins.data() : nullptr
            );
            states[parent].store(CHIP_TASK_PENDING);
        }
    }

    std::optional<SchedulerGraphBinding> prepare() {
        return SchedulerGraphBinding::prepare(
            header, 2, reinterpret_cast<const std::byte *>(&definition.object), sizeof(definition.object)
        );
    }
};

TEST(HbgGraphBinding, TwoInstancesShareOneDefinitionAndUseIndependentStorage) {
    BindingFixture f;
    auto binding = f.prepare();
    ASSERT_TRUE(binding);
    ASSERT_TRUE(binding->supported());
    EXPECT_EQ(binding->plan().node_count(), 8);
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(8, 0, 8, &layout));
    ASSERT_TRUE(binding->reserve(&layout));
    EXPECT_EQ(layout.graph.instance_count, 2u);
    EXPECT_EQ(layout.graph.public_task_count, 2u);
    EXPECT_EQ(layout.graph.definition_bytes, sizeof(f.definition.object));
    EXPECT_EQ(layout.graph.external_fanin_count, 1u);
    std::unique_ptr<void, decltype(&std::free)> host(
        std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size), &std::free
    );
    ASSERT_NE(host, nullptr);
    constexpr uint64_t device = 0x20000000;
    constexpr uint64_t device_sm = 0x10000000;
    ASSERT_TRUE(scheduler_init_data_from_layout(host.get(), layout, device));
    sm_layout::SegmentOffsets segments{};
    segments.storage = offsetof(BindingFixture::Image, tasks);
    ASSERT_TRUE(binding->bind(host.get(), device, layout, &f.image, device_sm, segments, 9, 0));
    const auto *instances = scheduler_state_at<SchedulerGraphInstance>(host.get(), layout.graph.instances_offset);
    const auto *nodes = scheduler_state_at<SchedulerGraphNode>(host.get(), layout.graph.nodes_offset);
    const auto *controls =
        scheduler_state_at<SchedulerGraphInstanceControl>(host.get(), layout.graph.instance_controls_offset);
    const auto *fanins = scheduler_state_at<int32_t>(host.get(), layout.graph.external_fanins_offset);
    EXPECT_EQ(instances[0].definition_section_address, instances[1].definition_section_address);
    EXPECT_NE(instances[0].task_storage_address, instances[1].task_storage_address);
    EXPECT_EQ(instances[0].boundary_tensors_address, device_sm + offsetof(BindingFixture::Image, tensors));
    EXPECT_EQ(instances[1].boundary_scalars_address, device_sm + offsetof(BindingFixture::Image, scalars) + 8);
    EXPECT_EQ(nodes[1].public_task_id, TaskId::make_sub_task(0, 0));
    EXPECT_EQ(nodes[5].public_task_id, TaskId::make_sub_task(1, 0));
    EXPECT_EQ(nodes[2].fanin_address, nodes[6].fanin_address);
    EXPECT_EQ(nodes[2].fanin_ref_base, 1);
    EXPECT_EQ(nodes[6].fanin_ref_base, 5);
    EXPECT_EQ(nodes[4].fanin_count, 1);
    EXPECT_EQ(fanins[0], 3);
    EXPECT_EQ(controls[0].remaining_children, 2u);
    EXPECT_EQ(controls[1].enter_ref, 4);
}

TEST(HbgGraphBinding, RejectsNonPredecessorExternalEdgesBeforeAllocation) {
    BindingFixture f;
    f.image.fanins[0] = 1;
    EXPECT_FALSE(f.prepare());
}

TEST(HbgGraphBinding, MalformedCsrIsRejected) {
    BindingFixture f;
    f.definition.object.body.fanin_indices[0] = 1;
    EXPECT_FALSE(f.prepare());
    f.definition.object.body.fanin_indices[0] = 0;
    auto binding = f.prepare();
    ASSERT_TRUE(binding);
    EXPECT_TRUE(binding->supported());
}

TEST(HbgGraphBinding, KeepsTheOriginalMixMasksBlockCountsAndSyncStartAttribute) {
    for (uint8_t mask : {1, 2, 4, 3, 5, 6, 7}) {
        for (int16_t blocks : {1, 2, 32, 33, 512}) {
            for (bool sync : {false, true}) {
                if (sync && blocks == 1) continue;
                BindingFixture f;
                auto &task = f.definition.object.body.tasks[0];
                task.active_mask = mask;
                task.logical_block_num = blocks;
                task.total_required_subtasks = blocks * __builtin_popcount(mask);
                TaskAttrs attrs{};
                if (sync) attrs.set_sync_start();
                task.task_attrs = attrs.raw();
                for (uint32_t lane = 0; lane < 3; ++lane)
                    task.kernel_id[lane] = (mask & (1U << lane)) != 0 ? 7 : INVALID_KERNEL_ID;
                auto binding = f.prepare();
                ASSERT_TRUE(binding);
                EXPECT_TRUE(binding->supported()) << "mask=" << +mask << " blocks=" << blocks << " sync=" << sync;
                SchedulerGraphView ordinary{reinterpret_cast<uint64_t>(f.image.tasks.data()), 0, 2, 1};
                const auto child = binding->submission(1, ordinary);
                EXPECT_EQ(child.active_mask.raw(), mask);
                EXPECT_EQ(child.logical_block_num, blocks);
                EXPECT_EQ(child.task_attrs.requires_sync_start(), sync);
            }
        }
    }
}

TEST(HbgGraphBinding, CheckedTailLayoutRejectsOverflow) {
    BindingFixture f;
    auto binding = f.prepare();
    ASSERT_TRUE(binding);
    AicoreSchedulerLayout layout{};
    layout.total_size = UINT64_MAX - 32;
    EXPECT_FALSE(binding->reserve(&layout));
}

TEST(HbgGraphBinding, PredicateMetadataMismatchIsNotAValidFallbackRequest) {
    BindingFixture f;
    f.definition.object.body.tasks[0].predicate_slot = 1;
    EXPECT_FALSE(f.prepare());
}

TEST(HbgGraphBinding, PredicateSupportRemainsSingleBlockAndSingleLane) {
    BindingFixture f;
    f.definition.enable_predicates();
    auto binding = f.prepare();
    ASSERT_TRUE(binding);
    EXPECT_TRUE(binding->supported());
    auto &task = f.definition.object.body.tasks[0];
    task.logical_block_num = task.total_required_subtasks = 2;
    binding = f.prepare();
    ASSERT_TRUE(binding);
    EXPECT_FALSE(binding->supported());
    task.logical_block_num = 1;
    task.active_mask = 3;
    task.kernel_id[0] = 6;
    binding = f.prepare();
    ASSERT_TRUE(binding);
    EXPECT_FALSE(binding->supported());
}

TEST(HbgGraphBinding, PredicateImageValidationPrecedesBackendSelection) {
    BindingFixture f;
    f.definition.enable_predicates();
    auto &predicate = f.definition.object.body.predicates[0];
    predicate.op = 0;
    EXPECT_FALSE(f.prepare());
    predicate.op = static_cast<uint8_t>(PredicateOp::EQ);
    predicate.elem_size = 8;
    EXPECT_FALSE(f.prepare());
    predicate.elem_size = 4;
    predicate.operand.dtype = static_cast<DataType>(255);
    EXPECT_FALSE(f.prepare());
    predicate.operand.dtype = DataType::FLOAT32;
    predicate.operand.owner_task_id = TaskId::make_sub_task(0, 0);
    EXPECT_FALSE(f.prepare());
    predicate.operand.owner_task_id = TaskId::make_param(1);
    EXPECT_FALSE(f.prepare());
    predicate.operand.owner_task_id = TaskId::make_param(0);
    f.definition.object.body.definition.off_predicates = sizeof(MaterializeFixture::Body) - 8;
    EXPECT_FALSE(f.prepare());
}

TEST(HbgGraphBinding, RejectsMetadataLookupOutsideTheExpandedRun) {
    BindingFixture f;
    auto binding = f.prepare();
    ASSERT_TRUE(binding);
    SchedulerGraphView ordinary{reinterpret_cast<uint64_t>(f.image.tasks.data()), 0, 2, 1};
    EXPECT_EQ(binding->submission(-1, ordinary).status, SchedulerGraphResult::INVALID_TASK_ID);
    EXPECT_EQ(
        binding->submission(binding->plan().node_count(), ordinary).status, SchedulerGraphResult::INVALID_TASK_ID
    );
}

TEST(HbgGraphBinding, ProtocolSnapshotOwnsSharedDefinitionsAndExpandedDependencies) {
    BindingFixture f;
    auto binding = f.prepare();
    ASSERT_TRUE(binding);
    std::vector<SchedulerTaskMetadata> metadata(8);
    metadata[1].logical_block_num = 1;
    const auto snapshot = binding->freeze_protocol(metadata);
    ASSERT_EQ(snapshot.tasks.size(), 8u);
    EXPECT_EQ(snapshot.tasks[0].dependencies, std::vector<int32_t>{});
    EXPECT_EQ(snapshot.tasks[1].dependencies, std::vector<int32_t>{0});
    EXPECT_EQ(snapshot.tasks[2].dependencies, std::vector<int32_t>{1});
    EXPECT_EQ(snapshot.tasks[3].dependencies, std::vector<int32_t>{});
    EXPECT_EQ(snapshot.tasks[4].dependencies, std::vector<int32_t>{3});
    EXPECT_EQ(snapshot.tasks[5].dependencies, std::vector<int32_t>{4});
    EXPECT_EQ(snapshot.tasks[6].dependencies, std::vector<int32_t>{5});
    EXPECT_EQ(snapshot.tasks[7].dependencies, std::vector<int32_t>{});
    ASSERT_EQ(snapshot.manifest.ranges.size(), 2u);
    EXPECT_EQ(snapshot.manifest.ranges[1].begin, 4);
    EXPECT_EQ(snapshot.manifest.ranges[1].end, 8);
    EXPECT_EQ(snapshot.manifest.instance_definitions, (std::vector<uint32_t>{0, 0}));
    ASSERT_EQ(snapshot.manifest.definitions.size(), 1u);
    const auto &definition = snapshot.manifest.definitions[0];
    EXPECT_EQ(definition.header.full_key, 17u);
    EXPECT_EQ(definition.tasks.size(), 2u);
    EXPECT_EQ(definition.fanin_rows, (std::vector<int32_t>{0, 0, 1}));
    EXPECT_EQ(definition.fanin_indices, std::vector<uint16_t>{0});
    f.definition.object.body.fanin_indices[0] = 1;
    f.definition.object.body.tasks[0].kernel_id[1] = 9;
    f.image.fanins[0] = 1;
    metadata[1].logical_block_num = 512;
    EXPECT_EQ(definition.fanin_indices[0], 0);
    EXPECT_EQ(definition.tasks[0].kernel_id[1], 7);
    EXPECT_EQ(snapshot.tasks[4].dependencies, std::vector<int32_t>{3});
    EXPECT_EQ(snapshot.tasks[1].metadata.logical_block_num, 1);
    EXPECT_EQ(f.definition.output.scalars[0], 0u);
    EXPECT_EQ(f.definition.output.tasks[0].task.task_id, TaskId::invalid());
}

TEST(HbgGraphBinding, ProtocolSnapshotRefusesPublicCountInPlaceOfExpandedCount) {
    BindingFixture f;
    const auto binding = f.prepare();
    ASSERT_TRUE(binding);
    EXPECT_THROW(binding->freeze_protocol(std::vector<SchedulerTaskMetadata>(2)), std::invalid_argument);
}

TEST(HbgGraphBinding, ProtocolSnapshotKeepsOrdinaryNodesAndEmptyCsrRoots) {
    BindingFixture f;
    f.image.tasks[0].slot.task_kind = TaskKind::DUMMY;
    f.definition.object.body.definition.edge_count = 0;
    f.definition.object.body.definition.off_fanin_indices = 0;
    f.definition.object.body.fanout_rows[1] = 0;
    GraphExecutionStorageLayout storage{};
    auto &definition = f.definition.object.body.definition;
    definition.task_count = definition.tensor_arg_count = definition.scalar_arg_count = 1;
    ASSERT_TRUE(graph_execution_storage_layout(1, 1, 1, 0, &storage));
    definition.execution_storage_bytes = storage.total_bytes;
    auto binding = f.prepare();
    ASSERT_TRUE(binding);
    const auto snapshot = binding->freeze_protocol(std::vector<SchedulerTaskMetadata>(4));
    EXPECT_EQ(snapshot.manifest.ranges[0].instance_index, -1);
    EXPECT_EQ(snapshot.manifest.ranges[1].instance_index, 0);
    EXPECT_EQ(snapshot.manifest.instance_definitions, std::vector<uint32_t>{0});
    EXPECT_EQ(snapshot.tasks[1].dependencies, std::vector<int32_t>{0});
    EXPECT_EQ(snapshot.tasks[2].dependencies, std::vector<int32_t>{1});
    EXPECT_TRUE(snapshot.tasks[3].dependencies.empty());
    EXPECT_TRUE(snapshot.manifest.definitions[0].fanin_indices.empty());
}

TEST(HbgGraphBinding, GraphProtocolStorageIsBoundedAndAbsentWithoutL4) {
    BindingFixture f;
    const auto binding = f.prepare();
    ASSERT_TRUE(binding);
    for (bool enabled : {false, true}) {
        AicoreSchedulerLayout layout{};
        ASSERT_TRUE(scheduler_plan_layout(8, 0, 4, &layout, false, 4, {3, 1, 0, 0}, enabled));
        ASSERT_TRUE(binding->reserve(&layout));
        EXPECT_EQ(layout.graph.protocol_instances_offset != 0, enabled);
        EXPECT_EQ(layout.graph.protocol_contributions_offset != 0, enabled);
        EXPECT_EQ(layout.graph.protocol_workers_offset != 0, enabled);
        if (enabled) {
            EXPECT_GE(layout.graph.protocol_instances_offset, layout.protocol.storage_end_offset);
            EXPECT_EQ(layout.graph.protocol_instances_offset % 128, 0u);
            EXPECT_EQ(layout.graph.protocol_contributions_offset % 128, 0u);
            EXPECT_EQ(layout.graph.protocol_workers_offset % 128, 0u);
            EXPECT_LE(
                layout.graph.protocol_instances_offset + 2 * sizeof(SchedulerGraphProtocolInstance),
                layout.graph.protocol_contributions_offset
            );
            EXPECT_LE(
                layout.graph.protocol_contributions_offset + 8 * sizeof(SchedulerGraphProtocolContribution),
                layout.graph.protocol_workers_offset
            );
            EXPECT_LE(
                layout.graph.protocol_workers_offset + 3 * sizeof(SchedulerGraphProtocolWorker), layout.total_size
            );
        }
        std::unique_ptr<void, decltype(&std::free)> host(
            std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size), &std::free
        );
        ASSERT_NE(host, nullptr);
        constexpr uint64_t device = 0x20000000;
        ASSERT_TRUE(scheduler_init_data_from_layout(host.get(), layout, device));
        sm_layout::SegmentOffsets segments{};
        segments.storage = offsetof(BindingFixture::Image, tasks);
        ASSERT_TRUE(binding->bind(host.get(), device, layout, &f.image, 0x10000000, segments, 9, 0));
        const auto &build =
            *scheduler_state_at<SchedulerGraphBuildControl>(host.get(), layout.graph.build_control_offset);
        EXPECT_EQ(build.protocol_instances_address, enabled ? device + layout.graph.protocol_instances_offset : 0u);
        EXPECT_EQ(
            build.protocol_contributions_address, enabled ? device + layout.graph.protocol_contributions_offset : 0u
        );
        EXPECT_EQ(build.protocol_workers_address, enabled ? device + layout.graph.protocol_workers_offset : 0u);
        const auto *controls =
            scheduler_state_at<SchedulerGraphInstanceControl>(host.get(), layout.graph.instance_controls_offset);
        EXPECT_EQ(controls[0].instance_index, 0u);
        EXPECT_EQ(controls[1].instance_index, 1u);
    }
}

TEST(HbgGraphBinding, GraphExportRequiresItsManifestAndPreservesActualIdentityAndPartialRecords) {
    BindingFixture f;
    const auto binding = f.prepare();
    ASSERT_TRUE(binding);
    const auto snapshot = binding->freeze_protocol(std::vector<SchedulerTaskMetadata>(8));
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(8, 0, 4, &layout, false, 4, {3, 1, 0, 0}, true));
    ASSERT_TRUE(binding->reserve(&layout));
    std::unique_ptr<void, decltype(&std::free)> host(
        std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size), &std::free
    );
    ASSERT_NE(host, nullptr);
    constexpr uint64_t device = 0x20000000;
    ASSERT_TRUE(scheduler_init_data_from_layout(host.get(), layout, device));
    ASSERT_TRUE(scheduler_init_spmd_task_records(host.get(), layout, 9));
    sm_layout::SegmentOffsets segments{};
    segments.storage = offsetof(BindingFixture::Image, tasks);
    ASSERT_TRUE(binding->bind(host.get(), device, layout, &f.image, 0x10000000, segments, 9, 0));
    EXPECT_THROW(
        scheduler_export_protocol(host.get(), layout.total_size, layout, 9, snapshot.tasks), std::invalid_argument
    );
    auto *records =
        scheduler_state_at<SchedulerGraphProtocolInstance>(host.get(), layout.graph.protocol_instances_offset);
    records[1].materialize_before = {0, 1};
    records[1].status = 6;
    const auto dump = [&] {
        return scheduler_export_protocol(
            host.get(), layout.total_size, layout, 9, snapshot.tasks, &snapshot.manifest, device
        );
    };
    const auto json = dump();
    EXPECT_NE(json.find("\"schema\":\"hbg-native-protocol-v9\""), std::string::npos);
    EXPECT_NE(json.find("\"instance_definitions\":[0,0]"), std::string::npos);
    EXPECT_NE(json.find("\"fanin_rows\":[0,0,1]"), std::string::npos);
    EXPECT_NE(json.find("\"materialize_before\":[0,1]"), std::string::npos);
    EXPECT_NE(json.find("\"materialize_after\":[0,0]"), std::string::npos);
    EXPECT_NE(json.find("\"status\":6"), std::string::npos);
    EXPECT_NE(json.find("\"public_task_id\":1,\"sub_task_id\":0"), std::string::npos);
    EXPECT_NE(json.find("\"public_completed\":0"), std::string::npos);
    EXPECT_NE(json.find("\"remaining_instances\":2"), std::string::npos);
    auto *nodes = scheduler_state_at<SchedulerGraphNode>(host.get(), layout.graph.nodes_offset);
    nodes[5].public_task_id = TaskId::make_sub_task(7, 9);
    EXPECT_NE(dump().find("\"public_task_id\":7,\"sub_task_id\":9"), std::string::npos);
    nodes[6].fanin_address = 8;
    EXPECT_THROW(dump(), std::invalid_argument);
}

}  // namespace

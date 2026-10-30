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
#include <numeric>
#include <vector>

#include "host_build_graph/runtime_types.h"
#include "scheduler/scheduler_graph.h"

namespace {

using Kind = SchedulerGraphNodeKind;

struct MappedGraph {
    std::array<ChipTaskStorage, 2> tasks{};
    std::array<SchedulerGraphNode, 3> nodes{};
    std::array<int32_t, 1> dependencies{0};

    MappedGraph() {
        tasks[0].task.task_id = TaskId::make_sub_task(17, 0);
        tasks[1].task.task_id = TaskId::make_global(16);
        for (auto &task : tasks) {
            task.task.kernel_id[0] = INVALID_KERNEL_ID;
            task.task.kernel_id[1] = 7;
            task.task.kernel_id[2] = INVALID_KERNEL_ID;
            task.payload.fanin_count = 0;
        }
        nodes[0] = {reinterpret_cast<uint64_t>(&tasks[1]), tasks[1].task.task_id, 0, 0, 0, Kind::ORDINARY};
        nodes[1] = {0, TaskId::make_global(17), 0, 0, 0, Kind::ENTER};
        nodes[2] = {
            reinterpret_cast<uint64_t>(&tasks[0]),
            tasks[0].task.task_id,
            reinterpret_cast<uint64_t>(dependencies.data()),
            0,
            1,
            Kind::CHILD
        };
    }

    SchedulerGraphView view() {
        return {0, 0, nodes.size(), nodes.size() - 1, reinterpret_cast<uint64_t>(nodes.data())};
    }
};

TEST(HbgGraphView, AddressesMappedStorageAndChecksPublicRatherThanDenseIdentity) {
    MappedGraph image;
    auto graph = image.view();
    EXPECT_EQ(scheduler_graph_descriptor(graph, 2), reinterpret_cast<uint8_t *>(&image.tasks[0].task));
    EXPECT_EQ(scheduler_graph_payload(graph, 2), reinterpret_cast<uint8_t *>(&image.tasks[0].payload));
    EXPECT_EQ(scheduler_graph_descriptor(graph, 0), reinterpret_cast<uint8_t *>(&image.tasks[1].task));
    SchedulerTaskShape shape{};
    ASSERT_EQ(scheduler_classify_task_shape(graph, 2, &shape), SchedulerGraphResult::OK);
    EXPECT_EQ(shape.task_ref, 2);
    EXPECT_EQ(shape.active_mask, 2);
    EXPECT_EQ(shape.kernel_ids[1], 7);
    image.tasks[0].task.task_id = TaskId::make_sub_task(18, 0);
    EXPECT_EQ(scheduler_classify_task_shape(graph, 2, &shape), SchedulerGraphResult::INVALID_TASK_ID);
}

TEST(HbgGraphView, SavedCsrRemainsVisibleWhenPayloadFaninIsZero) {
    MappedGraph image;
    auto graph = image.view();
    ASSERT_EQ(image.tasks[0].payload.fanin_count, 0);
    EXPECT_EQ(scheduler_graph_fanin_count(graph, 2), 1);
    EXPECT_EQ(scheduler_graph_fanin_id(graph, 2, 0), 0);
    EXPECT_EQ(scheduler_graph_fanin_id(graph, 2, -1), -1);
    EXPECT_EQ(scheduler_graph_fanin_id(graph, 2, 1), -1);
    SchedulerTaskShape shape{};
    image.nodes[2].fanin_address = 0;
    EXPECT_EQ(scheduler_classify_task_shape(graph, 2, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
}

TEST(HbgGraphView, InternalDependenciesCanExceedTheGlobalInlineFaninLimit) {
    constexpr int32_t kFanin = 129;
    std::vector<SchedulerGraphNode> nodes(kFanin + 1);
    std::vector<int32_t> deps(kFanin);
    std::iota(deps.begin(), deps.end(), 0);
    ChipTaskStorage task{};
    task.task.task_id = TaskId::make_sub_task(0, kFanin);
    task.task.kernel_id[0] = 7;
    task.task.kernel_id[1] = task.task.kernel_id[2] = INVALID_KERNEL_ID;
    nodes.back() = {reinterpret_cast<uint64_t>(&task),
                    task.task.task_id,
                    reinterpret_cast<uint64_t>(deps.data()),
                    0,
                    kFanin,
                    Kind::CHILD};
    SchedulerGraphView graph{0, 0, nodes.size(), nodes.size() - 1, reinterpret_cast<uint64_t>(nodes.data())};
    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(graph, kFanin, &shape), SchedulerGraphResult::OK);
    deps.back() = 0;
    EXPECT_EQ(scheduler_classify_task_shape(graph, kFanin, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
    deps.back() = kFanin;
    EXPECT_EQ(scheduler_classify_task_shape(graph, kFanin, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
    deps.back() = kFanin - 1;
    nodes.back().kind = Kind::ORDINARY;
    EXPECT_EQ(scheduler_classify_task_shape(graph, kFanin, &shape), SchedulerGraphResult::INVALID_FANIN_COUNT);
    nodes.back().kind = Kind::CHILD;
    nodes.back().fanin_count = SCHEDULER_GRAPH_MAX_BODY_FANIN + 1;
    EXPECT_EQ(scheduler_classify_task_shape(graph, kFanin, &shape), SchedulerGraphResult::INVALID_FANIN_COUNT);
}

TEST(HbgGraphView, ServiceNodesNeverPretendToHaveKernelPayloads) {
    MappedGraph image;
    SchedulerTaskShape shape{};
    SchedulerTaskPayloadArguments args{};
    for (auto kind : {Kind::ENTER, Kind::EXIT}) {
        image.nodes[1].kind = kind;
        EXPECT_EQ(scheduler_classify_task_shape(image.view(), 1, &shape), SchedulerGraphResult::UNSUPPORTED_SHAPE);
        EXPECT_EQ(
            scheduler_parse_task_payload_arguments(image.view(), 1, &args), SchedulerGraphResult::INVALID_ARGUMENTS
        );
    }
}

TEST(HbgGraphView, InvalidNodeKindAndMissingLeafStorageAreRejected) {
    MappedGraph image;
    SchedulerTaskShape shape{};
    image.nodes[2].kind = static_cast<Kind>(99);
    EXPECT_EQ(scheduler_classify_task_shape(image.view(), 2, &shape), SchedulerGraphResult::INVALID_ARGUMENTS);
    image.nodes[2].kind = Kind::CHILD;
    image.nodes[2].storage_address = 0;
    EXPECT_EQ(scheduler_classify_task_shape(image.view(), 2, &shape), SchedulerGraphResult::INVALID_ARGUMENTS);
}

TEST(HbgGraphView, MappedArgumentsUseTheActualChildPayloadAndPools) {
    struct Arguments {
        ChipTaskStorage task{};
        std::array<simpler::hbg::Tensor, 1> tensors{};
        std::array<uint64_t, 2> scalars{13, 29};
    } image;
    image.task.task.task_id = TaskId::make_sub_task(5, 2);
    image.task.payload.tensor_count = 1;
    image.task.payload.scalar_count = 2;
    image.task.payload.bind_regions(image.tensors.data(), image.scalars.data(), nullptr);
    SchedulerGraphNode node{reinterpret_cast<uint64_t>(&image.task), image.task.task.task_id, 0, 0, 0, Kind::CHILD};
    SchedulerGraphView graph{0, 0, 1, 0, reinterpret_cast<uint64_t>(&node)};
    SchedulerTaskPayloadArguments args{};
    ASSERT_EQ(scheduler_parse_task_payload_arguments(graph, 0, &args), SchedulerGraphResult::OK);
    EXPECT_EQ(args.tensor_count, 1);
    EXPECT_EQ(args.scalar_count, 2);
    EXPECT_EQ(args.tensors, reinterpret_cast<uint8_t *>(image.tensors.data()));
    EXPECT_EQ(args.scalars, image.scalars.data());
    EXPECT_EQ(args.scalars[1], 29u);
}

TEST(HbgGraphView, TwoInstancesReadOneDefinitionCsrWithDifferentDenseBases) {
    std::array<uint16_t, 1> recorded_fanin{0};
    std::array<ChipTaskStorage, 2> tasks{};
    std::array<SchedulerGraphNode, 8> nodes{};
    for (int32_t instance = 0; instance < 2; ++instance) {
        const int32_t ref = instance * 4 + 2;
        auto &task = tasks[instance];
        task.task.task_id = TaskId::make_sub_task(instance, 1);
        task.task.kernel_id[0] = 7;
        task.task.kernel_id[1] = task.task.kernel_id[2] = INVALID_KERNEL_ID;
        nodes[ref] = {
            reinterpret_cast<uint64_t>(&task),
            task.task.task_id,
            reinterpret_cast<uint64_t>(recorded_fanin.data()),
            0,
            1,
            Kind::CHILD,
            instance * 4 + 1,
            SchedulerGraphFaninKind::DEFINITION_CSR
        };
    }
    SchedulerGraphView graph{0, 0, nodes.size(), nodes.size() - 1, reinterpret_cast<uint64_t>(nodes.data())};
    EXPECT_EQ(scheduler_graph_fanin_id(graph, 2, 0), 1);
    EXPECT_EQ(scheduler_graph_fanin_id(graph, 6, 0), 5);
    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(graph, 2, &shape), SchedulerGraphResult::OK);
    EXPECT_EQ(scheduler_classify_task_shape(graph, 6, &shape), SchedulerGraphResult::OK);
    recorded_fanin[0] = 1;
    EXPECT_EQ(scheduler_classify_task_shape(graph, 2, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
    EXPECT_EQ(scheduler_classify_task_shape(graph, 6, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
}

TEST(HbgGraphView, RootGateIsAnInlineReferenceWithoutAnEdgeAllocation) {
    MappedGraph image;
    auto &node = image.nodes[2];
    node.fanin_address = 0;
    node.fanin_kind = SchedulerGraphFaninKind::ENTER_GATE;
    node.fanin_ref_base = 1;
    EXPECT_EQ(scheduler_graph_fanin_count(image.view(), 2), 1);
    EXPECT_EQ(scheduler_graph_fanin_id(image.view(), 2, 0), 1);
    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(image.view(), 2, &shape), SchedulerGraphResult::OK);
    node.fanin_count = 2;
    EXPECT_EQ(scheduler_classify_task_shape(image.view(), 2, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
    node.fanin_count = 1;
    node.fanin_kind = static_cast<SchedulerGraphFaninKind>(99);
    EXPECT_EQ(scheduler_classify_task_shape(image.view(), 2, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
}

}  // namespace

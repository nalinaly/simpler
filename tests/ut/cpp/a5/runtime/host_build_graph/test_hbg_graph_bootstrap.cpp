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
#include <chrono>
#include <thread>

#include "hbg_graph_test_support.h"
#include "aicore/aicore_executor.cpp"

namespace {

struct BootstrapFixture {
    struct State {
        std::array<uint8_t, 128> padding{};
        SchedulerGraphSummary summary{};
    } state;
    std::array<MaterializeFixture, 2> payloads;
    std::array<SchedulerGraphInstance, 2> instances{};
    std::array<SchedulerGraphNode, 8> nodes{};
    SchedulerGraphBuildControl build{};
    SchedulerWorkerContext context{};
    SchedulerRunControl run{};
    SchedulerGraphView graph{};

    BootstrapFixture() {
        state.summary.run_epoch = build.run_epoch = 7;
        build.instance_count = build.remaining_instances = build.public_task_count = 2;
        for (int32_t index = 0; index < 2; ++index) {
            auto &payload = payloads[index];
            auto &control = payload.control;
            control.run_epoch = 7;
            control.enter_ref = index * 4;
            control.child_begin = index * 4 + 1;
            control.exit_ref = index * 4 + 3;
            instances[index] = payload.instance;
            instances[index].parent_id = index;
            instances[index].definition_section_address = reinterpret_cast<uint64_t>(&payloads[0].object);
            for (int32_t local = 0; local < 4; ++local) {
                auto &node = nodes[index * 4 + local];
                node = payload.nodes[local];
                node.instance_control_address = reinterpret_cast<uint64_t>(&control);
                node.kind = local == 0 ? SchedulerGraphNodeKind::ENTER :
                            local == 3 ? SchedulerGraphNodeKind::EXIT :
                                         SchedulerGraphNodeKind::CHILD;
                node.public_task_id =
                    local == 0 || local == 3 ? TaskId::make_global(index) : TaskId::make_sub_task(index, local - 1);
            }
        }
        graph = {0,
                 0,
                 nodes.size(),
                 nodes.size() - 1,
                 reinterpret_cast<uint64_t>(nodes.data()),
                 reinterpret_cast<uint64_t>(&build)};
        context.graph_instance_count = 2;
        context.graph_instances_address = reinterpret_cast<uint64_t>(instances.data());
        context.scheduler_state_base_address = reinterpret_cast<uint64_t>(&state);
        context.graph_summary_offset = offsetof(State, summary);
        context.runtime_worker_count = 3;
        context.core_type = static_cast<int32_t>(CoreType::AIV);
        context.cluster_worker_ids[0] = 0;
        context.cluster_worker_ids[1] = 1;
        context.cluster_worker_ids[2] = 2;
    }

    bool execute(uint64_t worker = 1, uint64_t timeout = PLATFORM_PROF_SYS_CNT_FREQ) {
        auto current = context;
        current.worker_index = worker;
        return bootstrap_graph_instances(graph, &current, &run, timeout);
    }
};

TEST(HbgGraphBootstrap, SharedDefinitionPublishesEveryInstanceBeforeReturn) {
    BootstrapFixture f;
    ASSERT_TRUE(f.execute());
    EXPECT_EQ(f.build.remaining_instances, 0u);
    EXPECT_EQ(f.build.public_completed, 0u);
    for (int32_t index = 0; index < 2; ++index) {
        EXPECT_EQ(f.payloads[index].control.materialize_owner, 2u);
        EXPECT_EQ(f.payloads[index].control.materialize_state, 2u);
        EXPECT_EQ(f.payloads[index].output.tasks[0].task.task_id, TaskId::make_sub_task(index, 0));
        EXPECT_EQ(f.payloads[index].output.scalars[0], 97u);
    }
}

TEST(HbgGraphBootstrap, OldBuildEpochIsRejectedBeforeAnyOwnershipOrOutputWrite) {
    BootstrapFixture f;
    f.build.run_epoch = 6;
    EXPECT_FALSE(f.execute());
    EXPECT_EQ(f.build.next_instance, 0u);
    EXPECT_EQ(f.build.remaining_instances, 2u);
    EXPECT_EQ(f.payloads[0].control.materialize_owner, 0u);
    EXPECT_EQ(f.payloads[0].output.scalars[0], 0u);
}

TEST(HbgGraphBootstrap, OldInstanceEpochDoesNotPublishMaterialization) {
    BootstrapFixture f;
    f.payloads[0].control.run_epoch = 6;
    EXPECT_FALSE(f.execute());
    EXPECT_EQ(f.build.remaining_instances, 2u);
    EXPECT_EQ(f.payloads[0].control.materialize_owner, 0u);
    EXPECT_EQ(f.payloads[0].output.scalars[0], 0u);
}

TEST(HbgGraphBootstrap, InvalidControlAlignmentCannotCorruptAnAdjacentField) {
    BootstrapFixture f;
    f.instances[0].control_address += 8;
    EXPECT_FALSE(f.execute());
    EXPECT_EQ(f.payloads[0].control.materialize_owner, 0u);
    EXPECT_EQ(f.payloads[0].control.materialize_state, 0u);
    EXPECT_EQ(f.payloads[0].control.run_epoch, 7u);
    EXPECT_EQ(f.build.remaining_instances, 2u);
}

TEST(HbgGraphBootstrap, ExistingOwnerCannotBeOverwrittenByARepeatedTicket) {
    BootstrapFixture f;
    f.payloads[0].control.materialize_owner = 3;
    EXPECT_FALSE(f.execute());
    EXPECT_EQ(f.payloads[0].control.materialize_owner, 3u);
    EXPECT_EQ(f.payloads[0].control.materialize_state, 0u);
    EXPECT_EQ(f.payloads[0].output.scalars[0], 0u);
    EXPECT_EQ(f.build.remaining_instances, 2u);
}

TEST(HbgGraphBootstrap, FailedOwnerWakesOtherWorkersWithoutDecrementingTheBuildCount) {
    BootstrapFixture f;
    f.payloads[0].object.header.magic = 0;
    EXPECT_FALSE(f.execute());
    EXPECT_EQ(f.payloads[0].control.materialize_state, 3u);
    EXPECT_EQ(f.build.remaining_instances, 2u);
    EXPECT_NE(scheduler_read_error(&f.run), 0u);
    EXPECT_FALSE(f.execute(2));
    EXPECT_EQ(f.build.public_completed, 0u);
}

TEST(HbgGraphBootstrap, WaitingWorkerCannotPassAPausedOwner) {
    BootstrapFixture f;
    f.build.next_instance = 2;
    std::atomic<bool> returned{false};
    bool success = false;
    std::thread waiter([&] {
        success = f.execute(2);
        returned.store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (scheduler_gm_fetch_add(f.build.next_instance, UINT64_C(0)) == 2 &&
           std::chrono::steady_clock::now() < deadline) {}
    EXPECT_GT(scheduler_gm_fetch_add(f.build.next_instance, UINT64_C(0)), 2u);
    EXPECT_FALSE(returned.load(std::memory_order_acquire));
    for (int32_t index = 0; index < 2; ++index) {
        EXPECT_EQ(scheduler_graph_materialize(f.instances[index], f.graph), SchedulerGraphResult::OK);
        scheduler_gm_store(f.payloads[index].control.materialize_state, UINT64_C(2));
        scheduler_gm_fetch_add(f.build.remaining_instances, UINT64_MAX);
    }
    waiter.join();
    EXPECT_TRUE(success);
    EXPECT_TRUE(returned.load(std::memory_order_acquire));
    EXPECT_EQ(f.payloads[1].output.scalars[0], 97u);
}

TEST(HbgGraphBootstrap, AbandonedOwnerTimesOutWithoutPublicSuccess) {
    BootstrapFixture f;
    f.build.next_instance = 2;
    EXPECT_FALSE(f.execute(1, PLATFORM_PROF_SYS_CNT_FREQ / 1000));
    EXPECT_EQ(scheduler_read_error(&f.run), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_EQ(f.run.error_site, static_cast<uint64_t>(SchedulerErrorSite::GRAPH_MATERIALIZE_TIMEOUT));
    EXPECT_EQ(f.build.remaining_instances, 2u);
    EXPECT_EQ(f.build.public_completed, 0u);
}

}  // namespace

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
#include "scheduler/scheduler_graph_protocol_types.h"
#include "aicore/aicore_executor.cpp"

namespace {

struct BootstrapFixture {
    struct State {
        std::array<uint8_t, 128> padding{};
        SchedulerGraphSummary summary{};
        SchedulerProtocolLayout protocol{};
        std::array<SchedulerProtocolTaskTrace, 8> tasks{};
        std::array<SchedulerProtocolWorkerTrace, 3> workers{};
        std::array<SchedulerGraphProtocolInstance, 2> graph_instances{};
        std::array<SchedulerGraphProtocolContribution, 8> contributions{};
        std::array<SchedulerGraphProtocolWorker, 3> graph_workers{};
    } state;
    std::array<MaterializeFixture, 2> payloads;
    std::array<SchedulerGraphInstance, 2> instances{};
    std::array<SchedulerGraphNode, 8> nodes{};
    SchedulerGraphBuildControl build{};
    SchedulerWorkerContext context{};
    SchedulerRunControl run{};
    SchedulerGraphView graph{};
    SchedulerLocalProfilingState profile{};
    SchedulerLocalState local{};

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
            control.instance_index = index;
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

    void enable_protocol(uint32_t worker = 1) {
        auto &wire = state.protocol;
        wire.magic = SCHEDULER_PROTOCOL_MAGIC;
        wire.run_epoch = 7;
        wire.task_count = nodes.size();
        wire.worker_count = state.workers.size();
        wire.tasks_offset = offsetof(State, tasks);
        wire.members_offset = wire.blocks_offset = wire.executions_offset = wire.workers_offset =
            offsetof(State, workers);
        wire.storage_end_offset = wire.workers_offset + sizeof(state.workers);
        state.summary.protocol_layout_offset = offsetof(State, protocol);
        for (uint32_t index = 0; index < state.workers.size(); ++index) {
            state.workers[index].run_epoch = 7;
            state.workers[index].worker_index = index;
        }
        build.protocol_instances_address = reinterpret_cast<uint64_t>(state.graph_instances.data());
        build.protocol_contributions_address = reinterpret_cast<uint64_t>(state.contributions.data());
        build.protocol_workers_address = reinterpret_cast<uint64_t>(state.graph_workers.data());
        local.profiling = &profile;
        local.config.graph = &graph;
        local.config.graph_summary_offset = offsetof(State, summary);
        local.config.runtime_worker_count = 3;
        local.config.worker_ids[0] = worker;
        local.config.self_lane = 0;
        context.worker_index = worker;
        ASSERT_TRUE(scheduler_protocol_bind(&state, &local));
    }
};

TEST(HbgGraphBootstrap, SharedDefinitionPublishesEveryInstanceBeforeReturn) {
    BootstrapFixture f;
    ASSERT_TRUE(f.execute());
    EXPECT_EQ(f.build.remaining_instances, 0u);
    EXPECT_EQ(f.build.materialized, 1u);
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
    EXPECT_EQ(f.build.materialized, 0u);
    EXPECT_FALSE(f.execute(2));
    EXPECT_EQ(f.build.public_completed, 0u);
}

TEST(HbgGraphBootstrap, FailedChildBindingKeepsItsParentAndLocalIdentity) {
    BootstrapFixture f;
    f.payloads[1].boundary[0].start_offset = UINT64_MAX;
    EXPECT_FALSE(f.execute());
    EXPECT_EQ(f.run.error_site, static_cast<uint64_t>(SchedulerErrorSite::GRAPH_MATERIALIZE_FAILED));
    EXPECT_EQ(f.run.error_task_ref, 5u);
    EXPECT_EQ(f.run.error_public_task_id, 1u);
    EXPECT_EQ(f.run.error_sub_task_id, 0u);
    EXPECT_EQ(f.payloads[0].control.materialize_state, 2u);
    EXPECT_EQ(f.payloads[1].control.materialize_state, 3u);
    EXPECT_EQ(f.build.remaining_instances, 1u);
    EXPECT_EQ(f.build.public_completed, 0u);
    EXPECT_FALSE(f.execute(2));
    EXPECT_EQ(f.run.error_sub_task_id, 0u);
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
    EXPECT_FALSE(returned.load(std::memory_order_acquire));
    scheduler_publish_signal(f.build.materialized, UINT64_C(1));
    waiter.join();
    EXPECT_TRUE(success);
    EXPECT_TRUE(returned.load(std::memory_order_acquire));
    EXPECT_EQ(f.payloads[1].output.scalars[0], 97u);
}

TEST(HbgGraphBootstrap, SignalDoesNotReplaceAtomicCounterConfirmation) {
    BootstrapFixture f;
    f.build.next_instance = 2;
    f.build.materialized = 1;
    EXPECT_FALSE(f.execute(1, PLATFORM_PROF_SYS_CNT_FREQ / 1000));
    EXPECT_EQ(scheduler_read_error(&f.run), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_EQ(f.build.remaining_instances, 2u);
    EXPECT_EQ(f.build.public_completed, 0u);
}

TEST(HbgGraphBootstrap, ZeroCounterDoesNotInventItsPublicationSignal) {
    BootstrapFixture f;
    f.build.next_instance = 2;
    f.build.remaining_instances = 0;
    EXPECT_FALSE(f.execute(1, PLATFORM_PROF_SYS_CNT_FREQ / 1000));
    EXPECT_EQ(f.build.materialized, 0u);
    EXPECT_EQ(scheduler_read_error(&f.run), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_EQ(f.build.public_completed, 0u);
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

TEST(HbgGraphBootstrap, ProtocolCapturesMaterializationPublicationAndObservedZeroInOneWorkerStream) {
    BootstrapFixture f;
    ASSERT_NO_FATAL_FAILURE(f.enable_protocol());
    ASSERT_TRUE(bootstrap_graph_instances<true>(f.graph, &f.context, &f.run, PLATFORM_PROF_SYS_CNT_FREQ, &f.local));
    for (uint32_t index = 0; index < 2; ++index) {
        const auto &record = f.state.graph_instances[index];
        EXPECT_EQ(record.instance_index, index);
        EXPECT_EQ(record.parent_id, index);
        EXPECT_EQ(record.definition_key, 17u);
        EXPECT_EQ(record.run_epoch, 7u);
        EXPECT_EQ(record.worker_index, 1u);
        EXPECT_EQ(record.child_count, 2u);
        EXPECT_EQ(record.status, 0u);
        EXPECT_EQ(record.materialize_before.sequence, index * 4 + 1);
        EXPECT_EQ(record.materialize_after.sequence, index * 4 + 2);
        EXPECT_EQ(record.publish_before.sequence, index * 4 + 3);
        EXPECT_EQ(record.publish_after.sequence, index * 4 + 4);
        EXPECT_EQ(record.remaining_before, 2 - index);
    }
    const auto &wait = f.state.graph_workers[1];
    EXPECT_EQ(wait.before.sequence, 9u);
    EXPECT_EQ(wait.after.sequence, 10u);
    EXPECT_EQ(wait.remaining_observed, 0u);
    EXPECT_EQ(wait.poll_count, 1u);
    EXPECT_EQ(wait.run_epoch, 7u);
    EXPECT_EQ(wait.status, 0u);
    EXPECT_EQ(f.state.workers[1].next_sequence, 10u);
    EXPECT_EQ(f.state.workers[1].error_flags, 0u);
}

TEST(HbgGraphBootstrap, L0DoesNotTouchProtocolRecordsEvenWhenStorageIsPresent) {
    BootstrapFixture f;
    ASSERT_NO_FATAL_FAILURE(f.enable_protocol());
    ASSERT_TRUE(f.execute());
    EXPECT_EQ(f.state.workers[1].next_sequence, 0u);
    EXPECT_EQ(f.state.graph_instances[0].materialize_before.sequence, 0u);
    EXPECT_EQ(f.state.graph_workers[1].before.sequence, 0u);
}

TEST(HbgGraphBootstrap, FailedMaterializationKeepsItsFailureWithoutInventingPublication) {
    BootstrapFixture f;
    ASSERT_NO_FATAL_FAILURE(f.enable_protocol());
    f.payloads[0].object.header.magic = 0;
    EXPECT_FALSE(bootstrap_graph_instances<true>(f.graph, &f.context, &f.run, PLATFORM_PROF_SYS_CNT_FREQ, &f.local));
    const auto &record = f.state.graph_instances[0];
    EXPECT_EQ(record.materialize_before.sequence, 1u);
    EXPECT_EQ(record.materialize_after.sequence, 2u);
    EXPECT_NE(record.status, 0u);
    EXPECT_EQ(record.publish_before.sequence, 0u);
    EXPECT_EQ(record.publish_after.sequence, 0u);
    EXPECT_EQ(f.state.graph_workers[1].after.sequence, 0u);
    EXPECT_EQ(f.build.remaining_instances, 2u);
}

TEST(HbgGraphBootstrap, AbandonedOwnerTraceIsABoundedWaitWithTheObservedCounter) {
    BootstrapFixture f;
    ASSERT_NO_FATAL_FAILURE(f.enable_protocol());
    f.build.next_instance = 2;
    EXPECT_FALSE(
        bootstrap_graph_instances<true>(f.graph, &f.context, &f.run, PLATFORM_PROF_SYS_CNT_FREQ / 1000, &f.local)
    );
    const auto &wait = f.state.graph_workers[1];
    EXPECT_EQ(wait.before.sequence, 1u);
    EXPECT_EQ(wait.after.sequence, 2u);
    EXPECT_EQ(wait.remaining_observed, 2u);
    EXPECT_GT(wait.poll_count, 0u);
    EXPECT_EQ(wait.status, static_cast<uint32_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_EQ(f.state.workers[1].next_sequence, 2u);
}

}  // namespace

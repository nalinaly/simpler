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
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler_graph_plan.h"

namespace {

using Kind = SchedulerGraphNodeKind;

struct GraphServices {
    explicit GraphServices(const std::vector<SchedulerGraphPlanShape> &shapes, bool protocol = false) :
        plan(*SchedulerGraphPlan::build(shapes)),
        storage(plan.node_count(), 3, 1, {3, 1, 0, 0}, protocol),
        tasks(plan.node_count()),
        nodes(plan.node_count()),
        deps(plan.node_count()),
        instances(plan.instance_count()),
        contributions(protocol ? plan.node_count() : 0) {
        for (int32_t ref = 0; ref < plan.node_count(); ++ref) {
            const auto node = *plan.node_at(ref);
            auto &mapped = nodes[ref];
            mapped.kind = node.kind;
            mapped.public_task_id = node.task_id;
            tasks[ref].task.task_id = node.task_id;
            if (node.kind == Kind::CHILD || node.kind == Kind::ORDINARY)
                mapped.storage_address = reinterpret_cast<uint64_t>(&tasks[ref]);
            if (node.instance_index >= 0) {
                auto &instance = instances[node.instance_index];
                mapped.instance_control_address = reinterpret_cast<uint64_t>(&instance);
                const auto range = plan.ranges()[node.outer_id];
                instance.enter_ref = range.begin;
                instance.child_begin = range.begin + 1;
                instance.child_count = range.children;
                instance.exit_ref = range.end - 1;
                instance.remaining_children = range.children;
                instance.instance_index = node.instance_index;
                if (node.kind == Kind::CHILD) dependency(ref, range.begin);
            }
            auto &metadata = storage.metadata[ref];
            metadata.active_mask = 2;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_INLINE;
            if (!deps[ref].empty()) metadata.flags |= SCHEDULER_TASK_HAS_FANIN;
            metadata.logical_block_num = 1;
            for (auto &kernel : metadata.kernel_ids)
                kernel = UINT16_MAX;
        }
        view = {0, 0, nodes.size(), nodes.size() - 1, reinterpret_cast<uint64_t>(nodes.data())};
        if (protocol) {
            build.instance_count = instances.size();
            build.public_task_count = plan.ranges().size();
            build.protocol_contributions_address = reinterpret_cast<uint64_t>(contributions.data());
            view.build_control_address = reinterpret_cast<uint64_t>(&build);
            EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 0));
        }
        for (uint32_t worker = 0; worker < 3; ++worker) {
            storage.test_graphs[worker] = view;
            locals[worker] = storage.local_context(&storage.contexts[worker]);
            if (protocol) EXPECT_TRUE(scheduler_protocol_bind(base(), locals[worker]));
        }
    }

    void dependency(int32_t consumer, int32_t producer) {
        deps[consumer].push_back(producer);
        nodes[consumer].fanin_count = static_cast<int32_t>(deps[consumer].size());
        nodes[consumer].fanin_address = reinterpret_cast<uint64_t>(deps[consumer].data());
        storage.metadata[consumer].flags |= SCHEDULER_TASK_HAS_FANIN;
    }
    void *base() const { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() const { return storage.run_control; }
    SchedulerTaskControl &control(int32_t ref) { return *scheduler_task_control_at(base(), locals[1], ref); }
    SchedulerRouteResult route(int32_t ref) {
        return scheduler_route_task(view, base(), locals[1], run(), ref, nullptr);
    }
    bool complete(int32_t ref, uint32_t worker = 1) {
        return scheduler_complete_inline_task(view, base(), locals[worker], run(), ref, 0);
    }
    int64_t pop(uint32_t worker = 1) {
        SchedulerReadyClaim claim{};
        EXPECT_TRUE(
            scheduler_claim_ready_for_slot(view, base(), locals[worker], run(), 3, 1, &cursor, nullptr, &claim)
        );
        return claim.task_ref;
    }

    SchedulerGraphPlan plan;
    scheduler_test::FixtureStorage storage;
    std::vector<ChipTaskStorage> tasks;
    std::vector<SchedulerGraphNode> nodes;
    std::vector<std::vector<int32_t>> deps;
    std::vector<SchedulerGraphInstanceControl> instances;
    std::vector<SchedulerGraphProtocolContribution> contributions;
    SchedulerGraphBuildControl build{};
    SchedulerGraphView view{};
    std::array<SchedulerLocalState *, 3> locals{};
    uint64_t cursor{0};
};

TEST(HbgGraphServices, ExitIsNotAZeroFaninBootstrapRoot) {
    GraphServices f({{true, 2}});
    EXPECT_EQ(
        scheduler_bootstrap_route_task(f.view, f.base(), f.locals[1], f.run(), 3, nullptr),
        SchedulerRouteResult::WAITING
    );
    EXPECT_EQ(f.route(3), SchedulerRouteResult::WAITING);
    EXPECT_EQ(scheduler_gm_query(f.control(3).state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
}

TEST(HbgGraphServices, ExternalGateAndAllChildrenPrecedeOneParentCompletion) {
    GraphServices f({{}, {true, 2}, {}});
    f.dependency(1, 0);
    f.dependency(5, 4);
    for (int32_t ref : {1, 2, 3, 5})
        EXPECT_EQ(f.route(ref), SchedulerRouteResult::WAITING);
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(f.complete(0));
    ASSERT_EQ(f.pop(), 1);
    ASSERT_TRUE(f.complete(1));
    std::array<int64_t, 2> roots{f.pop(), f.pop()};
    std::sort(roots.begin(), roots.end());
    EXPECT_EQ(roots, (std::array<int64_t, 2>{2, 3}));
    ASSERT_TRUE(f.complete(2));
    EXPECT_EQ(scheduler_gm_query(f.instances[0].remaining_children), 1u);
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
    EXPECT_NE(scheduler_gm_query(f.control(4).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    ASSERT_TRUE(f.complete(3));
    EXPECT_EQ(scheduler_gm_query(f.instances[0].remaining_children), 0u);
    ASSERT_EQ(f.pop(), 4);
    ASSERT_TRUE(f.complete(4));
    ASSERT_EQ(f.pop(), 5);
    ASSERT_TRUE(f.complete(5));
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(f.locals[1]->pending_completed, 6u);
    EXPECT_EQ(scheduler_read_error(f.run()), 0u);
}

TEST(HbgGraphServices, IndependentGraphFinishesWhileAnotherGraphHasAnUnfinishedChild) {
    GraphServices f({{true, 2}, {true, 2}});
    for (int32_t ref : {1, 2, 5, 6})
        EXPECT_EQ(f.route(ref), SchedulerRouteResult::WAITING);
    ASSERT_TRUE(f.complete(0));
    ASSERT_TRUE(f.complete(4));
    std::vector<int64_t> roots;
    for (int i = 0; i < 4; ++i)
        roots.push_back(f.pop());
    std::sort(roots.begin(), roots.end());
    EXPECT_EQ(roots, (std::vector<int64_t>{1, 2, 5, 6}));
    ASSERT_TRUE(f.complete(1));
    ASSERT_TRUE(f.complete(5));
    ASSERT_TRUE(f.complete(6));
    ASSERT_EQ(f.pop(), 7);
    ASSERT_TRUE(f.complete(7));
    EXPECT_EQ(scheduler_gm_query(f.instances[0].remaining_children), 1u);
    EXPECT_NE(scheduler_gm_query(f.control(3).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_query(f.control(7).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    ASSERT_TRUE(f.complete(2));
    EXPECT_EQ(f.pop(), 3);
}

TEST(HbgGraphServices, DuplicateChildCompletionDoesNotDecrementTheInstanceTwice) {
    GraphServices f({{true, 2}});
    ASSERT_TRUE(f.complete(1));
    ASSERT_EQ(scheduler_gm_query(f.instances[0].remaining_children), 1u);
    EXPECT_FALSE(f.complete(1));
    EXPECT_EQ(scheduler_gm_query(f.instances[0].remaining_children), 1u);
    EXPECT_NE(scheduler_read_error(f.run()), 0u);
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
}

TEST(HbgGraphServices, ProtocolContributionsFollowChildDoneAndPrecedeParentDone) {
    GraphServices f({{true, 2}}, true);
    ASSERT_TRUE(f.complete(1, 1));
    ASSERT_TRUE(f.complete(2, 2));
    for (uint32_t child = 1; child <= 2; ++child) {
        const auto &contribution = f.contributions[child];
        const auto *events =
            scheduler_state_at<SchedulerProtocolTaskTrace>(f.base(), f.storage.layout.protocol.tasks_offset);
        EXPECT_EQ(contribution.worker_index, child);
        EXPECT_EQ(contribution.instance_index, 0u);
        EXPECT_EQ(contribution.task_ref, child);
        EXPECT_EQ(contribution.remaining_before, 3 - child);
        EXPECT_GT(contribution.before.sequence, events[child].completion.after.sequence);
        EXPECT_GT(contribution.after.sequence, contribution.before.sequence);
    }
    ASSERT_EQ(f.pop(2), 3);
    ASSERT_TRUE(f.complete(3, 2));
    const auto *events =
        scheduler_state_at<SchedulerProtocolTaskTrace>(f.base(), f.storage.layout.protocol.tasks_offset);
    EXPECT_GT(events[3].completion.before.sequence, f.contributions[2].after.sequence);
    EXPECT_EQ(f.build.public_completed, 1u);
    EXPECT_EQ(f.contributions[0].before.sequence, 0u);
    EXPECT_EQ(f.contributions[3].before.sequence, 0u);
    const auto *workers =
        scheduler_state_at<SchedulerProtocolWorkerTrace>(f.base(), f.storage.layout.protocol.workers_offset);
    EXPECT_EQ(workers[1].error_flags, 0u);
    EXPECT_EQ(workers[2].error_flags, 0u);
    const uint64_t sequence = workers[2].next_sequence;
    EXPECT_FALSE(f.complete(2, 2));
    EXPECT_EQ(workers[2].next_sequence, sequence);
    EXPECT_EQ(f.instances[0].remaining_children, 0u);
}

TEST(HbgGraphServices, ProtocolRejectsAContributionCompletedByAnotherWorker) {
    GraphServices f({{true, 2}}, true);
    auto *record = scheduler_graph_protocol_contribution_begin(f.base(), f.locals[1], f.view, &f.instances[0], 1);
    ASSERT_NE(record, nullptr);
    scheduler_graph_protocol_contribution_end(f.base(), f.locals[2], record, 2);
    EXPECT_EQ(record->after.sequence, 0u);
    const auto *workers =
        scheduler_state_at<SchedulerProtocolWorkerTrace>(f.base(), f.storage.layout.protocol.workers_offset);
    EXPECT_EQ(workers[1].next_sequence, 1u);
    EXPECT_EQ(workers[2].next_sequence, 0u);
    EXPECT_EQ(workers[2].error_flags, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
}

TEST(HbgGraphServices, OldInstanceEpochCannotContributeToTheCurrentRun) {
    GraphServices f({{true, 2}});
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(f.base(), f.storage.layout.graph_summary_offset);
    summary->run_epoch = 7;
    f.instances[0].run_epoch = 6;
    EXPECT_FALSE(f.complete(1));
    EXPECT_EQ(f.instances[0].remaining_children, 2u);
    EXPECT_NE(scheduler_read_error(f.run()), 0u);
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
}

TEST(HbgGraphServices, AggregateUnderflowIsAnErrorAndNeverReleasesTheParent) {
    GraphServices f({{true, 2}});
    scheduler_gm_store(f.instances[0].remaining_children, UINT64_C(0));
    EXPECT_FALSE(f.complete(1));
    EXPECT_NE(scheduler_read_error(f.run()), 0u);
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
    EXPECT_NE(scheduler_gm_query(f.control(3).state), static_cast<int64_t>(SchedulerTaskState::DONE));
}

TEST(HbgGraphServices, MaximumBodyDoesNotUseAnExitFaninRow) {
    GraphServices f({{true, SUB_TASK_MAX_NUM}});
    ASSERT_TRUE(f.complete(0));
    for (int32_t child = 1; child < SUB_TASK_MAX_NUM; ++child)
        ASSERT_TRUE(f.complete(child));
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(scheduler_gm_query(f.instances[0].remaining_children), 1u);
    ASSERT_TRUE(f.complete(SUB_TASK_MAX_NUM));
    const int32_t exit = SUB_TASK_MAX_NUM + 1;
    EXPECT_EQ(scheduler_graph_fanin_count(f.view, exit), 0);
    EXPECT_EQ(f.pop(), exit);
    ASSERT_TRUE(f.complete(exit));
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(f.locals[1]->pending_completed, static_cast<uint64_t>(SUB_TASK_MAX_NUM + 2));
}

TEST(HbgGraphServices, ConcurrentLastChildrenPublishExactlyOneExit) {
    for (int iteration = 0; iteration < 32; ++iteration) {
        GraphServices f({{true, 2}});
        ASSERT_TRUE(f.complete(0));
        std::atomic<uint32_t> arrived{0};
        std::array<bool, 2> succeeded{};
        auto finish = [&](uint32_t index) {
            arrived.fetch_add(1, std::memory_order_acq_rel);
            while (arrived.load(std::memory_order_acquire) != 2) {}
            succeeded[index] = f.complete(static_cast<int32_t>(index + 1), index + 1);
        };
        std::thread first(finish, 0);
        std::thread second(finish, 1);
        first.join();
        second.join();
        ASSERT_TRUE(succeeded[0]);
        ASSERT_TRUE(succeeded[1]);
        EXPECT_EQ(scheduler_gm_query(f.instances[0].remaining_children), 0u);
        EXPECT_EQ(f.pop(), 3);
        EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(scheduler_read_error(f.run()), 0u);
    }
}

TEST(HbgGraphServices, AChildFromAnotherInstanceCannotFinishThisParent) {
    GraphServices f({{true, 2}, {true, 2}});
    f.nodes[1].instance_control_address = reinterpret_cast<uint64_t>(&f.instances[1]);
    EXPECT_FALSE(f.complete(1));
    EXPECT_EQ(scheduler_gm_query(f.instances[0].remaining_children), 2u);
    EXPECT_EQ(scheduler_gm_query(f.instances[1].remaining_children), 2u);
    EXPECT_EQ(
        scheduler_gm_query(f.run()->error_site), static_cast<uint64_t>(SchedulerErrorSite::GRAPH_INSTANCE_INVALID)
    );
}

TEST(HbgGraphServices, RecordedDiamondMovesWaitersAcrossBothInternalParents) {
    GraphServices f({{true, 4}});
    const std::array<uint16_t, 4> fanins{0, 0, 1, 2};
    for (int32_t ref = 2; ref <= 4; ++ref) {
        auto &node = f.nodes[ref];
        node.fanin_kind = SchedulerGraphFaninKind::DEFINITION_CSR;
        node.fanin_ref_base = 1;
        node.fanin_address = reinterpret_cast<uint64_t>(fanins.data() + ref - 2);
        node.fanin_count = ref == 4 ? 2 : 1;
    }
    for (int32_t ref = 1; ref <= 4; ++ref)
        EXPECT_EQ(f.route(ref), SchedulerRouteResult::WAITING);
    ASSERT_TRUE(f.complete(0));
    ASSERT_EQ(f.pop(), 1);
    ASSERT_TRUE(f.complete(1));
    std::array<int64_t, 2> middle{f.pop(), f.pop()};
    std::sort(middle.begin(), middle.end());
    EXPECT_EQ(middle, (std::array<int64_t, 2>{2, 3}));
    ASSERT_TRUE(f.complete(2));
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(f.control(4).waiting_producer, 3);
    ASSERT_TRUE(f.complete(3));
    ASSERT_EQ(f.pop(), 4);
    ASSERT_TRUE(f.complete(4));
    EXPECT_EQ(scheduler_gm_query(f.instances[0].remaining_children), 0u);
    ASSERT_EQ(f.pop(), 5);
    ASSERT_TRUE(f.complete(5));
    EXPECT_EQ(f.pop(), SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(f.locals[1]->pending_completed, 6u);
}

TEST(HbgGraphServices, PublicCompletionCountsOnlyOrdinaryTasksAndGraphExit) {
    GraphServices f({{}, {true, 2}, {}});
    SchedulerGraphBuildControl build{};
    build.public_task_count = 3;
    f.view.build_control_address = reinterpret_cast<uint64_t>(&build);
    ASSERT_TRUE(f.complete(0));
    EXPECT_EQ(build.public_completed, 1u);
    ASSERT_TRUE(f.complete(1));
    ASSERT_TRUE(f.complete(2));
    EXPECT_EQ(build.public_completed, 1u);
    ASSERT_TRUE(f.complete(3));
    EXPECT_EQ(build.public_completed, 1u);
    ASSERT_EQ(f.pop(), 4);
    ASSERT_TRUE(f.complete(4));
    EXPECT_EQ(build.public_completed, 2u);
    ASSERT_TRUE(f.complete(5));
    EXPECT_EQ(build.public_completed, 3u);
    EXPECT_EQ(f.locals[1]->pending_completed, 6u);
}

}  // namespace

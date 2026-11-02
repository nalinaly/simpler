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
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "scheduler/scheduler_graph_work_state.h"

namespace {

using Completion = SchedulerGraphWorkCompletion;

TEST(HbgGraphWorkState, ExactlyOneLastContributionAcrossCompletionOrders) {
    for (uint32_t seed = 0; seed < 32; ++seed) {
        SCOPED_TRACE(seed);
        SchedulerGraphWorkGroup group{17, 10, 17};
        std::array<SchedulerGraphWorkState, 17> states{};
        std::array<SchedulerGraphWorkToken, 17> tokens{};
        std::array<uint32_t, 17> order{};
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), std::mt19937(seed));
        for (uint32_t index = 0; index < states.size(); ++index)
            ASSERT_TRUE(scheduler_graph_work_claim(group, states[index], index + 10, index % 5, 7, &tokens[index]));
        for (uint32_t position = 0; position < order.size(); ++position) {
            const uint32_t index = order[position];
            uint64_t prior = 0;
            const auto result = scheduler_graph_work_complete(group, states[index], tokens[index], 7, &prior);
            EXPECT_EQ(prior, order.size() - position);
            EXPECT_EQ(result, position + 1 == order.size() ? Completion::LAST : Completion::COMPLETED);
            EXPECT_EQ(states[index].owner_state, (tokens[index].owner << 2) | 2u);
            EXPECT_EQ(group.remaining, order.size() - position - 1);
        }
    }
}

TEST(HbgGraphWorkState, DuplicateClaimsAndCompletionsCannotDrainTheCounter) {
    SchedulerGraphWorkGroup group{2, 4, 2};
    SchedulerGraphWorkState state{};
    SchedulerGraphWorkToken owner{}, duplicate{};
    ASSERT_TRUE(scheduler_graph_work_claim(group, state, 4, 2, 7, &owner));
    EXPECT_FALSE(scheduler_graph_work_claim(group, state, 4, 3, 7, &duplicate));
    EXPECT_EQ(duplicate.owner, 0u);
    EXPECT_EQ(scheduler_graph_work_complete(group, state, owner, 7, nullptr), Completion::COMPLETED);
    for (int repeat = 0; repeat < 8; ++repeat) {
        EXPECT_EQ(scheduler_graph_work_complete(group, state, owner, 7, nullptr), Completion::REJECTED);
        EXPECT_FALSE(scheduler_graph_work_fail(group, state, owner, 7));
        EXPECT_EQ(group.remaining, 1u);
    }
}

TEST(HbgGraphWorkState, InvalidRangeAndOwnerCannotClaimOrContribute) {
    SchedulerGraphWorkGroup group{1, 9, 1};
    SchedulerGraphWorkState state{};
    SchedulerGraphWorkToken token{};
    EXPECT_FALSE(scheduler_graph_work_claim(group, state, 8, 0, 7, &token));
    EXPECT_FALSE(scheduler_graph_work_claim(group, state, 10, 0, 7, &token));
    EXPECT_FALSE(scheduler_graph_work_claim(group, state, UINT64_MAX, 0, 7, &token));
    EXPECT_FALSE(scheduler_graph_work_claim(group, state, 9, UINT32_MAX, 7, &token));
    EXPECT_FALSE(scheduler_graph_work_claim(group, state, 9, 0, 7, nullptr));
    EXPECT_EQ(state.owner_state, 0u);
    ASSERT_TRUE(scheduler_graph_work_claim(group, state, 9, 1, 7, &token));
    auto forged = token;
    forged.owner = 1;
    EXPECT_EQ(scheduler_graph_work_complete(group, state, forged, 7, nullptr), Completion::REJECTED);
    EXPECT_FALSE(scheduler_graph_work_fail(group, state, forged, 7));
    forged = token;
    forged.index = 10;
    EXPECT_EQ(scheduler_graph_work_complete(group, state, forged, 7, nullptr), Completion::REJECTED);
    EXPECT_EQ(group.remaining, 1u);
    EXPECT_EQ(scheduler_graph_work_complete(group, state, token, 7, nullptr), Completion::LAST);
}

TEST(HbgGraphWorkState, OldEpochCannotFinishReusedWorkOwnedByTheSameWorker) {
    SchedulerGraphWorkGroup group{1, 0, 1};
    SchedulerGraphWorkState state{};
    SchedulerGraphWorkToken old{}, current{};
    ASSERT_TRUE(scheduler_graph_work_claim(group, state, 0, 3, 7, &old));
    ASSERT_EQ(scheduler_graph_work_complete(group, state, old, 7, nullptr), Completion::LAST);
    group.remaining = 1;
    state.owner_state = 0;
    ASSERT_TRUE(scheduler_graph_work_claim(group, state, 0, 3, 8, &current));
    EXPECT_EQ(scheduler_graph_work_complete(group, state, old, 8, nullptr), Completion::REJECTED);
    EXPECT_FALSE(scheduler_graph_work_fail(group, state, old, 8));
    EXPECT_EQ(group.remaining, 1u);
    EXPECT_EQ(state.owner_state, (current.owner << 2) | 1u);
    EXPECT_EQ(scheduler_graph_work_complete(group, state, current, 8, nullptr), Completion::LAST);
}

TEST(HbgGraphWorkState, FailedWorkPreventsLastSuccessAndCannotBeCompletedLater) {
    SchedulerGraphWorkGroup group{3, 0, 3};
    std::array<SchedulerGraphWorkState, 3> states{};
    std::array<SchedulerGraphWorkToken, 3> tokens{};
    for (uint32_t index = 0; index < 3; ++index)
        ASSERT_TRUE(scheduler_graph_work_claim(group, states[index], index, index, 7, &tokens[index]));
    ASSERT_TRUE(scheduler_graph_work_fail(group, states[1], tokens[1], 7));
    EXPECT_EQ(scheduler_graph_work_complete(group, states[0], tokens[0], 7, nullptr), Completion::COMPLETED);
    EXPECT_EQ(scheduler_graph_work_complete(group, states[2], tokens[2], 7, nullptr), Completion::COMPLETED);
    EXPECT_EQ(scheduler_graph_work_complete(group, states[1], tokens[1], 7, nullptr), Completion::REJECTED);
    EXPECT_FALSE(scheduler_graph_work_fail(group, states[1], tokens[1], 7));
    EXPECT_EQ(group.remaining, 1u);
    EXPECT_EQ(states[1].owner_state, (tokens[1].owner << 2) | 3u);
}

TEST(HbgGraphWorkState, ATokenCannotCompleteAnotherClaimByTheSameWorker) {
    SchedulerGraphWorkGroup group{2, 0, 2};
    std::array<SchedulerGraphWorkState, 2> states{};
    std::array<SchedulerGraphWorkToken, 2> tokens{};
    for (uint32_t index = 0; index < 2; ++index)
        ASSERT_TRUE(scheduler_graph_work_claim(group, states[index], index, 0, 7, &tokens[index]));
    EXPECT_EQ(scheduler_graph_work_complete(group, states[1], tokens[0], 7, nullptr), Completion::REJECTED);
    EXPECT_FALSE(scheduler_graph_work_fail(group, states[1], tokens[0], 7));
    EXPECT_EQ(group.remaining, 2u);
    EXPECT_EQ(scheduler_graph_work_complete(group, states[0], tokens[0], 7, nullptr), Completion::COMPLETED);
    EXPECT_EQ(scheduler_graph_work_complete(group, states[1], tokens[1], 7, nullptr), Completion::LAST);
}

TEST(HbgGraphWorkState, ConcurrentWritersCannotPublishWhileOneClaimIsOutstanding) {
    SchedulerGraphWorkGroup group{257, 0, 257};
    std::array<SchedulerGraphWorkState, 257> states{};
    SchedulerGraphWorkToken held{};
    ASSERT_TRUE(scheduler_graph_work_claim(group, states[0], 0, 0, 7, &held));
    std::atomic<uint32_t> next{1};
    std::atomic<uint32_t> rejected{0}, last{0};
    std::vector<std::thread> workers;
    for (uint32_t worker = 1; worker <= 8; ++worker) {
        workers.emplace_back([&, worker] {
            for (;;) {
                const uint32_t index = next.fetch_add(1);
                if (index >= states.size()) break;
                SchedulerGraphWorkToken token{};
                if (!scheduler_graph_work_claim(group, states[index], index, worker, 7, &token)) {
                    ++rejected;
                    continue;
                }
                const auto result = scheduler_graph_work_complete(group, states[index], token, 7, nullptr);
                if (result == Completion::LAST) ++last;
                if (result == Completion::REJECTED) ++rejected;
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(rejected.load(), 0u);
    EXPECT_EQ(last.load(), 0u);
    EXPECT_EQ(group.remaining, 1u);
    EXPECT_EQ(scheduler_graph_work_complete(group, states[0], held, 7, nullptr), Completion::LAST);
    EXPECT_EQ(group.remaining, 0u);
}

TEST(HbgGraphWorkState, PublicationIsUniqueAndPreservesChildCompletionAccounting) {
    SchedulerGraphInstanceControl instance{};
    instance.run_epoch = 7;
    instance.materialize_state = 1;
    instance.remaining_children = 1024;
    instance.work = {0, 0, 3};
    SchedulerGraphBuildControl build{};
    build.run_epoch = 7;
    build.instance_count = build.remaining_instances = 2;
    const SchedulerGraphWorkToken last{2, 7, 4};
    uint64_t prior = 0;
    ASSERT_TRUE(scheduler_graph_work_publish(instance, build, last, &prior));
    EXPECT_EQ(prior, 2u);
    EXPECT_EQ(instance.materialize_state, 2u);
    EXPECT_EQ(instance.materialize_owner, 4u);
    EXPECT_EQ(instance.remaining_children, 1024u);
    EXPECT_EQ(build.remaining_instances, 1u);
    EXPECT_FALSE(scheduler_graph_work_publish(instance, build, last, &prior));
    EXPECT_EQ(build.remaining_instances, 1u);
}

TEST(HbgGraphWorkState, FailedOrIncompleteInstanceCannotPublishSuccess) {
    for (bool failed : {false, true}) {
        SchedulerGraphInstanceControl instance{};
        instance.run_epoch = 7;
        instance.materialize_state = failed ? 3 : 1;
        instance.work = {failed ? 0u : 1u, 0, 3};
        SchedulerGraphBuildControl build{};
        build.run_epoch = 7;
        build.instance_count = build.remaining_instances = 1;
        EXPECT_FALSE(scheduler_graph_work_publish(instance, build, {2, 7, 1}, nullptr));
        EXPECT_EQ(instance.materialize_state, failed ? 3u : 1u);
        EXPECT_EQ(build.remaining_instances, 1u);
        EXPECT_EQ(build.materialized, 0u);
    }
}

TEST(HbgGraphWorkState, StalePublicationCannotClaimTheNewInstanceOwner) {
    SchedulerGraphInstanceControl instance{};
    instance.run_epoch = 8;
    instance.materialize_state = 1;
    instance.work = {0, 0, 1};
    SchedulerGraphBuildControl build{};
    build.run_epoch = 8;
    build.instance_count = build.remaining_instances = 1;
    EXPECT_FALSE(scheduler_graph_work_publish(instance, build, {0, 7, 1}, nullptr));
    EXPECT_EQ(instance.materialize_owner, 0u);
    EXPECT_EQ(instance.materialize_state, 1u);
    EXPECT_EQ(build.remaining_instances, 1u);
}

}  // namespace

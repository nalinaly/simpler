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

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "utils/thread_completion_gate.h"

TEST(ThreadCompletionGateTest, CleanupCannotBeClaimedWhileFinalizerIsRunning) {
    simpler::ThreadCompletionGate gate;
    gate.arrive_and_finalize_if_last(2, [] {});

    std::mutex mutex;
    std::condition_variable condition;
    bool finalizer_started = false;
    bool allow_finalizer_to_finish = false;
    bool waiter_returned = false;

    std::thread last_thread([&] {
        gate.arrive_and_finalize_if_last(2, [&] {
            std::unique_lock<std::mutex> lock(mutex);
            finalizer_started = true;
            condition.notify_one();
            condition.wait(lock, [&] {
                return allow_finalizer_to_finish;
            });
        });
    });

    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] {
            return finalizer_started;
        });
    }
    std::thread waiter([&] {
        gate.wait_for_finalization();
        {
            std::lock_guard<std::mutex> lock(mutex);
            waiter_returned = true;
        }
        condition.notify_one();
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_FALSE(waiter_returned);
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        allow_finalizer_to_finish = true;
    }
    condition.notify_one();
    last_thread.join();
    waiter.join();

    EXPECT_FALSE(gate.depart_and_claim_cleanup_if_last(2));
    EXPECT_TRUE(gate.depart_and_claim_cleanup_if_last(2));
    EXPECT_FALSE(gate.depart_and_claim_cleanup_if_last(2));
}

TEST(ThreadCompletionGateTest, ResetAllowsAnotherRun) {
    simpler::ThreadCompletionGate gate;
    int finalized = 0;

    gate.arrive_and_finalize_if_last(1, [&] {
        ++finalized;
    });
    gate.wait_for_finalization();
    ASSERT_TRUE(gate.depart_and_claim_cleanup_if_last(1));

    gate.reset();
    gate.arrive_and_finalize_if_last(1, [&] {
        ++finalized;
    });
    gate.wait_for_finalization();
    EXPECT_TRUE(gate.depart_and_claim_cleanup_if_last(1));
    EXPECT_EQ(finalized, 2);
}

TEST(ThreadCompletionGateTest, EveryParticipantSnapshotsFailureBeforeCleanupResetsGeneration) {
    constexpr int participant_count = 4;
    simpler::ThreadCompletionGate gate;
    std::atomic<int> shared_error{-17};
    std::atomic<int> finalizer_count{0};
    std::atomic<int> cleanup_count{0};
    std::atomic<int> observed_failure_count{0};
    std::vector<std::thread> participants;

    for (int i = 0; i < participant_count; ++i) {
        participants.emplace_back([&] {
            gate.arrive_and_finalize_if_last(participant_count, [&] {
                finalizer_count.fetch_add(1, std::memory_order_relaxed);
            });
            gate.wait_for_finalization();
            if (shared_error.load(std::memory_order_acquire) == -17) {
                observed_failure_count.fetch_add(1, std::memory_order_relaxed);
            }
            if (gate.depart_and_claim_cleanup_if_last(participant_count)) {
                cleanup_count.fetch_add(1, std::memory_order_relaxed);
                shared_error.store(0, std::memory_order_release);
            }
        });
    }
    for (auto &participant : participants) {
        participant.join();
    }

    EXPECT_EQ(finalizer_count.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(cleanup_count.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(observed_failure_count.load(std::memory_order_relaxed), participant_count);
    EXPECT_EQ(shared_error.load(std::memory_order_acquire), 0);
}

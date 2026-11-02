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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_ordinary_pool.h"

namespace {

using Result = SchedulerOrdinaryPoolResult;
using Phase = SchedulerOrdinaryPoolPhase;
using Token = SchedulerOrdinaryPoolReservation;

struct PoolFixture {
    explicit PoolFixture(uint64_t tasks = 1, uint16_t blocks = 3, LaneMask mask = LaneMask::aic()) :
        words((tasks + 15) / 16),
        controls(tasks),
        metadata(tasks) {
        for (auto &item : metadata) {
            item.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD;
            item.active_mask = mask;
            item.logical_block_num = blocks;
        }
        view = {words.data(), controls.data(), metadata.data(), tasks, 19, mask.has_aic() ? 0U : 1U};
    }

    std::vector<SchedulerOrdinaryPoolWord> words;
    std::vector<SchedulerTaskControl> controls;
    std::vector<SchedulerTaskMetadata> metadata;
    SchedulerOrdinaryPoolView view;
};

TEST(HbgOrdinaryPool, OnlyOrdinarySingleLaneMultiblockShapesQualify) {
    const std::array<LaneMask, 7> masks{
        LaneMask::aic(),
        LaneMask::aiv0(),
        LaneMask::aiv1(),
        LaneMask::both_aiv(),
        LaneMask::all(),
        LaneMask{true, true, false},
        LaneMask{true, false, true}
    };
    for (const auto mask : masks) {
        for (const uint32_t blocks : {0U, 1U, 2U, 32767U, 32768U, UINT32_MAX}) {
            constexpr uint8_t flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD;
            const bool expected = mask.count() == 1 && blocks > 1 && blocks <= 32767;
            EXPECT_EQ(scheduler_ordinary_pool_shape(flags, mask, blocks), expected);
            EXPECT_FALSE(scheduler_ordinary_pool_shape(flags | SCHEDULER_TASK_HAS_PREDICATE, mask, blocks));
            EXPECT_FALSE(scheduler_ordinary_pool_shape(flags | SCHEDULER_TASK_SYNC_START, mask, blocks));
            EXPECT_FALSE(scheduler_ordinary_pool_shape(flags | SCHEDULER_TASK_MIX, mask, blocks));
            EXPECT_FALSE(scheduler_ordinary_pool_shape(flags | SCHEDULER_TASK_INLINE, mask, blocks));
            EXPECT_FALSE(scheduler_ordinary_pool_shape(flags | 128, mask, blocks));
        }
    }
    EXPECT_FALSE(scheduler_ordinary_pool_shape(SCHEDULER_TASK_SPMD, LaneMask::aic(), 2));
    EXPECT_FALSE(scheduler_ordinary_pool_shape(SCHEDULER_TASK_EXECUTABLE, LaneMask::aic(), 2));
    EXPECT_FALSE(scheduler_ordinary_pool_shape(9, LaneMask::none(), 2));
    for (uint8_t raw : {8, 16, 32, 64, 128}) {
        LaneMask invalid{};
        std::memcpy(&invalid, &raw, sizeof(raw));
        EXPECT_FALSE(scheduler_ordinary_pool_shape(9, invalid, 2));
    }
}

TEST(HbgOrdinaryPool, PublicationIsUniqueAndCannotCarryIntoAnotherTaskBit) {
    PoolFixture f(17);
    for (const int64_t task : {0, 1, 15, 16})
        ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, task));
    const uint64_t expected = UINT64_C(3) | (UINT64_C(1) << 15);
    ASSERT_EQ(f.words[0].bits, expected);
    ASSERT_EQ(f.words[1].bits, 1U);
    for (const int64_t task : {0, 1, 15, 16}) {
        EXPECT_FALSE(scheduler_ordinary_pool_publish(f.view, task));
        EXPECT_EQ(f.controls[task].admission_claim_lock, SCHEDULER_ORDINARY_POOL_OWNER);
    }
    EXPECT_EQ(f.words[0].bits, expected);
    EXPECT_EQ(f.words[1].bits, 1U);
}

TEST(HbgOrdinaryPool, SixteenTaskWordsPublishAndRetireIndependentAtomicRegions) {
    PoolFixture f(33, 2);
    ASSERT_EQ(f.words.size(), 3U);
    for (int64_t task : {0, 15, 16, 32})
        ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, task));
    ASSERT_EQ(f.words[0].bits, UINT64_C(1) | (UINT64_C(1) << 15));
    ASSERT_EQ(f.words[1].bits, 1U);
    ASSERT_EQ(f.words[2].bits, 1U);
    for (int64_t task : {16, 0, 32, 15}) {
        for (uint32_t block = 0; block < 2; ++block) {
            Token token{};
            uint64_t previous = UINT64_MAX;
            ASSERT_EQ(scheduler_ordinary_pool_reserve(f.view, task, &token), Result::RESERVED);
            ASSERT_EQ(scheduler_ordinary_pool_confirm(f.view, &token, &previous), Result::CONFIRMED);
            ASSERT_EQ(
                scheduler_ordinary_pool_finish(f.view, &token, &previous),
                block == 0 ? Result::COMPLETED : Result::TASK_LAST
            );
        }
        EXPECT_EQ(f.words[task / 16].bits & (UINT64_C(1) << (task % 16)), 0U);
    }
    for (const auto &word : f.words)
        EXPECT_EQ(word.bits, 0U);
}

TEST(HbgOrdinaryPool, NoTicketBeforePublicationAndNoRoleSubstitution) {
    PoolFixture f;
    Token token{};
    EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &token), Result::EMPTY);
    EXPECT_EQ(token.phase, Phase::EMPTY);
    EXPECT_EQ(f.controls[0].claim_tickets, 0U);
    auto vector_view = f.view;
    vector_view.core_type = 1;
    EXPECT_FALSE(scheduler_ordinary_pool_publish(vector_view, 0));
    EXPECT_EQ(scheduler_ordinary_pool_reserve(vector_view, 0, &token), Result::ERROR);
    EXPECT_EQ(f.words[0].bits, 0U);
}

TEST(HbgOrdinaryPool, LastTicketIsNeitherClaimConfirmationNorTaskCompletion) {
    PoolFixture f;
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 0));
    std::array<Token, 3> tokens{};
    for (auto &token : tokens)
        ASSERT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &token), Result::RESERVED);
    EXPECT_EQ(f.controls[0].claim_tickets, 3U);
    EXPECT_EQ(f.controls[0].next_block, 0U);
    EXPECT_EQ(f.controls[0].completed_blocks, 0U);
    EXPECT_EQ(f.words[0].bits, 1U);
    Token exhausted{};
    for (unsigned attempt = 0; attempt < 100; ++attempt)
        EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &exhausted), Result::EMPTY);
    EXPECT_EQ(f.controls[0].claim_tickets, 3U);
    uint64_t previous = UINT64_MAX;
    // Hold the lowest ticket while the two later reservations finish.
    for (unsigned index : {2U, 1U}) {
        ASSERT_EQ(scheduler_ordinary_pool_confirm(f.view, &tokens[index], &previous), Result::CONFIRMED);
        ASSERT_EQ(scheduler_ordinary_pool_finish(f.view, &tokens[index], &previous), Result::COMPLETED);
    }
    EXPECT_EQ(f.words[0].bits, 1U);
    EXPECT_EQ(f.controls[0].completed_blocks, 2U);
    ASSERT_EQ(scheduler_ordinary_pool_confirm(f.view, &tokens[0], &previous), Result::CONFIRMED);
    EXPECT_EQ(previous, 2U);
    EXPECT_EQ(f.words[0].bits, 0U);
    EXPECT_EQ(f.controls[0].completed_blocks, 2U);
    EXPECT_EQ(scheduler_ordinary_pool_finish(f.view, &tokens[0], &previous), Result::TASK_LAST);
    EXPECT_EQ(previous, 2U);
    // TaskDone/dependency resolution belongs to the caller, exactly once.
    EXPECT_EQ(f.controls[0].state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
}

TEST(HbgOrdinaryPool, DuplicateAndOutOfOrderLocalTransitionsDoNotIncrementCounters) {
    PoolFixture f;
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 0));
    Token token{};
    uint64_t previous = UINT64_MAX;
    EXPECT_EQ(scheduler_ordinary_pool_confirm(f.view, &token, &previous), Result::ERROR);
    ASSERT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &token), Result::RESERVED);
    EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &token), Result::ERROR);
    EXPECT_EQ(scheduler_ordinary_pool_finish(f.view, &token, &previous), Result::ERROR);
    EXPECT_EQ(f.controls[0].claim_tickets, 1U);
    EXPECT_EQ(f.controls[0].next_block, 0U);
    EXPECT_EQ(f.controls[0].completed_blocks, 0U);
    ASSERT_EQ(scheduler_ordinary_pool_confirm(f.view, &token, &previous), Result::CONFIRMED);
    EXPECT_EQ(scheduler_ordinary_pool_confirm(f.view, &token, &previous), Result::ERROR);
    ASSERT_EQ(scheduler_ordinary_pool_finish(f.view, &token, &previous), Result::COMPLETED);
    EXPECT_EQ(scheduler_ordinary_pool_finish(f.view, &token, &previous), Result::ERROR);
    EXPECT_EQ(f.controls[0].next_block, 1U);
    EXPECT_EQ(f.controls[0].completed_blocks, 1U);
}

TEST(HbgOrdinaryPool, ReservationKeepsItsIdentityAcrossUnrelatedWorkAndRejectsAnotherEpoch) {
    PoolFixture f(2);
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 0));
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 1));
    Token held{}, other{};
    uint64_t previous = 0;
    ASSERT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &held), Result::RESERVED);
    ASSERT_EQ(scheduler_ordinary_pool_confirm(f.view, &held, &previous), Result::CONFIRMED);
    ASSERT_EQ(scheduler_ordinary_pool_reserve(f.view, 1, &other), Result::RESERVED);
    ASSERT_EQ(scheduler_ordinary_pool_confirm(f.view, &other, &previous), Result::CONFIRMED);
    ASSERT_EQ(scheduler_ordinary_pool_finish(f.view, &other, &previous), Result::COMPLETED);
    auto next_run = f.view;
    ++next_run.run_epoch;
    EXPECT_EQ(scheduler_ordinary_pool_finish(next_run, &held, &previous), Result::ERROR);
    EXPECT_EQ(f.controls[0].completed_blocks, 0U);
    EXPECT_EQ(held.task_ref, 0);
    EXPECT_EQ(held.block_idx, 0U);
    EXPECT_EQ(held.run_epoch, f.view.run_epoch);
    EXPECT_EQ(scheduler_ordinary_pool_finish(f.view, &held, &previous), Result::COMPLETED);
    EXPECT_EQ(f.controls[0].claim_tickets, 1U);
}

TEST(HbgOrdinaryPool, SparseRoundRobinDiscoveryVisitsEveryWordAndLatePublication) {
    PoolFixture f(137);
    for (int64_t task : {0, 63, 64, 136})
        ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, task));
    uint64_t cursor = 1;
    std::vector<int64_t> observed;
    for (unsigned calls = 0; calls < 2 * (f.words.size() + 4) && observed.size() < 6; ++calls) {
        int64_t task = -1;
        auto result = scheduler_ordinary_pool_discover(f.view, &cursor, &task);
        ASSERT_NE(result, Result::ERROR);
        if (result == Result::FOUND) observed.push_back(task);
    }
    ASSERT_EQ(observed, (std::vector<int64_t>{63, 64, 136, 0, 63, 64}));
    PoolFixture late(137);
    cursor = 64;
    int64_t task = -1;
    EXPECT_EQ(scheduler_ordinary_pool_discover(late.view, &cursor, &task), Result::EMPTY);
    ASSERT_TRUE(scheduler_ordinary_pool_publish(late.view, 7));
    Result found = Result::EMPTY;
    for (unsigned calls = 0; calls <= late.words.size() && found == Result::EMPTY; ++calls)
        found = scheduler_ordinary_pool_discover(late.view, &cursor, &task);
    EXPECT_EQ(found, Result::FOUND);
    EXPECT_EQ(task, 7);
}

TEST(HbgOrdinaryPool, AllSameTypeWorkersCanReserveBothVectorSubtaskKinds) {
    for (const auto mask : {LaneMask::aic(), LaneMask::aiv0(), LaneMask::aiv1()}) {
        PoolFixture f(1, 96, mask);
        ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 0));
        std::array<Token, 96> tokens{};
        for (auto &token : tokens) {
            auto worker_view = f.view;
            ASSERT_EQ(scheduler_ordinary_pool_reserve(worker_view, 0, &token), Result::RESERVED);
        }
        uint64_t previous = 0;
        for (uint32_t worker = 0; worker < tokens.size(); ++worker) {
            EXPECT_EQ(tokens[worker].block_idx, worker);
            EXPECT_EQ(scheduler_ordinary_pool_confirm(f.view, &tokens[worker], &previous), Result::CONFIRMED);
            EXPECT_EQ(
                scheduler_ordinary_pool_finish(f.view, &tokens[worker], &previous),
                worker + 1 == tokens.size() ? Result::TASK_LAST : Result::COMPLETED
            );
        }
        EXPECT_EQ(f.words[0].bits, 0U);
    }
}

TEST(HbgOrdinaryPool, TwoConcurrentClaimantsExecuteEveryBlockExactlyOnce) {
    PoolFixture f(1, 32767);
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 0));
    std::vector<std::atomic<unsigned>> seen(32767);
    for (auto &value : seen)
        value.store(0);
    std::atomic<unsigned> errors{0}, last{0};
    auto work = [&] {
        for (unsigned attempt = 0; attempt < 32768; ++attempt) {
            SchedulerOrdinaryPoolDiscovery publication{};
            uint64_t cursor = 0;
            int64_t task = -1;
            const auto found = scheduler_ordinary_pool_discover(f.view, &cursor, &task, &publication);
            if (found == Result::EMPTY) return;
            if (found != Result::FOUND) {
                ++errors;
                return;
            }
            Token token{};
            auto result = scheduler_ordinary_pool_reserve(f.view, task, &token, &publication);
            if (result == Result::EMPTY) return;
            if (result != Result::RESERVED || seen[token.block_idx].fetch_add(1) != 0) {
                ++errors;
                return;
            }
            uint64_t previous = UINT64_MAX;
            if (scheduler_ordinary_pool_confirm(f.view, &token, &previous) != Result::CONFIRMED) ++errors;
            result = scheduler_ordinary_pool_finish(f.view, &token, &previous);
            if (result == Result::TASK_LAST) ++last;
            else if (result != Result::COMPLETED) ++errors;
        }
        ++errors;
    };
    std::thread peer(work);
    work();
    peer.join();
    EXPECT_EQ(errors.load(), 0U);
    EXPECT_EQ(last.load(), 1U);
    for (const auto &value : seen)
        ASSERT_EQ(value.load(), 1U);
    EXPECT_EQ(f.controls[0].next_block, 32767U);
    EXPECT_EQ(f.controls[0].completed_blocks, 32767U);
    EXPECT_LE(f.controls[0].claim_tickets, 32768U);
    EXPECT_EQ(f.words[0].bits, 0U);
}

TEST(HbgOrdinaryPool, ConcurrentPublicationAndRetirementPreserveNeighboringBitsIncludingBit15) {
    for (unsigned round = 0; round < 64; ++round) {
        PoolFixture f(16, 2);
        std::atomic<unsigned> errors{0};
        auto work = [&](int64_t task) {
            if (!scheduler_ordinary_pool_publish(f.view, task)) ++errors;
            for (unsigned block = 0; block < 2; ++block) {
                Token token{};
                uint64_t previous = 0;
                if (scheduler_ordinary_pool_reserve(f.view, task, &token) != Result::RESERVED) ++errors;
                if (scheduler_ordinary_pool_confirm(f.view, &token, &previous) != Result::CONFIRMED) ++errors;
                const auto expected = block == 1 ? Result::TASK_LAST : Result::COMPLETED;
                if (scheduler_ordinary_pool_finish(f.view, &token, &previous) != expected) ++errors;
            }
        };
        std::thread peer(work, 15);
        work(0);
        peer.join();
        EXPECT_EQ(errors.load(), 0U);
        EXPECT_EQ(f.words[0].bits, 0U);
        EXPECT_EQ(f.controls[0].completed_blocks, 2U);
        EXPECT_EQ(f.controls[15].completed_blocks, 2U);
    }
}

TEST(HbgOrdinaryPool, EveryThreeTaskPublicationRetirementInterleavingPreservesTheBitmap) {
    std::array<unsigned, 6> order{0, 1, 2, 3, 4, 5};
    const std::array<int64_t, 3> tasks{0, 1, 15};
    unsigned valid_orders = 0;
    do {
        bool valid = true;
        std::array<bool, 3> published{};
        for (auto event : order) {
            if (event < 3) published[event] = true;
            else if (!published[event - 3]) valid = false;
        }
        if (!valid) continue;
        ++valid_orders;
        PoolFixture f(16, 2);
        uint64_t expected = 0;
        for (auto event : order) {
            const auto task = tasks[event % 3];
            const uint64_t bit = UINT64_C(1) << task;
            if (event < 3) {
                ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, task));
                expected |= bit;
            } else {
                for (unsigned block = 0; block < 2; ++block) {
                    Token token{};
                    uint64_t previous = 0;
                    ASSERT_EQ(scheduler_ordinary_pool_reserve(f.view, task, &token), Result::RESERVED);
                    ASSERT_EQ(scheduler_ordinary_pool_confirm(f.view, &token, &previous), Result::CONFIRMED);
                    ASSERT_EQ(
                        scheduler_ordinary_pool_finish(f.view, &token, &previous),
                        block == 1 ? Result::TASK_LAST : Result::COMPLETED
                    );
                }
                expected &= ~bit;
            }
            ASSERT_EQ(f.words[0].bits, expected);
        }
        EXPECT_EQ(expected, 0U);
    } while (std::next_permutation(order.begin(), order.end()));
    EXPECT_EQ(valid_orders, 90U);
}

TEST(HbgOrdinaryPool, MalformedViewsAndTerminalControlsFailWithoutPublishingWork) {
    PoolFixture f;
    for (auto field :
         {&SchedulerTaskControl::claim_tickets, &SchedulerTaskControl::next_block,
          &SchedulerTaskControl::completed_blocks, &SchedulerTaskControl::admission_state,
          &SchedulerTaskControl::admission_claim_lock}) {
        f.controls[0].*field = 1;
        EXPECT_FALSE(scheduler_ordinary_pool_publish(f.view, 0));
        EXPECT_EQ(f.words[0].bits, 0U);
        f.controls[0].*field = 0;
    }
    f.controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    EXPECT_FALSE(scheduler_ordinary_pool_publish(f.view, 0));
    EXPECT_FALSE(scheduler_ordinary_pool_publish(f.view, -1));
    EXPECT_FALSE(scheduler_ordinary_pool_publish(f.view, 1));
    uint64_t cursor = 1;
    int64_t task = -1;
    EXPECT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task), Result::ERROR);
    cursor = 0;
    f.words[0].bits = 2;
    EXPECT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task), Result::ERROR);
    EXPECT_EQ(task, SCHEDULER_TASK_REF_INVALID);
    auto invalid = f.view;
    invalid.run_epoch = 0;
    EXPECT_FALSE(scheduler_ordinary_pool_view_valid(invalid));
    invalid = f.view;
    invalid.task_count = UINT64_MAX;
    EXPECT_FALSE(scheduler_ordinary_pool_view_valid(invalid));
    invalid = f.view;
    invalid.words = nullptr;
    EXPECT_FALSE(scheduler_ordinary_pool_view_valid(invalid));
}

TEST(HbgOrdinaryPool, AFillHintDefersNewPublicationOnlyUntilTheNextFill) {
    PoolFixture f;
    SchedulerOrdinaryPoolDiscovery poll{};
    uint64_t cursor = 0;
    int64_t task = -1;
    EXPECT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &poll), Result::EMPTY);
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 0));
    EXPECT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &poll), Result::EMPTY);
    EXPECT_EQ(task, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(f.controls[0].claim_tickets, 0U);
    SchedulerOrdinaryPoolDiscovery next_poll{};
    ASSERT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &next_poll), Result::FOUND);
    EXPECT_EQ(task, 0);
    Token token{};
    EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, task, &token, &next_poll), Result::RESERVED);
    EXPECT_EQ(token.block_idx, 0U);
}

TEST(HbgOrdinaryPool, AStalePositiveHintCannotReserveRetiredWork) {
    PoolFixture f(1, 2);
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 0));
    SchedulerOrdinaryPoolDiscovery poll{};
    uint64_t cursor = 0;
    int64_t task = -1;
    ASSERT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &poll), Result::FOUND);
    for (unsigned block = 0; block < 2; ++block) {
        Token token{};
        uint64_t previous = 0;
        ASSERT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &token), Result::RESERVED);
        ASSERT_EQ(scheduler_ordinary_pool_confirm(f.view, &token, &previous), Result::CONFIRMED);
        EXPECT_EQ(
            scheduler_ordinary_pool_finish(f.view, &token, &previous),
            block == 0 ? Result::COMPLETED : Result::TASK_LAST
        );
    }
    ASSERT_EQ(f.words[0].bits, 0U);
    ASSERT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &poll), Result::FOUND);
    Token duplicate{};
    EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, task, &duplicate, &poll), Result::EMPTY);
    EXPECT_EQ(duplicate.phase, Phase::EMPTY);
    EXPECT_EQ(f.controls[0].claim_tickets, 2U);
    SchedulerOrdinaryPoolDiscovery next_poll{};
    EXPECT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &next_poll), Result::EMPTY);
}

TEST(HbgOrdinaryPool, DiscoveryHintCannotCrossBitmapAddressesOrEpochs) {
    PoolFixture f(65, 2);
    SchedulerOrdinaryPoolDiscovery poll{};
    uint64_t cursor = 0;
    int64_t task = -1;
    ASSERT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &poll), Result::EMPTY);
    ASSERT_EQ(cursor, 16U);
    cursor = 64;
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 64));
    ASSERT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &poll), Result::FOUND);
    EXPECT_EQ(task, 64);
    PoolFixture other(1, 2, LaneMask::aiv1());
    ASSERT_TRUE(scheduler_ordinary_pool_publish(other.view, 0));
    cursor = 0;
    ASSERT_EQ(scheduler_ordinary_pool_discover(other.view, &cursor, &task, &poll), Result::FOUND);
    EXPECT_EQ(task, 0);
    PoolFixture epoch(1, 2);
    cursor = 0;
    ASSERT_EQ(scheduler_ordinary_pool_discover(epoch.view, &cursor, &task, &poll), Result::EMPTY);
    ++epoch.view.run_epoch;
    ASSERT_TRUE(scheduler_ordinary_pool_publish(epoch.view, 0));
    EXPECT_EQ(scheduler_ordinary_pool_discover(epoch.view, &cursor, &task, &poll), Result::FOUND);
}

TEST(HbgOrdinaryPool, ReservationRejectsMissingNegativeOrUnrelatedPublicationProof) {
    PoolFixture f(65, 2);
    SchedulerOrdinaryPoolDiscovery empty{}, negative{}, positive{};
    uint64_t cursor = 0;
    int64_t task = -1;
    ASSERT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &negative), Result::EMPTY);
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 0));
    cursor = 0;
    ASSERT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &positive), Result::FOUND);
    Token token{};
    EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &token, &empty), Result::ERROR);
    EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &token, &negative), Result::ERROR);
    EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, 1, &token, &positive), Result::ERROR);
    EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, 64, &token, &positive), Result::ERROR);
    auto other_epoch = f.view;
    ++other_epoch.run_epoch;
    EXPECT_EQ(scheduler_ordinary_pool_reserve(other_epoch, 0, &token, &positive), Result::ERROR);
    PoolFixture other(1, 2);
    ASSERT_TRUE(scheduler_ordinary_pool_publish(other.view, 0));
    EXPECT_EQ(scheduler_ordinary_pool_reserve(other.view, 0, &token, &positive), Result::ERROR);
    EXPECT_EQ(token.phase, Phase::EMPTY);
    EXPECT_EQ(f.controls[0].claim_tickets, 0U);
    EXPECT_EQ(f.controls[1].claim_tickets, 0U);
    EXPECT_EQ(f.controls[64].claim_tickets, 0U);
    EXPECT_EQ(other.controls[0].claim_tickets, 0U);
    EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &token, &positive), Result::RESERVED);
    EXPECT_EQ(token.block_idx, 0U);
}

TEST(HbgOrdinaryPool, PositivePublicationDoesNotReplaceTheFreshOwnerMarkerCheck) {
    PoolFixture f(1, 2);
    ASSERT_TRUE(scheduler_ordinary_pool_publish(f.view, 0));
    SchedulerOrdinaryPoolDiscovery publication{};
    uint64_t cursor = 0;
    int64_t task = -1;
    ASSERT_EQ(scheduler_ordinary_pool_discover(f.view, &cursor, &task, &publication), Result::FOUND);
    for (uint64_t corrupt_marker : {UINT64_C(0), UINT64_C(1), UINT64_C(3)}) {
        f.controls[0].admission_claim_lock = corrupt_marker;
        Token token{};
        EXPECT_EQ(scheduler_ordinary_pool_reserve(f.view, 0, &token, &publication), Result::ERROR);
        EXPECT_EQ(token.phase, Phase::EMPTY);
        EXPECT_EQ(f.controls[0].claim_tickets, 0U);
    }
}

}  // namespace

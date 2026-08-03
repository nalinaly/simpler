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

#include <cstdint>
#include <string>

#include "dist_engine/common/private_claim_participation.h"

namespace {

constexpr uint32_t kPrivateCursorShards = 4;

uint32_t participant_count(uint32_t task_id, uint32_t candidate_count, uint32_t interval) {
    uint32_t count = 0;
    for (uint32_t rank = 0; rank < candidate_count; ++rank) {
        count += fdwic_private_claim_participates(task_id, rank, kPrivateCursorShards, interval) ? 1U : 0U;
    }
    return count;
}

TEST(FdwicPrivateClaimParticipation, AcceptsOnlyMeasuredIntervals) {
    EXPECT_TRUE(fdwic_private_claim_participation_interval_supported(1));
    EXPECT_TRUE(fdwic_private_claim_participation_interval_supported(2));
    EXPECT_TRUE(fdwic_private_claim_participation_interval_supported(4));
    EXPECT_TRUE(fdwic_private_claim_participation_interval_supported(8));
    EXPECT_FALSE(fdwic_private_claim_participation_interval_supported(0));
    EXPECT_FALSE(fdwic_private_claim_participation_interval_supported(3));
    EXPECT_FALSE(fdwic_private_claim_participation_interval_supported(16));
}

TEST(FdwicPrivateClaimParticipation, ReducesEachRolePopulationByExactInterval) {
    constexpr uint32_t kCandidateCounts[] = {96, 32, 64};
    constexpr uint32_t kIntervals[] = {1, 2, 4, 8};
    for (uint32_t interval : kIntervals) {
        for (uint32_t task_id = 0; task_id < 1280; ++task_id) {
            for (uint32_t candidate_count : kCandidateCounts) {
                SCOPED_TRACE(
                    "I=" + std::to_string(interval) + " task=" + std::to_string(task_id) +
                    " candidates=" + std::to_string(candidate_count)
                );
                EXPECT_EQ(participant_count(task_id, candidate_count, interval), candidate_count / interval);
            }
        }
    }
}

TEST(FdwicPrivateClaimParticipation, KeepsOneResidueForEveryTaskSharingACursor) {
    constexpr uint32_t kIntervals[] = {1, 2, 4, 8};
    for (uint32_t interval : kIntervals) {
        for (uint32_t shard = 0; shard < kPrivateCursorShards; ++shard) {
            for (uint32_t task_id = shard; task_id < 1280; task_id += kPrivateCursorShards) {
                EXPECT_EQ(
                    fdwic_private_claim_participation_residue(task_id, kPrivateCursorShards, interval),
                    fdwic_private_claim_participation_residue(shard, kPrivateCursorShards, interval)
                );
                for (uint32_t rank = 0; rank < 96; ++rank) {
                    EXPECT_EQ(
                        fdwic_private_claim_participates(task_id, rank, kPrivateCursorShards, interval),
                        fdwic_private_claim_participates(shard, rank, kPrivateCursorShards, interval)
                    );
                }
            }
        }
    }
}

TEST(FdwicPrivateClaimParticipation, MatchesCase1ClaimAttemptModel) {
    constexpr uint32_t kIntervals[] = {1, 2, 4, 8};
    constexpr uint32_t kAllocTasks = 256;
    constexpr uint32_t kAicTasks = 512;
    constexpr uint32_t kAivTasks = 512;
    constexpr uint32_t kAllocCandidates = 96;
    constexpr uint32_t kAicCandidates = 32;
    constexpr uint32_t kAivCandidates = 64;
    constexpr uint32_t kFlatAttempts =
        kAllocTasks * kAllocCandidates + kAicTasks * kAicCandidates + kAivTasks * kAivCandidates;

    for (uint32_t interval : kIntervals) {
        uint32_t attempts = 0;
        for (uint32_t task_id = 0; task_id < kAllocTasks; ++task_id) {
            attempts += participant_count(task_id, kAllocCandidates, interval);
        }
        for (uint32_t task_id = 0; task_id < kAicTasks; ++task_id) {
            attempts += participant_count(task_id, kAicCandidates, interval);
            attempts += participant_count(task_id, kAivCandidates, interval);
        }
        EXPECT_EQ(attempts, kFlatAttempts / interval);
    }
}

}  // namespace

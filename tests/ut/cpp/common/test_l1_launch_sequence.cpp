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
#include <vector>

#include "l1_launch_sequence.h"

namespace {

class FakeLaunchSequence {
public:
    L1LaunchSequenceOps ops() {
        return {
            .context = this,
            .wait_event = wait_event,
            .memset_handshake = memset_handshake,
            .record_event = record_event,
            .launch_aicpu = launch_aicpu,
            .launch_aicore = launch_aicore,
            .cancel_waiting_aicore = cancel_waiting_aicore,
        };
    }

    static int wait_event(void *context, void *stream, void *event) noexcept {
        return static_cast<FakeLaunchSequence *>(context)->append("wait", stream, event);
    }

    static int memset_handshake(void *context, void *stream) noexcept {
        return static_cast<FakeLaunchSequence *>(context)->append("memset", stream, nullptr);
    }

    static int record_event(void *context, void *event, void *stream) noexcept {
        return static_cast<FakeLaunchSequence *>(context)->append("record", stream, event);
    }

    static int launch_aicpu(void *context, void *stream) noexcept {
        return static_cast<FakeLaunchSequence *>(context)->append("aicpu", stream, nullptr);
    }

    static int launch_aicore(void *context, void *stream) noexcept {
        return static_cast<FakeLaunchSequence *>(context)->append("aicore", stream, nullptr);
    }

    static int cancel_waiting_aicore(void *context, void *stream) noexcept {
        return static_cast<FakeLaunchSequence *>(context)->append("cancel", stream, nullptr);
    }

    int append(const char *operation, void *stream, void *event) noexcept {
        try {
            calls.push_back(
                std::string(operation) + ":s" + std::to_string(reinterpret_cast<uintptr_t>(stream)) + ":e" +
                std::to_string(reinterpret_cast<uintptr_t>(event))
            );
        } catch (...) {
            return -999;
        }
        if (calls.size() == fail_at) return failure;
        return calls.size() == cleanup_fail_at ? cleanup_failure : 0;
    }

    std::vector<std::string> calls;
    size_t fail_at{0};
    int failure{-77};
    size_t cleanup_fail_at{0};
    int cleanup_failure{-88};
};

L1LaunchSequenceHandles handles(bool wait_for_serial_tail, bool wait_for_prepare_tail = true) {
    return {
        .caller_stream = reinterpret_cast<void *>(1),
        .hidden_stream = reinterpret_cast<void *>(2),
        .prepare_tail_event = reinterpret_cast<void *>(3),
        .start_event = reinterpret_cast<void *>(4),
        .aicore_done_event = reinterpret_cast<void *>(5),
        .serial_tail_event = reinterpret_cast<void *>(6),
        .wait_for_prepare_tail = wait_for_prepare_tail,
        .wait_for_serial_tail = wait_for_serial_tail,
    };
}

TEST(L1LaunchSequence, FirstInvocationHasExactSingleOperatorForkJoinOrder) {
    FakeLaunchSequence fake;
    ASSERT_EQ(enqueue_l1_launch_sequence(fake.ops(), handles(false)), 0);
    EXPECT_EQ(
        fake.calls, (std::vector<std::string>{
                        "wait:s1:e3", "memset:s1:e0", "record:s1:e4", "wait:s2:e4", "aicore:s2:e0", "record:s2:e5",
                        "aicpu:s1:e0", "wait:s1:e5", "record:s1:e6"
                    })
    );
}

TEST(L1LaunchSequence, SubsequentInvocationWaitsForPriorSerialTailBeforeMutation) {
    FakeLaunchSequence fake;
    ASSERT_EQ(enqueue_l1_launch_sequence(fake.ops(), handles(true, false)), 0);
    EXPECT_EQ(
        fake.calls, (std::vector<std::string>{
                        "wait:s1:e6", "memset:s1:e0", "record:s1:e4", "wait:s2:e4", "aicore:s2:e0", "record:s2:e5",
                        "aicpu:s1:e0", "wait:s1:e5", "record:s1:e6"
                    })
    );
}

TEST(L1LaunchSequence, StopsBeforeHiddenAicoreBecomesResident) {
    for (size_t failure_index = 1; failure_index <= 5; ++failure_index) {
        FakeLaunchSequence fake;
        fake.fail_at = failure_index;
        EXPECT_EQ(enqueue_l1_launch_sequence(fake.ops(), handles(true, false)), fake.failure);
        EXPECT_EQ(fake.calls.size(), failure_index);
    }
}

TEST(L1LaunchSequence, CompletionRecordFailureCancelsAndRetriesTheJoin) {
    FakeLaunchSequence fake;
    fake.fail_at = 6;

    EXPECT_EQ(enqueue_l1_launch_sequence(fake.ops(), handles(true, false)), fake.failure);
    EXPECT_EQ(
        fake.calls, (std::vector<std::string>{
                        "wait:s1:e6", "memset:s1:e0", "record:s1:e4", "wait:s2:e4", "aicore:s2:e0", "record:s2:e5",
                        "cancel:s1:e0", "record:s2:e5", "wait:s1:e5", "record:s1:e6"
                    })
    );
}

TEST(L1LaunchSequence, AicpuLaunchFailureCancelsAndJoinsHiddenAicore) {
    FakeLaunchSequence fake;
    fake.fail_at = 7;

    EXPECT_EQ(enqueue_l1_launch_sequence(fake.ops(), handles(true, false)), fake.failure);
    EXPECT_EQ(
        fake.calls, (std::vector<std::string>{
                        "wait:s1:e6", "memset:s1:e0", "record:s1:e4", "wait:s2:e4", "aicore:s2:e0", "record:s2:e5",
                        "aicpu:s1:e0", "cancel:s1:e0", "wait:s1:e5", "record:s1:e6"
                    })
    );
}

TEST(L1LaunchSequence, ReportsCleanupFailureWhenHostCancelCannotBeEnqueued) {
    FakeLaunchSequence fake;
    fake.fail_at = 7;
    fake.cleanup_fail_at = 8;

    EXPECT_EQ(enqueue_l1_launch_sequence(fake.ops(), handles(true, false)), fake.cleanup_failure);
    EXPECT_EQ(fake.calls.back(), "cancel:s1:e0");
    EXPECT_EQ(fake.calls.size(), 8U);
}

TEST(L1LaunchSequence, StopsAtPostLaunchJoinAndTailFailures) {
    for (size_t failure_index : {8U, 9U}) {
        FakeLaunchSequence fake;
        fake.fail_at = failure_index;
        EXPECT_EQ(enqueue_l1_launch_sequence(fake.ops(), handles(true, false)), fake.failure);
        EXPECT_EQ(fake.calls.size(), failure_index);
    }
}

TEST(L1LaunchSequence, PrepareTailIsConsumedOnlyByWarmup) {
    FakeLaunchSequence warmup;
    ASSERT_EQ(enqueue_l1_launch_sequence(warmup.ops(), handles(false, true)), 0);
    ASSERT_FALSE(warmup.calls.empty());
    EXPECT_EQ(warmup.calls.front(), "wait:s1:e3");

    FakeLaunchSequence capture;
    ASSERT_EQ(enqueue_l1_launch_sequence(capture.ops(), handles(true, false)), 0);
    ASSERT_FALSE(capture.calls.empty());
    EXPECT_EQ(capture.calls.front(), "wait:s1:e6");
}

TEST(L1LaunchSequence, RejectsMissingHandlesBeforeEnqueue) {
    FakeLaunchSequence fake;
    auto invalid = handles(false);
    invalid.hidden_stream = nullptr;
    EXPECT_EQ(enqueue_l1_launch_sequence(fake.ops(), invalid), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_TRUE(fake.calls.empty());
}

TEST(L1LaunchSequence, RejectsMissingHostCancelBeforeEnqueue) {
    FakeLaunchSequence fake;
    auto invalid = fake.ops();
    invalid.cancel_waiting_aicore = nullptr;
    EXPECT_EQ(enqueue_l1_launch_sequence(invalid, handles(false)), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_TRUE(fake.calls.empty());
}

}  // namespace

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
#include <unordered_set>
#include <utility>
#include <vector>

#include "l1_execution_state.h"

namespace {

class FakeL1Runtime {
public:
    L1RuntimeOps ops() {
        return L1RuntimeOps{
            .context = this,
            .get_current_device = get_current_device,
            .create_hidden_stream = create_hidden_stream,
            .destroy_hidden_stream = destroy_hidden_stream,
            .create_event = create_event,
            .destroy_event = destroy_event,
        };
    }

    void fail_create_attempt(size_t attempt, int error = -77) {
        failed_create_attempt_ = attempt;
        create_error_ = error;
    }

    void fail_next_destroys(size_t count, int error = -88) {
        destroy_failures_ = count;
        destroy_error_ = error;
    }

    static int get_current_device(void *context, int *device_id) {
        auto &self = *static_cast<FakeL1Runtime *>(context);
        self.calls_.emplace_back("get_current_device");
        if (self.get_device_error_ != 0) return self.get_device_error_;
        *device_id = self.current_device_;
        return 0;
    }

    static int create_hidden_stream(void *context, void **stream) {
        auto &self = *static_cast<FakeL1Runtime *>(context);
        self.calls_.emplace_back("create_hidden_stream");
        return self.create_handle(stream);
    }

    static int destroy_hidden_stream(void *context, void *stream) {
        auto &self = *static_cast<FakeL1Runtime *>(context);
        self.calls_.push_back("destroy_hidden_stream:" + std::to_string(reinterpret_cast<uintptr_t>(stream)));
        return self.destroy_handle(stream);
    }

    static int create_event(void *context, void **event) {
        auto &self = *static_cast<FakeL1Runtime *>(context);
        self.calls_.emplace_back("create_event");
        return self.create_handle(event);
    }

    static int destroy_event(void *context, void *event) {
        auto &self = *static_cast<FakeL1Runtime *>(context);
        self.calls_.push_back("destroy_event:" + std::to_string(reinterpret_cast<uintptr_t>(event)));
        return self.destroy_handle(event);
    }

    int current_device_{3};
    int get_device_error_{0};
    std::vector<std::string> calls_;
    std::unordered_set<void *> live_;

private:
    int create_handle(void **out) {
        ++create_attempt_;
        if (create_attempt_ == failed_create_attempt_) {
            return create_error_;
        }
        void *handle = reinterpret_cast<void *>(++next_handle_);
        live_.insert(handle);
        *out = handle;
        return 0;
    }

    int destroy_handle(void *handle) {
        if (destroy_failures_ > 0) {
            --destroy_failures_;
            return destroy_error_;
        }
        if (live_.erase(handle) != 1) {
            ADD_FAILURE() << "destroyed a handle that was not live";
            return -1;
        }
        return 0;
    }

    uintptr_t next_handle_{0};
    size_t create_attempt_{0};
    size_t failed_create_attempt_{0};
    int create_error_{-77};
    size_t destroy_failures_{0};
    int destroy_error_{-88};
};

TEST(DeviceExecutionModeState, SelectsExactlyOneIrreversibleOwnershipModel) {
    DeviceExecutionModeState mode;
    EXPECT_EQ(mode.mode(), DeviceExecutionMode::Uninitialized);

    ASSERT_EQ(mode.claim_l1_borrowed(), 0);
    EXPECT_TRUE(mode.accepts_l1_calls());
    EXPECT_TRUE(mode.requires_explicit_l1_close());
    EXPECT_EQ(mode.claim_l2_owned(), PTO_RUNTIME_ERR_INVALID_STATE);

    ASSERT_EQ(mode.abort_l1_initialization(), 0);
    ASSERT_EQ(mode.claim_l2_owned(), 0);
    EXPECT_TRUE(mode.accepts_l2_calls());
    EXPECT_EQ(mode.claim_l1_borrowed(), PTO_RUNTIME_ERR_INVALID_STATE);

    ASSERT_EQ(mode.mark_closed(), 0);
    EXPECT_EQ(mode.mode(), DeviceExecutionMode::Closed);
    EXPECT_EQ(mode.mark_closed(), 0);
    EXPECT_EQ(mode.claim_l1_borrowed(), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(L1ExecutionState, CreatesOnlyOneHiddenStreamAndFourPersistentEvents) {
    FakeL1Runtime runtime;
    L1ExecutionState state;

    ASSERT_EQ(state.initialize(3, runtime.ops()), 0);
    EXPECT_EQ(state.phase(), L1ContextPhase::Collecting);
    EXPECT_EQ(state.device_id(), 3);
    EXPECT_NE(state.hidden_aicore_stream(), nullptr);
    EXPECT_EQ(runtime.live_.size(), 5u);
    for (size_t i = 0; i < static_cast<size_t>(L1EventKind::Count); ++i) {
        EXPECT_NE(state.event(static_cast<L1EventKind>(i)), nullptr);
    }

    ASSERT_EQ(state.mark_ready_enqueued(), 0);
    EXPECT_EQ(state.phase(), L1ContextPhase::ReadyEnqueued);
    ASSERT_EQ(state.seal(), 0);
    EXPECT_EQ(state.phase(), L1ContextPhase::Sealed);

    ASSERT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), L1ContextPhase::Closed);
    EXPECT_TRUE(runtime.live_.empty());
    EXPECT_EQ(
        runtime.calls_, (std::vector<std::string>{
                            "get_current_device", "create_hidden_stream", "create_event", "create_event",
                            "create_event", "create_event", "get_current_device", "destroy_event:5", "destroy_event:4",
                            "destroy_event:3", "destroy_event:2", "destroy_hidden_stream:1"
                        })
    );

    const size_t call_count = runtime.calls_.size();
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(runtime.calls_.size(), call_count) << "idempotent close must not touch the runtime again";
}

TEST(L1ExecutionState, DeviceMismatchFailsBeforeCreatingAnyResource) {
    FakeL1Runtime runtime;
    L1ExecutionState state;

    EXPECT_EQ(state.initialize(4, runtime.ops()), PTO_RUNTIME_ERR_DEVICE_MISMATCH);
    EXPECT_EQ(state.phase(), L1ContextPhase::New);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_TRUE(runtime.live_.empty());
    EXPECT_EQ(runtime.calls_, (std::vector<std::string>{"get_current_device"}));
}

TEST(L1ExecutionState, CreateFailureRollsBackOnlyAlreadyOwnedHandlesInReverseOrder) {
    FakeL1Runtime runtime;
    runtime.fail_create_attempt(4, -77);  // stream + two events succeed; third event fails
    L1ExecutionState state;

    EXPECT_EQ(state.initialize(3, runtime.ops()), PTO_RUNTIME_ERR_RUNTIME_FAILURE);
    EXPECT_EQ(state.last_runtime_error(), -77);
    EXPECT_EQ(state.phase(), L1ContextPhase::New);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_TRUE(runtime.live_.empty());
    EXPECT_EQ(
        runtime.calls_, (std::vector<std::string>{
                            "get_current_device", "create_hidden_stream", "create_event", "create_event",
                            "create_event", "destroy_event:3", "destroy_event:2", "destroy_hidden_stream:1"
                        })
    );
}

TEST(L1ExecutionState, FailedRollbackPoisonsAndKeepsTheHandleForExplicitCloseRetry) {
    FakeL1Runtime runtime;
    runtime.fail_create_attempt(3, -77);  // stream + one event succeed; second event fails
    runtime.fail_next_destroys(1, -88);   // first rollback destroy keeps that event live
    L1ExecutionState state;

    EXPECT_EQ(state.initialize(3, runtime.ops()), PTO_RUNTIME_ERR_RUNTIME_FAILURE);
    EXPECT_EQ(state.phase(), L1ContextPhase::Poisoned);
    EXPECT_TRUE(state.has_live_resources());
    EXPECT_EQ(runtime.live_.size(), 1u);
    EXPECT_EQ(state.last_runtime_error(), -88);

    ASSERT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), L1ContextPhase::Closed);
    EXPECT_TRUE(runtime.live_.empty());
}

TEST(L1ExecutionState, CloseOnTheWrongCurrentDevicePreservesEveryOwnedHandle) {
    FakeL1Runtime runtime;
    L1ExecutionState state;
    ASSERT_EQ(state.initialize(3, runtime.ops()), 0);
    const size_t live_before = runtime.live_.size();

    runtime.current_device_ = 4;
    EXPECT_EQ(state.close(), PTO_RUNTIME_ERR_DEVICE_MISMATCH);
    EXPECT_EQ(state.phase(), L1ContextPhase::Collecting);
    EXPECT_EQ(runtime.live_.size(), live_before);

    runtime.current_device_ = 3;
    EXPECT_EQ(state.close(), 0);
    EXPECT_TRUE(runtime.live_.empty());
}

TEST(L1ExecutionState, ExplicitPoisonIsStickyUntilClose) {
    FakeL1Runtime runtime;
    L1ExecutionState state;
    ASSERT_EQ(state.initialize(3, runtime.ops()), 0);

    state.poison(12345);
    EXPECT_EQ(state.phase(), L1ContextPhase::Poisoned);
    EXPECT_EQ(state.last_runtime_error(), 12345);
    EXPECT_EQ(state.seal(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.close(), 0);
}

}  // namespace

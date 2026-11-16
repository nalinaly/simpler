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
#include "host/scheduler_notification_cache.h"
#include <gtest/gtest.h>

namespace {
struct Counters {
    int created{0};
    int prepared{0};
    int released{0};
    int abandoned{0};
    bool fail_release{false};
};
class FakePool final : public SchedulerNotificationStorage {
public:
    FakePool(void *context, Counters &counts) :
        SchedulerNotificationStorage(context),
        counts_(counts) {
        ++counts_.created;
    }
    size_t capacity() const override { return capacity_; }
    SchedulerDoneNotifications prepare(size_t count, size_t bit_count) override {
        ++counts_.prepared;
        capacity_ = std::max(capacity_, count);
        SchedulerDoneNotifications result{std::vector<uint64_t>(count, 32), std::vector<uint64_t>(count, 0), {}};
        std::fill_n(result.bit_write_addresses.begin(), bit_count, 64);
        return result;
    }
    int release() override {
        ++counts_.released;
        return counts_.fail_release ? -7 : 0;
    }
    void abandon() noexcept override { ++counts_.abandoned; }

private:
    Counters &counts_;
    size_t capacity_{0};
};
struct Fixture {
    SchedulerNotificationCache cache;
    Counters counts;
    int context{};
    int acquire(
        SchedulerDoneNotifications &out, size_t size = 8, size_t bit_count = 7,
        SchedulerNotificationPurpose purpose = SchedulerNotificationPurpose::TASK_STATE
    ) {
        return cache.acquire(
            0, purpose, &context, size, bit_count,
            [&](void *ctx) {
                return std::make_shared<FakePool>(ctx, counts);
            },
            &out
        );
    }
};

TEST(SchedulerNotificationCache, CapturedLeaseForcesAnotherPoolAndBlocksRelease) {
    Fixture f;
    SchedulerDoneNotifications first, second;
    ASSERT_EQ(f.acquire(first), 0);
    auto captured = first.owner;
    first = {};
    ASSERT_EQ(f.acquire(second), 0);
    EXPECT_NE(second.owner, captured);
    EXPECT_EQ(f.counts.created, 2);
    EXPECT_NE(f.cache.release(), 0);
    EXPECT_EQ(f.counts.released, 0);
    captured.reset();
    second = {};
    EXPECT_EQ(f.cache.release(), 0);
    EXPECT_EQ(f.counts.released, 2);
}

TEST(SchedulerNotificationCache, ReusesAndGrowsUnleasedPoolWithIndependentPurpose) {
    Fixture f;
    SchedulerDoneNotifications first;
    ASSERT_EQ(f.acquire(first), 0);
    auto *identity = first.owner.get();
    first = {};
    ASSERT_EQ(f.acquire(first, 16, 15), 0);
    EXPECT_EQ(first.owner.get(), identity);
    EXPECT_EQ(first.bit_write_addresses[14], 64u);
    EXPECT_EQ(first.bit_write_addresses[15], 0u);
    SchedulerDoneNotifications mix;
    ASSERT_EQ(f.acquire(mix, 6, 0, SchedulerNotificationPurpose::MIX_ACK), 0);
    EXPECT_NE(mix.owner, first.owner);
    EXPECT_EQ(f.counts.created, 2);
    first = {};
    mix = {};
    EXPECT_EQ(f.cache.release(), 0);
}

TEST(SchedulerNotificationCache, ReleaseFailureRetainsHandleForRetry) {
    Fixture f;
    SchedulerDoneNotifications lease;
    ASSERT_EQ(f.acquire(lease), 0);
    lease = {};
    f.counts.fail_release = true;
    EXPECT_EQ(f.cache.release(), -7);
    EXPECT_EQ(f.counts.released, 1);
    f.counts.fail_release = false;
    EXPECT_EQ(f.cache.release(), 0);
    EXPECT_EQ(f.counts.released, 2);
    EXPECT_EQ(f.cache.release(), 0);
    EXPECT_EQ(f.counts.released, 2);
}

TEST(SchedulerNotificationCache, PostResetAbandonDoesNotCallDeviceRelease) {
    Fixture f;
    SchedulerDoneNotifications lease;
    ASSERT_EQ(f.acquire(lease), 0);
    f.cache.abandon();
    EXPECT_EQ(f.counts.abandoned, 1);
    EXPECT_EQ(f.counts.released, 0);
    lease = {};
    EXPECT_EQ(f.counts.released, 0);
    EXPECT_EQ(f.cache.release(), 0);
}

TEST(SchedulerNotificationCache, RefusesReuseAcrossContextGenerations) {
    Fixture f;
    SchedulerDoneNotifications lease;
    ASSERT_EQ(f.acquire(lease), 0);
    lease = {};
    int another_context = 1;
    EXPECT_NE(
        f.cache.acquire(
            0, SchedulerNotificationPurpose::TASK_STATE, &another_context, 8, 7,
            [&](void *ctx) {
                return std::make_shared<FakePool>(ctx, f.counts);
            },
            &lease
        ),
        0
    );
    EXPECT_EQ(f.counts.created, 1);
    EXPECT_EQ(f.cache.release(), 0);
}
}  // namespace

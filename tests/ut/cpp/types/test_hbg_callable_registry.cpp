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

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "hbg_callable_registry.h"

namespace {

using simpler::hbg::acquire_hbg_callable_registration;
using simpler::hbg::HbgCallableRegistration;
using simpler::hbg::HbgCallableRegistry;
using simpler::hbg::HbgCallableRegistryPhase;
using simpler::hbg::HbgCallableRegistryStatus;
using simpler::hbg::HbgCallableStatus;
using simpler::hbg::publish_hbg_callable_registration;
using simpler::hbg::reset_hbg_callable_registry;
using simpler::hbg::seal_hbg_callable_registration;

HbgCallableRegistration make_registration(int32_t callable_id = 3) {
    HbgCallableRegistration registration;
    registration.callable_id = callable_id;
    registration.tensor_count = 2;
    registration.scalar_count = 1;
    registration.callable_hash = 0x1122334455667788ULL;
    registration.function_binding_hash = 0x8877665544332211ULL;
    EXPECT_EQ(seal_hbg_callable_registration(&registration), HbgCallableStatus::Ok);
    return registration;
}

TEST(HbgCallableRegistry, PublishesIndependentCallableIdsAndAcquiresExactSnapshots) {
    HbgCallableRegistry registry;
    const HbgCallableRegistration first = make_registration(3);
    HbgCallableRegistration second = make_registration(7);
    ++second.callable_hash;
    ASSERT_EQ(seal_hbg_callable_registration(&second), HbgCallableStatus::Ok);

    EXPECT_EQ(publish_hbg_callable_registration(&registry, &first), HbgCallableRegistryStatus::Published);
    EXPECT_EQ(publish_hbg_callable_registration(&registry, &second), HbgCallableRegistryStatus::Published);
    EXPECT_EQ(registry.entries[3].phase.load(std::memory_order_acquire), HbgCallableRegistryPhase::Ready);
    EXPECT_EQ(registry.entries[7].phase.load(std::memory_order_acquire), HbgCallableRegistryPhase::Ready);

    HbgCallableRegistration acquired{};
    EXPECT_EQ(acquire_hbg_callable_registration(&registry, 7, &acquired), HbgCallableRegistryStatus::Acquired);
    EXPECT_EQ(std::memcmp(&acquired, &second, sizeof(second)), 0);
}

TEST(HbgCallableRegistry, AcceptsOnlyAnIdenticalDuplicateAtOneCallableId) {
    HbgCallableRegistry registry;
    const HbgCallableRegistration registration = make_registration();
    ASSERT_EQ(publish_hbg_callable_registration(&registry, &registration), HbgCallableRegistryStatus::Published);
    EXPECT_EQ(
        publish_hbg_callable_registration(&registry, &registration), HbgCallableRegistryStatus::AlreadyRegistered
    );

    HbgCallableRegistration conflicting = registration;
    ++conflicting.function_binding_hash;
    ASSERT_EQ(seal_hbg_callable_registration(&conflicting), HbgCallableStatus::Ok);
    EXPECT_EQ(publish_hbg_callable_registration(&registry, &conflicting), HbgCallableRegistryStatus::Conflict);

    HbgCallableRegistration acquired{};
    ASSERT_EQ(acquire_hbg_callable_registration(&registry, 3, &acquired), HbgCallableRegistryStatus::Acquired);
    EXPECT_EQ(std::memcmp(&acquired, &registration, sizeof(registration)), 0);
}

TEST(HbgCallableRegistry, InvalidRegistrationNeverClaimsOrMutatesAnEntry) {
    HbgCallableRegistry registry;
    HbgCallableRegistration invalid = make_registration();
    ++invalid.registration_hash;

    EXPECT_EQ(publish_hbg_callable_registration(&registry, &invalid), HbgCallableRegistryStatus::CallableRejected);
    EXPECT_EQ(registry.entries[3].phase.load(std::memory_order_acquire), HbgCallableRegistryPhase::Empty);
    const HbgCallableRegistration empty{};
    EXPECT_EQ(std::memcmp(&registry.entries[3].registration, &empty, sizeof(empty)), 0);

    invalid = make_registration();
    invalid.callable_id = MAX_REGISTERED_CALLABLE_IDS;
    EXPECT_EQ(seal_hbg_callable_registration(&invalid), HbgCallableStatus::InvalidCallableId);
    EXPECT_EQ(publish_hbg_callable_registration(&registry, &invalid), HbgCallableRegistryStatus::CallableRejected);
}

TEST(HbgCallableRegistry, PublishingCorruptAndEmptyStatesFailClosed) {
    HbgCallableRegistry registry;
    HbgCallableRegistration output = make_registration(9);
    const HbgCallableRegistration before = output;

    EXPECT_EQ(acquire_hbg_callable_registration(&registry, 3, &output), HbgCallableRegistryStatus::NotReady);
    EXPECT_EQ(std::memcmp(&output, &before, sizeof(output)), 0);

    registry.entries[3].phase.store(HbgCallableRegistryPhase::Publishing, std::memory_order_release);
    EXPECT_EQ(acquire_hbg_callable_registration(&registry, 3, &output), HbgCallableRegistryStatus::Publishing);
    EXPECT_EQ(std::memcmp(&output, &before, sizeof(output)), 0);

    registry.entries[3].phase.store(static_cast<HbgCallableRegistryPhase>(99), std::memory_order_release);
    EXPECT_EQ(acquire_hbg_callable_registration(&registry, 3, &output), HbgCallableRegistryStatus::CorruptState);
    EXPECT_EQ(std::memcmp(&output, &before, sizeof(output)), 0);
}

TEST(HbgCallableRegistry, RejectsNullArgumentsAndOutOfRangeAcquires) {
    HbgCallableRegistry registry;
    const HbgCallableRegistration registration = make_registration();
    HbgCallableRegistration output{};

    EXPECT_EQ(publish_hbg_callable_registration(nullptr, &registration), HbgCallableRegistryStatus::NullArgument);
    EXPECT_EQ(publish_hbg_callable_registration(&registry, nullptr), HbgCallableRegistryStatus::NullArgument);
    EXPECT_EQ(acquire_hbg_callable_registration(nullptr, 3, &output), HbgCallableRegistryStatus::NullArgument);
    EXPECT_EQ(acquire_hbg_callable_registration(&registry, 3, nullptr), HbgCallableRegistryStatus::NullArgument);
    EXPECT_EQ(
        acquire_hbg_callable_registration(&registry, MAX_REGISTERED_CALLABLE_IDS, &output),
        HbgCallableRegistryStatus::InvalidCallableId
    );
}

TEST(HbgCallableRegistry, ResetAllowsCallableIdsToBelongToANewContext) {
    HbgCallableRegistry registry;
    const HbgCallableRegistration first = make_registration(0);
    ASSERT_EQ(publish_hbg_callable_registration(&registry, &first), HbgCallableRegistryStatus::Published);

    reset_hbg_callable_registry(&registry);
    HbgCallableRegistration output{};
    EXPECT_EQ(acquire_hbg_callable_registration(&registry, 0, &output), HbgCallableRegistryStatus::NotReady);

    HbgCallableRegistration second = first;
    ++second.callable_hash;
    ASSERT_EQ(seal_hbg_callable_registration(&second), HbgCallableStatus::Ok);
    EXPECT_EQ(publish_hbg_callable_registration(&registry, &second), HbgCallableRegistryStatus::Published);
}

}  // namespace

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

#include "hbg_context_registry.h"
#include "hbg_execution_slot_registry.h"

namespace {

using simpler::hbg::acquire_hbg_execution_slot_registration;
using simpler::hbg::hbg_minimum_launch_blob_size;
using simpler::hbg::HbgContextRegistry;
using simpler::hbg::HbgExecutionSlotRegistration;
using simpler::hbg::HbgExecutionSlotRegistry;
using simpler::hbg::HbgExecutionSlotRegistryPhase;
using simpler::hbg::HbgExecutionSlotRegistryStatus;
using simpler::hbg::initialize_hbg_context_registry;
using simpler::hbg::publish_hbg_execution_slot_registration;
using simpler::hbg::reset_hbg_execution_slot_registry;
using simpler::hbg::seal_hbg_execution_slot_registration;
using simpler::hbg::valid_hbg_context_registry_header;

HbgExecutionSlotRegistration make_registration(uint64_t generation = 7) {
    HbgExecutionSlotRegistration registration;
    registration.device_id = 1;
    registration.binding.shared_memory_base = 0x100000;
    registration.binding.shared_memory_capacity = 0x2000;
    registration.binding.runtime_arena_base = 0x200000;
    registration.binding.runtime_arena_capacity = 0x3000;
    registration.binding.gm_heap_base = 0x300000;
    registration.binding.gm_heap_capacity = 0x4000;
    registration.binding.runtime_offset = 0x400;
    registration.binding.slot_generation = generation;
    registration.outer_runtime_base = 0x400000;
    registration.outer_runtime_size = 0x1000;
    registration.device_kernel_args_base = 0x500000;
    registration.device_kernel_args_size = 0x800;
    registration.binary_generation = 11;
    EXPECT_TRUE(hbg_minimum_launch_blob_size(registration.binding, &registration.max_launch_blob_size));
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), simpler::hbg::HbgExecutionSlotStatus::Ok);
    return registration;
}

TEST(HbgExecutionSlotRegistry, PublishesOnlyAfterValidationAndAcquiresAnImmutableSnapshot) {
    HbgExecutionSlotRegistry registry;
    const HbgExecutionSlotRegistration registration = make_registration();

    EXPECT_EQ(
        publish_hbg_execution_slot_registration(&registry, &registration, 1), HbgExecutionSlotRegistryStatus::Published
    );
    EXPECT_EQ(registry.phase.load(std::memory_order_acquire), HbgExecutionSlotRegistryPhase::Ready);

    HbgExecutionSlotRegistration acquired{};
    EXPECT_EQ(
        acquire_hbg_execution_slot_registration(&registry, 1, &acquired), HbgExecutionSlotRegistryStatus::Acquired
    );
    EXPECT_EQ(std::memcmp(&acquired, &registration, sizeof(registration)), 0);
}

TEST(HbgExecutionSlotRegistry, InvalidInputDoesNotPublishOrMutateTheOwner) {
    HbgExecutionSlotRegistry registry;
    HbgExecutionSlotRegistration invalid = make_registration();
    ++invalid.registration_hash;

    EXPECT_EQ(
        publish_hbg_execution_slot_registration(&registry, &invalid, 1), HbgExecutionSlotRegistryStatus::SlotRejected
    );
    EXPECT_EQ(registry.phase.load(std::memory_order_acquire), HbgExecutionSlotRegistryPhase::Empty);
    const HbgExecutionSlotRegistration empty{};
    EXPECT_EQ(std::memcmp(&registry.registration, &empty, sizeof(empty)), 0);

    const HbgExecutionSlotRegistration valid = make_registration();
    EXPECT_EQ(
        publish_hbg_execution_slot_registration(&registry, &valid, 0), HbgExecutionSlotRegistryStatus::SlotRejected
    );
    EXPECT_EQ(registry.phase.load(std::memory_order_acquire), HbgExecutionSlotRegistryPhase::Empty);
}

TEST(HbgExecutionSlotRegistry, AcceptsOnlyByteIdenticalDuplicateRegistration) {
    HbgExecutionSlotRegistry registry;
    const HbgExecutionSlotRegistration registration = make_registration();
    ASSERT_EQ(
        publish_hbg_execution_slot_registration(&registry, &registration, 1), HbgExecutionSlotRegistryStatus::Published
    );

    EXPECT_EQ(
        publish_hbg_execution_slot_registration(&registry, &registration, 1),
        HbgExecutionSlotRegistryStatus::AlreadyRegistered
    );

    const HbgExecutionSlotRegistration conflicting = make_registration(8);
    EXPECT_EQ(
        publish_hbg_execution_slot_registration(&registry, &conflicting, 1), HbgExecutionSlotRegistryStatus::Conflict
    );
    EXPECT_EQ(std::memcmp(&registry.registration, &registration, sizeof(registration)), 0);
}

TEST(HbgExecutionSlotRegistry, PublishingAndCorruptPhasesFailClosedWithoutAnOutputSnapshot) {
    HbgExecutionSlotRegistry registry;
    HbgExecutionSlotRegistration output = make_registration(99);
    const HbgExecutionSlotRegistration before = output;

    registry.phase.store(HbgExecutionSlotRegistryPhase::Publishing, std::memory_order_release);
    EXPECT_EQ(
        acquire_hbg_execution_slot_registration(&registry, 1, &output), HbgExecutionSlotRegistryStatus::Publishing
    );
    EXPECT_EQ(std::memcmp(&output, &before, sizeof(output)), 0);

    registry.phase.store(static_cast<HbgExecutionSlotRegistryPhase>(99), std::memory_order_release);
    EXPECT_EQ(
        acquire_hbg_execution_slot_registration(&registry, 1, &output), HbgExecutionSlotRegistryStatus::CorruptState
    );
    EXPECT_EQ(std::memcmp(&output, &before, sizeof(output)), 0);
}

TEST(HbgExecutionSlotRegistry, RejectsNullArgumentsAndAnEmptyAcquire) {
    HbgExecutionSlotRegistry registry;
    const HbgExecutionSlotRegistration registration = make_registration();
    HbgExecutionSlotRegistration output{};

    EXPECT_EQ(
        publish_hbg_execution_slot_registration(nullptr, &registration, 1), HbgExecutionSlotRegistryStatus::NullArgument
    );
    EXPECT_EQ(
        publish_hbg_execution_slot_registration(&registry, nullptr, 1), HbgExecutionSlotRegistryStatus::NullArgument
    );
    EXPECT_EQ(
        acquire_hbg_execution_slot_registration(nullptr, 1, &output), HbgExecutionSlotRegistryStatus::NullArgument
    );
    EXPECT_EQ(
        acquire_hbg_execution_slot_registration(&registry, 1, nullptr), HbgExecutionSlotRegistryStatus::NullArgument
    );
    EXPECT_EQ(acquire_hbg_execution_slot_registration(&registry, 1, &output), HbgExecutionSlotRegistryStatus::NotReady);
}

TEST(HbgExecutionSlotRegistry, ResetStartsANewExternallyQuiescedContext) {
    HbgExecutionSlotRegistry registry;
    const HbgExecutionSlotRegistration first = make_registration(7);
    ASSERT_EQ(publish_hbg_execution_slot_registration(&registry, &first, 1), HbgExecutionSlotRegistryStatus::Published);

    reset_hbg_execution_slot_registry(&registry);
    HbgExecutionSlotRegistration output{};
    EXPECT_EQ(acquire_hbg_execution_slot_registration(&registry, 1, &output), HbgExecutionSlotRegistryStatus::NotReady);

    const HbgExecutionSlotRegistration second = make_registration(8);
    EXPECT_EQ(
        publish_hbg_execution_slot_registration(&registry, &second, 1), HbgExecutionSlotRegistryStatus::Published
    );
}

TEST(HbgContextRegistry, SeparateContextOwnersKeepIndependentRegistrations) {
    HbgContextRegistry first_context;
    HbgContextRegistry second_context;
    ASSERT_TRUE(initialize_hbg_context_registry(&first_context, 7));
    ASSERT_TRUE(initialize_hbg_context_registry(&second_context, 8));
    ASSERT_TRUE(valid_hbg_context_registry_header(&first_context));
    ASSERT_TRUE(valid_hbg_context_registry_header(&second_context));

    const HbgExecutionSlotRegistration slot = make_registration(7);
    ASSERT_EQ(
        publish_hbg_execution_slot_registration(&first_context.execution_slot, &slot, 1),
        HbgExecutionSlotRegistryStatus::Published
    );
    HbgExecutionSlotRegistration acquired{};
    EXPECT_EQ(
        acquire_hbg_execution_slot_registration(&first_context.execution_slot, 1, &acquired),
        HbgExecutionSlotRegistryStatus::Acquired
    );
    EXPECT_EQ(
        acquire_hbg_execution_slot_registration(&second_context.execution_slot, 1, &acquired),
        HbgExecutionSlotRegistryStatus::NotReady
    );
}

TEST(HbgContextRegistry, RejectsNullZeroGenerationAndCorruptHeaders) {
    HbgContextRegistry registry;
    EXPECT_FALSE(initialize_hbg_context_registry(nullptr, 7));
    EXPECT_FALSE(initialize_hbg_context_registry(&registry, 0));
    EXPECT_FALSE(valid_hbg_context_registry_header(&registry));

    ASSERT_TRUE(initialize_hbg_context_registry(&registry, 9));
    ASSERT_TRUE(valid_hbg_context_registry_header(&registry));
    ++registry.abi_minor;
    EXPECT_FALSE(valid_hbg_context_registry_header(&registry));
}

}  // namespace

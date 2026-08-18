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
#include <limits>

#include <gtest/gtest.h>

#include "hbg_execution_slot.h"

namespace {

using simpler::hbg::build_hbg_execution_slot_registration;
using simpler::hbg::HBG_EXECUTION_SLOT_CAPACITY_FROZEN;
using simpler::hbg::HBG_EXECUTION_SLOT_REQUIRED_FLAGS;
using simpler::hbg::hbg_l1_launch_control;
using simpler::hbg::hbg_l1_launch_control_or_fallback;
using simpler::hbg::hbg_minimum_launch_blob_size;
using simpler::hbg::HbgExecutionSlotRegistration;
using simpler::hbg::HbgExecutionSlotRegistrationSpec;
using simpler::hbg::HbgExecutionSlotStatus;
using simpler::hbg::seal_hbg_execution_slot_registration;
using simpler::hbg::validate_hbg_execution_slot_registration;
using simpler::hbg::validate_hbg_launch_blob_size_for_slot;

HbgExecutionSlotRegistration make_registration() {
    HbgExecutionSlotRegistration registration;
    registration.device_id = 1;
    registration.binding.shared_memory_base = 0x100000;
    registration.binding.shared_memory_capacity = 0x2000;
    registration.binding.runtime_arena_base = 0x200000;
    registration.binding.runtime_arena_capacity = 0x3000;
    registration.binding.gm_heap_base = 0x300000;
    registration.binding.gm_heap_capacity = 0x4000;
    registration.binding.runtime_offset = 0x400;
    registration.binding.slot_generation = 7;
    registration.outer_runtime_base = 0x400000;
    registration.outer_runtime_size = 0x1000;
    registration.device_kernel_args_base = 0x500000;
    registration.device_kernel_args_size = 0x800;
    registration.binary_generation = 11;
    uint64_t minimum_size = 0;
    EXPECT_TRUE(hbg_minimum_launch_blob_size(registration.binding, &minimum_size));
    registration.max_launch_blob_size = minimum_size + 0x1000;
    return registration;
}

HbgExecutionSlotRegistrationSpec make_registration_spec() {
    const HbgExecutionSlotRegistration registration = make_registration();
    return HbgExecutionSlotRegistrationSpec{
        registration.device_id,
        registration.prelaunch_control_offset,
        registration.max_launch_blob_size,
        registration.binding,
        registration.outer_runtime_base,
        registration.outer_runtime_size,
        registration.device_kernel_args_base,
        registration.device_kernel_args_size,
        registration.binary_generation,
    };
}

TEST(HbgExecutionSlot, BuildsAndOwnsOneCompleteSealedRegistrationTransactionally) {
    const HbgExecutionSlotRegistrationSpec spec = make_registration_spec();
    HbgExecutionSlotRegistration registration{};

    ASSERT_EQ(build_hbg_execution_slot_registration(spec, &registration), HbgExecutionSlotStatus::Ok);
    EXPECT_EQ(registration.device_id, spec.device_id);
    EXPECT_EQ(registration.prelaunch_control_offset, spec.prelaunch_control_offset);
    EXPECT_EQ(registration.max_launch_blob_size, spec.max_launch_blob_size);
    EXPECT_TRUE(hbg_execution_binding_matches(registration.binding, spec.binding));
    EXPECT_EQ(registration.binding.slot_generation, spec.binding.slot_generation);
    EXPECT_EQ(registration.outer_runtime_base, spec.outer_runtime_base);
    EXPECT_EQ(registration.outer_runtime_size, spec.outer_runtime_size);
    EXPECT_EQ(registration.device_kernel_args_base, spec.device_kernel_args_base);
    EXPECT_EQ(registration.device_kernel_args_size, spec.device_kernel_args_size);
    EXPECT_EQ(registration.binary_generation, spec.binary_generation);
    EXPECT_NE(registration.registration_hash, 0u);
    EXPECT_EQ(validate_hbg_execution_slot_registration(&registration, spec.device_id), HbgExecutionSlotStatus::Ok);
    EXPECT_EQ(reinterpret_cast<uint64_t>(hbg_l1_launch_control(registration)), registration.outer_runtime_base);
}

TEST(HbgExecutionSlot, ResolvesPrepareTimeFallbackWhenRegistryTrustRootIsUnavailable) {
    HbgExecutionSlotRegistration registration = make_registration();
    ASSERT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::Ok);
    EXPECT_EQ(
        reinterpret_cast<uint64_t>(hbg_l1_launch_control_or_fallback(&registration, 0xabc000)),
        registration.outer_runtime_base + registration.prelaunch_control_offset
    );

    registration.magic = 0;
    EXPECT_EQ(reinterpret_cast<uint64_t>(hbg_l1_launch_control_or_fallback(&registration, 0xabc000)), 0xabc000u);

    registration = make_registration();
    ASSERT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::Ok);
    registration.registration_hash ^= 1;
    EXPECT_EQ(reinterpret_cast<uint64_t>(hbg_l1_launch_control_or_fallback(&registration, 0xabc000)), 0xabc000u);
    EXPECT_EQ(reinterpret_cast<uint64_t>(hbg_l1_launch_control_or_fallback(nullptr, 0xdef000)), 0xdef000u);
    EXPECT_EQ(hbg_l1_launch_control_or_fallback(nullptr, 0), nullptr);
}

TEST(HbgExecutionSlot, FailedBuildPreservesThePriorRegistrationOwner) {
    HbgExecutionSlotRegistration registration = make_registration();
    registration.registration_hash = 0xfeedfaceULL;
    const HbgExecutionSlotRegistration before = registration;

    HbgExecutionSlotRegistrationSpec spec = make_registration_spec();
    spec.binding.slot_generation = 0;
    EXPECT_EQ(build_hbg_execution_slot_registration(spec, &registration), HbgExecutionSlotStatus::InvalidGeneration);
    EXPECT_EQ(std::memcmp(&registration, &before, sizeof(registration)), 0);
    EXPECT_EQ(
        build_hbg_execution_slot_registration(make_registration_spec(), nullptr), HbgExecutionSlotStatus::NullArgument
    );
}

TEST(HbgExecutionSlot, SealsTheCompleteFrozenBindingAndDetectsMutation) {
    HbgExecutionSlotRegistration registration = make_registration();
    ASSERT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::Ok);
    EXPECT_NE(registration.registration_hash, 0u);
    EXPECT_EQ(validate_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::Ok);
    EXPECT_EQ(validate_hbg_execution_slot_registration(&registration, 0), HbgExecutionSlotStatus::DeviceMismatch);

    ++registration.binding.slot_generation;
    EXPECT_EQ(validate_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::HashMismatch);
}

TEST(HbgExecutionSlot, RequiresFrozenSerialOnlyAndEveryPersistentWindow) {
    HbgExecutionSlotRegistration registration = make_registration();
    registration.flags = HBG_EXECUTION_SLOT_CAPACITY_FROZEN;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidFlags);

    registration = make_registration();
    registration.device_id = -1;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration), HbgExecutionSlotStatus::InvalidDevice);

    registration = make_registration();
    registration.binding.slot_generation = 0;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidGeneration);

    registration = make_registration();
    registration.outer_runtime_base = 0;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidRuntimeWindow);

    registration = make_registration();
    registration.prelaunch_control_offset = 1;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidLaunchControl);

    registration = make_registration();
    registration.prelaunch_control_offset = static_cast<uint32_t>(registration.outer_runtime_size);
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidLaunchControl);

    registration = make_registration();
    registration.device_kernel_args_size = 0;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidKernelArgsWindow);

    registration = make_registration();
    registration.binary_generation = 0;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidBinaryGeneration);
}

TEST(HbgExecutionSlot, FreezesEnoughPackageCapacityWithoutEncodingARuntimeLimit) {
    HbgExecutionSlotRegistration registration = make_registration();
    uint64_t minimum_size = 0;
    ASSERT_TRUE(hbg_minimum_launch_blob_size(registration.binding, &minimum_size));

    registration.max_launch_blob_size = minimum_size - 1;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidPackageCapacity);

    registration = make_registration();
    registration.max_launch_blob_size = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidPackageCapacity);

    registration = make_registration();
    ASSERT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::Ok);
    EXPECT_EQ(
        validate_hbg_launch_blob_size_for_slot(registration, registration.max_launch_blob_size),
        HbgExecutionSlotStatus::Ok
    );
    EXPECT_EQ(validate_hbg_launch_blob_size_for_slot(registration, 0), HbgExecutionSlotStatus::InvalidPackageCapacity);
    EXPECT_EQ(
        validate_hbg_launch_blob_size_for_slot(registration, registration.max_launch_blob_size + 1),
        HbgExecutionSlotStatus::InvalidPackageCapacity
    );
}

TEST(HbgExecutionSlot, CapacityIncludesTheSerializerPaddingBetweenPristineRegions) {
    HbgExecutionSlotRegistration registration = make_registration();
    registration.binding.shared_memory_capacity = 15;
    registration.binding.runtime_arena_capacity = 9;
    uint64_t minimum_size = 0;
    ASSERT_TRUE(hbg_minimum_launch_blob_size(registration.binding, &minimum_size));

    // Two descriptors make the canonical header 240 bytes. The second source
    // begins at align_up(shared_memory_capacity, 8), exactly as the serializer
    // lays it out: 240 + 16 + 9.
    EXPECT_EQ(minimum_size, 265u);
}

TEST(HbgExecutionSlot, RejectsAliasingBetweenAllMutableAndPersistentRegions) {
    HbgExecutionSlotRegistration registration = make_registration();
    registration.binding.runtime_arena_base = registration.binding.shared_memory_base + 0x1000;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::AddressOverlap);

    registration = make_registration();
    registration.outer_runtime_base = registration.binding.gm_heap_base + 0x100;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::AddressOverlap);

    registration = make_registration();
    registration.device_kernel_args_base = registration.outer_runtime_base + registration.outer_runtime_size - 8;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::AddressOverlap);
}

TEST(HbgExecutionSlot, RejectsOverflowingWindowsAndMinimumPackageArithmetic) {
    HbgExecutionSlotRegistration registration = make_registration();
    registration.binding.shared_memory_base = std::numeric_limits<uint64_t>::max() - 7;
    registration.binding.shared_memory_capacity = 16;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidBinding);

    registration = make_registration();
    registration.binding.shared_memory_capacity = std::numeric_limits<uint64_t>::max();
    uint64_t minimum_size = 17;
    EXPECT_FALSE(hbg_minimum_launch_blob_size(registration.binding, &minimum_size));
    EXPECT_EQ(minimum_size, 17u);
}

TEST(HbgExecutionSlot, FailedSealDoesNotPublishANewHash) {
    HbgExecutionSlotRegistration registration = make_registration();
    registration.registration_hash = 0xabcdef;
    registration.prelaunch_control_offset = 1;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::InvalidLaunchControl);
    EXPECT_EQ(registration.registration_hash, 0xabcdefu);
}

}  // namespace

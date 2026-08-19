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
#include <vector>

#include <gtest/gtest.h>

#include "hbg_aicpu_invocation.h"
#include "hbg_launch_blob_builder.h"

namespace {

using namespace simpler::hbg;

struct FixtureData {
    std::vector<uint8_t> sm = std::vector<uint8_t>(64, 0x11);
    std::vector<uint8_t> arena = std::vector<uint8_t>(128, 0x22);
    std::vector<uint8_t> blob;
    HbgExecutionSlotRegistration slot{};

    FixtureData() {
        HbgExecutionBinding binding{};
        binding.shared_memory_base = 0x100000;
        binding.shared_memory_capacity = sm.size();
        binding.runtime_arena_base = 0x200000;
        binding.runtime_arena_capacity = arena.size();
        binding.runtime_offset = 8;
        binding.slot_generation = 17;

        HbgInvocationIdentity identity{};
        identity.callable_hash = 0x1234;
        identity.argument_snapshot_hash = 0x5678;
        identity.function_binding_hash = 0x9abc;
        identity.tensor_count = 2;
        identity.scalar_count = 1;
        identity.host_total_tasks = 5;
        identity.callable_id = 3;

        const uint32_t flags = HBG_REGION_REQUIRED | HBG_REGION_IMMUTABLE_SOURCE;
        EXPECT_EQ(
            build_hbg_launch_blob(
                binding, identity, 9,
                {{HbgLaunchRegionKind::SharedMemoryImage, flags, sm.data(), sm.size(), 0},
                 {HbgLaunchRegionKind::RuntimeArenaImage, flags, arena.data(), arena.size(), 0}},
                &blob
            ),
            HbgLaunchBlobStatus::Ok
        );
        auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
        header->inline_payload_addr = reinterpret_cast<uint64_t>(blob.data() + header->header_size);

        slot.device_id = 1;
        slot.max_launch_blob_size = blob.size();
        slot.binding = binding;
        slot.outer_runtime_base = 0x300000;
        slot.outer_runtime_size = 4096;
        slot.device_kernel_args_base = 0x400000;
        slot.device_kernel_args_size = 256;
        slot.binary_generation = 23;
        EXPECT_EQ(seal_hbg_execution_slot_registration(&slot, 1), HbgExecutionSlotStatus::Ok);
    }
};

TEST(HbgAicpuInvocation, AcquiresOnlyTheMatchingFixedPrefix) {
    FixtureData data;
    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    EXPECT_EQ(view.blob, data.blob.data());
    EXPECT_EQ(view.header.identity.callable_id, 3);
    EXPECT_EQ(view.slot.registration_hash, data.slot.registration_hash);
}

TEST(HbgAicpuInvocation, ExtractsTheTaskOwnedFaultMarkerWithoutTypedUnalignedAccess) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::BeforeDispatch);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    EXPECT_EQ(hbg_aicpu_fault_stage(&view), HbgL1FaultStage::BeforeDispatch);

    regions[0].reserved = 0;
    EXPECT_EQ(hbg_aicpu_fault_stage(&view), HbgL1FaultStage::None);
}

TEST(HbgAicpuInvocation, AuthenticatesTheFaultMarkerAgainstTheCompletePackage) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::SchedulerInit);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    HbgL1FaultStage stage = HbgL1FaultStage::None;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(stage, HbgL1FaultStage::SchedulerInit);

    ++header->plan_hash;
    stage = HbgL1FaultStage::BeforeDispatch;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::HashMismatch);
    EXPECT_EQ(stage, HbgL1FaultStage::BeforeDispatch);
}

TEST(HbgAicpuInvocation, AuthenticatesSchedulerAssignmentFaultBeforeExecutorInit) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::SchedulerAssign);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    HbgL1FaultStage stage = HbgL1FaultStage::None;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(stage, HbgL1FaultStage::SchedulerAssign);
}

TEST(HbgAicpuInvocation, AuthenticatesSchedulerDispatchFaultForTheRestoredGeneration) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::SchedulerDispatch);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    HbgL1FaultStage stage = HbgL1FaultStage::None;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(stage, HbgL1FaultStage::SchedulerDispatch);
}

TEST(HbgAicpuInvocation, AuthenticatesPlatformBridgeFaultBeforeGenerationEntry) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::PlatformBridge);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    HbgL1FaultStage stage = HbgL1FaultStage::None;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(stage, HbgL1FaultStage::PlatformBridge);
}

TEST(HbgAicpuInvocation, AuthenticatesAffinityInputFaultBeforeTheGateBarrier) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::AffinityInputs);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    HbgL1FaultStage stage = HbgL1FaultStage::None;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(stage, HbgL1FaultStage::AffinityInputs);
}

TEST(HbgAicpuInvocation, AuthenticatesKernelArgsRuntimeFaultBeforePlatformEntry) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::KernelArgsRuntime);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    HbgL1FaultStage stage = HbgL1FaultStage::None;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(stage, HbgL1FaultStage::KernelArgsRuntime);
}

TEST(HbgAicpuInvocation, AuthenticatesPhysicalCoreMappingFaultDuringHandshake) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::PhysicalCoreMapping);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    HbgL1FaultStage stage = HbgL1FaultStage::None;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(stage, HbgL1FaultStage::PhysicalCoreMapping);
}

TEST(HbgAicpuInvocation, AuthenticatesPhysicalCoreIdFaultDuringHandshake) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::PhysicalCoreId);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    HbgL1FaultStage stage = HbgL1FaultStage::None;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(stage, HbgL1FaultStage::PhysicalCoreId);
}

TEST(HbgAicpuInvocation, AuthenticatesSlotFallbackControlFaultBeforeGeneration) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::SlotFallbackControl);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );

    HbgAicpuInvocationView view{};
    ASSERT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &view), HbgAicpuInvocationStatus::Ok);
    HbgL1FaultStage stage = HbgL1FaultStage::None;
    EXPECT_EQ(authenticate_hbg_aicpu_fault_stage(&view, &stage), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(stage, HbgL1FaultStage::SlotFallbackControl);
}

TEST(HbgAicpuInvocation, AcceptsSelfContainedCallableIdentityBeyondTheLegacySlotCount) {
    FixtureData data;
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    header->identity.callable_id = 65;
    header->plan_hash = hbg_plan_hash(
        header->identity, hbg_launch_regions(header), header->region_count, hbg_inline_payload(header),
        header->inline_payload_size
    );

    HbgAicpuInvocationView output{};
    EXPECT_EQ(make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &output), HbgAicpuInvocationStatus::Ok);
    EXPECT_EQ(output.header.identity.callable_id, 65);
}

TEST(HbgAicpuInvocation, CapacityMismatchDoesNotPublishOutput) {
    FixtureData data;
    HbgAicpuInvocationView output{};
    output.blob = reinterpret_cast<void *>(0x5555);
    const HbgAicpuInvocationView before = output;

    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(data.blob.data());
    ++header->total_size;
    ++header->inline_payload_size;
    EXPECT_EQ(
        make_hbg_aicpu_invocation_view(data.blob.data(), data.slot, &output),
        HbgAicpuInvocationStatus::InvalidPackageSize
    );
    EXPECT_EQ(std::memcmp(&output, &before, sizeof(output)), 0);
}

}  // namespace

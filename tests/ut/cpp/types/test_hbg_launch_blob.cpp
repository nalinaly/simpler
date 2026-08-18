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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "hbg_launch_blob.h"
#include "hbg_launch_blob_builder.h"
#include "hbg_graph_plan.h"
#include "hbg_restore.h"

namespace {

using simpler::hbg::build_hbg_graph_plan;
using simpler::hbg::build_hbg_launch_blob;
using simpler::hbg::hbg_inline_payload;
using simpler::hbg::hbg_l1_decode_fault_marker;
using simpler::hbg::hbg_l1_encode_fault_marker;
using simpler::hbg::hbg_l1_fault_stage;
using simpler::hbg::hbg_launch_regions;
using simpler::hbg::HBG_LAUNCH_TEST_FAULT_INJECTION;
using simpler::hbg::hbg_plan_hash;
using simpler::hbg::HBG_REGION_IMMUTABLE_SOURCE;
using simpler::hbg::HBG_REGION_REQUIRED;
using simpler::hbg::HbgExecutionBinding;
using simpler::hbg::HbgExecutionSlotRegistration;
using simpler::hbg::HbgExecutionSlotStatus;
using simpler::hbg::HbgGraphPlan;
using simpler::hbg::HbgHostRegionInput;
using simpler::hbg::HbgInvocationIdentity;
using simpler::hbg::HbgL1FaultStage;
using simpler::hbg::HbgLaunchBlobAddressMode;
using simpler::hbg::HbgLaunchBlobHeader;
using simpler::hbg::HbgLaunchBlobStatus;
using simpler::hbg::HbgLaunchRegion;
using simpler::hbg::HbgLaunchRegionKind;
using simpler::hbg::HbgRestoreCommit;
using simpler::hbg::HbgRestoreOps;
using simpler::hbg::HbgRestoreStatus;
using simpler::hbg::make_hbg_launch_placeholder;
using simpler::hbg::restore_hbg_launch_blob;
using simpler::hbg::seal_hbg_execution_slot_registration;
using simpler::hbg::validate_hbg_launch_blob;
using simpler::host_args::HostArgsLaunchStatus;
using simpler::host_args::HostArgsPlaceholder;
using simpler::host_args::validate_host_args_launch_layout;

constexpr uint32_t kRegionFlags = HBG_REGION_REQUIRED | HBG_REGION_IMMUTABLE_SOURCE;

struct Sources {
    std::array<uint8_t, 16> sm{};
    std::array<uint8_t, 24> arena{};
};

struct RestoreHarness {
    std::array<uint8_t, 16> working_sm{};
    std::array<uint8_t, 24> working_arena{};
    std::array<uint8_t, sizeof(simpler::hbg::HbgL1LaunchControl)> outer_runtime{};
    std::array<uint8_t, 16> device_kernel_args{};
    int copy_calls{0};
    int publish_calls{0};
    int fail_copy_at{-1};
    int fail_publish_at{-1};
};

struct MultiLineRestoreHarness {
    std::vector<uint8_t> working_sm;
    std::vector<uint8_t> working_arena;
    std::array<uint8_t, sizeof(simpler::hbg::HbgL1LaunchControl)> outer_runtime{};
    std::array<uint8_t, 16> device_kernel_args{};
    std::array<size_t, 4> published_sizes{};
    int copy_calls{0};
    int publish_calls{0};
};

int restore_copy(void *context, HbgLaunchRegionKind, void *destination, const void *source, size_t size) noexcept {
    auto *harness = static_cast<RestoreHarness *>(context);
    const int call = harness->copy_calls++;
    if (call == harness->fail_copy_at) return -91;
    std::memcpy(destination, source, size);
    return 0;
}

int restore_publish(void *context, HbgLaunchRegionKind, const void *, size_t) noexcept {
    auto *harness = static_cast<RestoreHarness *>(context);
    const int call = harness->publish_calls++;
    return call == harness->fail_publish_at ? -92 : 0;
}

int multi_line_restore_copy(
    void *context, HbgLaunchRegionKind, void *destination, const void *source, size_t size
) noexcept {
    auto *harness = static_cast<MultiLineRestoreHarness *>(context);
    ++harness->copy_calls;
    std::memcpy(destination, source, size);
    return 0;
}

int multi_line_restore_publish(void *context, HbgLaunchRegionKind, const void *, size_t size) noexcept {
    auto *harness = static_cast<MultiLineRestoreHarness *>(context);
    if (harness->publish_calls < static_cast<int>(harness->published_sizes.size())) {
        harness->published_sizes[static_cast<size_t>(harness->publish_calls)] = size;
    }
    ++harness->publish_calls;
    return 0;
}

HbgExecutionBinding make_restore_binding(const RestoreHarness &harness) {
    HbgExecutionBinding binding;
    binding.shared_memory_base = reinterpret_cast<uint64_t>(harness.working_sm.data());
    binding.shared_memory_capacity = harness.working_sm.size();
    binding.runtime_arena_base = reinterpret_cast<uint64_t>(harness.working_arena.data());
    binding.runtime_arena_capacity = harness.working_arena.size();
    binding.runtime_offset = 8;
    binding.slot_generation = 17;
    return binding;
}

HbgExecutionSlotRegistration make_restore_registration(
    const RestoreHarness &harness, const HbgExecutionBinding &binding, size_t max_launch_blob_size
) {
    HbgExecutionSlotRegistration registration;
    registration.device_id = 1;
    registration.max_launch_blob_size = max_launch_blob_size;
    registration.binding = binding;
    registration.outer_runtime_base = reinterpret_cast<uint64_t>(harness.outer_runtime.data());
    registration.outer_runtime_size = harness.outer_runtime.size();
    registration.device_kernel_args_base = reinterpret_cast<uint64_t>(harness.device_kernel_args.data());
    registration.device_kernel_args_size = harness.device_kernel_args.size();
    registration.binary_generation = 23;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::Ok);
    return registration;
}

HbgExecutionBinding make_binding(const Sources &sources, bool with_heap = false) {
    HbgExecutionBinding binding;
    binding.shared_memory_base = 0x100000;
    binding.shared_memory_capacity = sources.sm.size();
    binding.runtime_arena_base = 0x200000;
    binding.runtime_arena_capacity = sources.arena.size();
    binding.runtime_offset = 8;
    binding.slot_generation = 9;
    if (with_heap) {
        binding.gm_heap_base = 0x300000;
        binding.gm_heap_capacity = 64;
    }
    return binding;
}

HbgExecutionBinding make_multi_line_binding(const MultiLineRestoreHarness &harness) {
    HbgExecutionBinding binding;
    binding.shared_memory_base = reinterpret_cast<uint64_t>(harness.working_sm.data());
    binding.shared_memory_capacity = harness.working_sm.size();
    binding.runtime_arena_base = reinterpret_cast<uint64_t>(harness.working_arena.data());
    binding.runtime_arena_capacity = harness.working_arena.size();
    binding.runtime_offset = 64;
    binding.slot_generation = 29;
    return binding;
}

HbgExecutionSlotRegistration make_multi_line_registration(
    const MultiLineRestoreHarness &harness, const HbgExecutionBinding &binding, size_t max_launch_blob_size
) {
    HbgExecutionSlotRegistration registration;
    registration.device_id = 1;
    registration.max_launch_blob_size = max_launch_blob_size;
    registration.binding = binding;
    registration.outer_runtime_base = reinterpret_cast<uint64_t>(harness.outer_runtime.data());
    registration.outer_runtime_size = harness.outer_runtime.size();
    registration.device_kernel_args_base = reinterpret_cast<uint64_t>(harness.device_kernel_args.data());
    registration.device_kernel_args_size = harness.device_kernel_args.size();
    registration.binary_generation = 31;
    EXPECT_EQ(seal_hbg_execution_slot_registration(&registration, 1), HbgExecutionSlotStatus::Ok);
    return registration;
}

HbgInvocationIdentity make_identity() {
    HbgInvocationIdentity identity;
    identity.callable_hash = 0x1111222233334444ULL;
    identity.argument_snapshot_hash = 0x5555666677778888ULL;
    identity.function_binding_hash = 0x9999aaaabbbbccccULL;
    identity.tensor_count = 2;
    identity.scalar_count = 1;
    identity.host_total_tasks = 3;
    identity.callable_id = 4;
    return identity;
}

std::vector<HbgHostRegionInput> make_inputs(const Sources &sources) {
    return {
        {HbgLaunchRegionKind::SharedMemoryImage, kRegionFlags, sources.sm.data(), sources.sm.size(), 0},
        {HbgLaunchRegionKind::RuntimeArenaImage, kRegionFlags, sources.arena.data(), sources.arena.size(), 0},
    };
}

std::vector<uint8_t> make_restore_blob(
    const Sources &sources, const HbgExecutionBinding &binding, const HbgInvocationIdentity &identity,
    uint64_t plan_generation
) {
    std::vector<uint8_t> blob;
    EXPECT_EQ(
        build_hbg_launch_blob(binding, identity, plan_generation, make_inputs(sources), &blob), HbgLaunchBlobStatus::Ok
    );
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    header->inline_payload_addr = reinterpret_cast<uint64_t>(blob.data()) + header->header_size;
    return blob;
}

std::vector<uint8_t> make_blob(Sources *sources = nullptr) {
    static Sources fallback;
    Sources &resolved = sources == nullptr ? fallback : *sources;
    for (size_t i = 0; i < resolved.sm.size(); ++i)
        resolved.sm[i] = static_cast<uint8_t>(0x10 + i);
    for (size_t i = 0; i < resolved.arena.size(); ++i)
        resolved.arena[i] = static_cast<uint8_t>(0x80 + i);

    std::vector<uint8_t> blob;
    EXPECT_EQ(
        build_hbg_launch_blob(make_binding(resolved), make_identity(), 7, make_inputs(resolved), &blob),
        HbgLaunchBlobStatus::Ok
    );
    return blob;
}

void set_fault_marker(std::vector<uint8_t> &blob, HbgL1FaultStage stage, size_t region_index = 0) {
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    regions[region_index].reserved = hbg_l1_encode_fault_marker(stage);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );
}

TEST(HbgLaunchBlob, SerializesAnIndependentCanonicalSnapshot) {
    Sources sources;
    std::vector<uint8_t> blob = make_blob(&sources);
    ASSERT_FALSE(blob.empty());

    const auto *header = reinterpret_cast<const HbgLaunchBlobHeader *>(blob.data());
    const HbgLaunchRegion *regions = hbg_launch_regions(header);
    const uint8_t *payload = hbg_inline_payload(header);
    ASSERT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::Ok
    );
    EXPECT_EQ(header->region_count, 2u);
    EXPECT_EQ(header->plan_generation, 7u);
    EXPECT_EQ(header->binding.slot_generation, 9u);
    EXPECT_EQ(header->inline_payload_addr, 0u);
    EXPECT_EQ(regions[0].size, sources.sm.size());
    EXPECT_EQ(regions[1].size, sources.arena.size());
    EXPECT_EQ(std::memcmp(payload + regions[0].source_offset, sources.sm.data(), sources.sm.size()), 0);
    EXPECT_EQ(std::memcmp(payload + regions[1].source_offset, sources.arena.data(), sources.arena.size()), 0);

    const uint8_t snapshotted_first_byte = payload[regions[0].source_offset];
    sources.sm[0] ^= 0xff;
    EXPECT_EQ(payload[regions[0].source_offset], snapshotted_first_byte);
}

TEST(HbgLaunchBlob, TestFaultMarkerIsTaskLocalHashedAndStrictlyValidated) {
    std::vector<uint8_t> blob = make_blob();
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    EXPECT_EQ(hbg_l1_fault_stage(header), HbgL1FaultStage::None);

    set_fault_marker(blob, HbgL1FaultStage::RestoreCopy);
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::Ok
    );
    EXPECT_EQ(hbg_l1_fault_stage(header), HbgL1FaultStage::RestoreCopy);
    EXPECT_EQ(
        hbg_l1_decode_fault_marker(hbg_l1_encode_fault_marker(HbgL1FaultStage::RuntimeDestroy)),
        HbgL1FaultStage::RuntimeDestroy
    );
    EXPECT_EQ(
        hbg_l1_decode_fault_marker(hbg_l1_encode_fault_marker(HbgL1FaultStage::SchedulerInit)),
        HbgL1FaultStage::SchedulerInit
    );
    EXPECT_EQ(
        hbg_l1_decode_fault_marker(hbg_l1_encode_fault_marker(HbgL1FaultStage::SchedulerAssign)),
        HbgL1FaultStage::SchedulerAssign
    );
    EXPECT_EQ(
        hbg_l1_decode_fault_marker(hbg_l1_encode_fault_marker(HbgL1FaultStage::SchedulerDispatch)),
        HbgL1FaultStage::SchedulerDispatch
    );
    EXPECT_EQ(
        hbg_l1_decode_fault_marker(hbg_l1_encode_fault_marker(HbgL1FaultStage::PlatformBridge)),
        HbgL1FaultStage::PlatformBridge
    );
    EXPECT_EQ(
        hbg_l1_decode_fault_marker(hbg_l1_encode_fault_marker(HbgL1FaultStage::AffinityInputs)),
        HbgL1FaultStage::AffinityInputs
    );
    EXPECT_EQ(
        hbg_l1_decode_fault_marker(hbg_l1_encode_fault_marker(HbgL1FaultStage::KernelArgsRuntime)),
        HbgL1FaultStage::KernelArgsRuntime
    );
    EXPECT_EQ(
        hbg_l1_decode_fault_marker(hbg_l1_encode_fault_marker(HbgL1FaultStage::PhysicalCoreMapping)),
        HbgL1FaultStage::PhysicalCoreMapping
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidRegion
    );

    blob = make_blob();
    set_fault_marker(blob, HbgL1FaultStage::RestorePublish, 1);
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidRegion
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    regions[0].reserved = hbg_l1_encode_fault_marker(HbgL1FaultStage::BeforeDispatch);
    header->plan_hash = hbg_plan_hash(
        header->identity, regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidRegion
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    auto *bad_regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    header->flags |= HBG_LAUNCH_TEST_FAULT_INJECTION;
    bad_regions[0].reserved = static_cast<uint64_t>(simpler::hbg::HBG_L1_FAULT_MARKER_MAGIC) << 32U;
    header->plan_hash = hbg_plan_hash(
        header->identity, bad_regions, header->region_count, hbg_inline_payload(header), header->inline_payload_size
    );
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidRegion
    );
}

TEST(HbgGraphPlan, OwnsCanonicalBytesAndProducesFreshWritableTaskSnapshots) {
    Sources sources;
    sources.sm.fill(0x31);
    sources.arena.fill(0x72);
    const HbgExecutionBinding binding = make_binding(sources);
    const HbgInvocationIdentity identity = make_identity();
    std::unique_ptr<const HbgGraphPlan> plan;

    ASSERT_EQ(build_hbg_graph_plan(binding, identity, 19, make_inputs(sources), &plan), HbgLaunchBlobStatus::Ok);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->binding().slot_generation, binding.slot_generation);
    EXPECT_EQ(plan->identity().host_total_tasks, identity.host_total_tasks);
    EXPECT_EQ(plan->plan_generation(), 19u);
    EXPECT_NE(plan->plan_hash(), 0u);

    sources.sm.fill(0xee);
    sources.arena.fill(0xdd);
    std::vector<uint8_t> first;
    std::vector<uint8_t> second;
    ASSERT_EQ(plan->serialize(&first), HbgLaunchBlobStatus::Ok);
    ASSERT_EQ(plan->serialize(&second), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(first, second);

    const auto *header = reinterpret_cast<const HbgLaunchBlobHeader *>(first.data());
    const HbgLaunchRegion *regions = hbg_launch_regions(header);
    const uint8_t *payload = hbg_inline_payload(header);
    EXPECT_EQ(header->plan_hash, plan->plan_hash());
    EXPECT_EQ(payload[regions[0].source_offset], 0x31);
    EXPECT_EQ(payload[regions[1].source_offset], 0x72);

    HostArgsPlaceholder placeholder{};
    ASSERT_EQ(make_hbg_launch_placeholder(first.data(), first.size(), &placeholder), HbgLaunchBlobStatus::Ok);
    const uint64_t patched = reinterpret_cast<uint64_t>(first.data()) + placeholder.data_offset;
    std::memcpy(first.data() + placeholder.addr_offset, &patched, sizeof(patched));
    EXPECT_NE(first, second);

    std::vector<uint8_t> third;
    ASSERT_EQ(plan->serialize(&third), HbgLaunchBlobStatus::Ok);
    EXPECT_EQ(third, second);
}

TEST(HbgGraphPlan, FailedBuildDoesNotReplaceAnExistingOwner) {
    Sources sources;
    std::unique_ptr<const HbgGraphPlan> plan;
    ASSERT_EQ(
        build_hbg_graph_plan(make_binding(sources), make_identity(), 23, make_inputs(sources), &plan),
        HbgLaunchBlobStatus::Ok
    );
    const HbgGraphPlan *const original = plan.get();
    const uint64_t original_hash = plan->plan_hash();

    std::vector<HbgHostRegionInput> invalid = make_inputs(sources);
    invalid[0].data = nullptr;
    EXPECT_EQ(
        build_hbg_graph_plan(make_binding(sources), make_identity(), 24, invalid, &plan),
        HbgLaunchBlobStatus::InvalidRegion
    );
    EXPECT_EQ(plan.get(), original);
    EXPECT_EQ(plan->plan_hash(), original_hash);
}

TEST(HbgLaunchBlob, DistinguishesHostAndRuntimePatchedPointerStates) {
    std::vector<uint8_t> blob = make_blob();
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    ASSERT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::Ok
    );

    header->inline_payload_addr = reinterpret_cast<uint64_t>(blob.data() + header->header_size);
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::DevicePatched),
        HbgLaunchBlobStatus::Ok
    );
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::Either), HbgLaunchBlobStatus::Ok
    );
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidHeader
    );

    ++header->inline_payload_addr;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::DevicePatched),
        HbgLaunchBlobStatus::InvalidHeader
    );
}

TEST(HbgLaunchBlob, BuildsOnePlaceholderWithoutMutatingCanonicalBytes) {
    std::vector<uint8_t> blob = make_blob();
    const std::vector<uint8_t> canonical = blob;
    const auto *header = reinterpret_cast<const HbgLaunchBlobHeader *>(blob.data());
    const HbgExecutionBinding expected_binding = header->binding;
    const HbgInvocationIdentity expected_identity = header->identity;
    HostArgsPlaceholder placeholder{0xffffffffU, 0xffffffffU};

    ASSERT_EQ(
        make_hbg_launch_placeholder(blob.data(), blob.size(), &placeholder, &expected_binding, &expected_identity),
        HbgLaunchBlobStatus::Ok
    );
    EXPECT_EQ(placeholder.addr_offset, offsetof(HbgLaunchBlobHeader, inline_payload_addr));
    EXPECT_EQ(placeholder.data_offset, header->header_size);
    EXPECT_EQ(blob, canonical);

    std::vector<uint8_t> runtime_owned_copy = blob;
    const uint64_t patched_payload_addr =
        reinterpret_cast<uint64_t>(runtime_owned_copy.data()) + placeholder.data_offset;
    std::memcpy(
        runtime_owned_copy.data() + placeholder.addr_offset, &patched_payload_addr, sizeof(patched_payload_addr)
    );
    EXPECT_EQ(
        validate_hbg_launch_blob(
            runtime_owned_copy.data(), runtime_owned_copy.size(), HbgLaunchBlobAddressMode::DevicePatched,
            &expected_binding, &expected_identity
        ),
        HbgLaunchBlobStatus::Ok
    );
}

TEST(HbgLaunchBlob, PlaceholderPreparationRejectsWrongIdentityAndPreservesOutput) {
    std::vector<uint8_t> blob = make_blob();
    const auto *header = reinterpret_cast<const HbgLaunchBlobHeader *>(blob.data());
    HbgInvocationIdentity expected_identity = header->identity;
    ++expected_identity.argument_snapshot_hash;
    const HostArgsPlaceholder sentinel{24, 32};
    HostArgsPlaceholder output = sentinel;

    EXPECT_EQ(
        make_hbg_launch_placeholder(blob.data(), blob.size(), &output, nullptr, &expected_identity),
        HbgLaunchBlobStatus::IdentityMismatch
    );
    EXPECT_EQ(output.addr_offset, sentinel.addr_offset);
    EXPECT_EQ(output.data_offset, sentinel.data_offset);
    EXPECT_EQ(make_hbg_launch_placeholder(blob.data(), blob.size(), nullptr), HbgLaunchBlobStatus::NullArgument);
}

TEST(HbgLaunchBlob, GenericHostArgsLayoutRejectsEveryLossyOrUnsafePlaceholder) {
    alignas(8) std::array<uint8_t, 64> args{};
    HostArgsPlaceholder placeholders[2]{{8, 32}, {16, 40}};
    EXPECT_EQ(validate_host_args_launch_layout(args.data(), args.size(), placeholders, 2), HostArgsLaunchStatus::Ok);
    EXPECT_EQ(
        validate_host_args_launch_layout(nullptr, args.size(), placeholders, 2), HostArgsLaunchStatus::NullArguments
    );
    EXPECT_EQ(validate_host_args_launch_layout(args.data(), 0, placeholders, 2), HostArgsLaunchStatus::EmptyArguments);
    EXPECT_EQ(
        validate_host_args_launch_layout(
            args.data(), static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1, placeholders, 2
        ),
        HostArgsLaunchStatus::ArgumentsTooLarge
    );
    EXPECT_EQ(
        validate_host_args_launch_layout(args.data(), args.size(), nullptr, 1),
        HostArgsLaunchStatus::PlaceholderPointerMismatch
    );
    EXPECT_EQ(
        validate_host_args_launch_layout(
            args.data(), args.size(), placeholders, static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1
        ),
        HostArgsLaunchStatus::TooManyPlaceholders
    );

    placeholders[0] = {4, 32};
    EXPECT_EQ(
        validate_host_args_launch_layout(args.data(), args.size(), placeholders, 1),
        HostArgsLaunchStatus::MisalignedAddressField
    );
    placeholders[0] = {64, 32};
    EXPECT_EQ(
        validate_host_args_launch_layout(args.data(), args.size(), placeholders, 1),
        HostArgsLaunchStatus::AddressFieldOutOfBounds
    );
    placeholders[0] = {8, 64};
    EXPECT_EQ(
        validate_host_args_launch_layout(args.data(), args.size(), placeholders, 1),
        HostArgsLaunchStatus::DataOffsetOutOfBounds
    );
    placeholders[0] = {8, 32};
    placeholders[1] = {8, 40};
    EXPECT_EQ(
        validate_host_args_launch_layout(args.data(), args.size(), placeholders, 2),
        HostArgsLaunchStatus::OverlappingAddressFields
    );
}

TEST(HbgLaunchBlob, RejectsTruncationHeaderAndGenerationCorruption) {
    std::vector<uint8_t> blob = make_blob();
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());

    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size() - 1, HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidHeader
    );
    header->magic ^= 1;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidMagic
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    header->plan_generation = 0;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidGeneration
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    header->identity.callable_hash = 0;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidIdentity
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    header->identity.argument_snapshot_hash = 0;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidIdentity
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    header->identity.host_total_tasks = -1;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidIdentity
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    header->identity.callable_id = MAX_REGISTERED_CALLABLE_IDS;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidIdentity
    );
}

TEST(HbgLaunchBlob, RejectsPartialOrOverlappingRestoreImagesBeforeHashing) {
    std::vector<uint8_t> blob = make_blob();
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    auto *regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    --regions[0].size;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::InvalidRegion
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    regions[1].source_offset = regions[0].source_offset;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::Overlap
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    regions = const_cast<HbgLaunchRegion *>(hbg_launch_regions(header));
    regions[1].destination_offset = header->binding.runtime_arena_capacity;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::OutOfBounds
    );
}

TEST(HbgLaunchBlob, HashExcludesPlaceholderButCoversIdentityDescriptorsAndPayload) {
    std::vector<uint8_t> blob = make_blob();
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    uint8_t *payload = blob.data() + header->header_size;
    payload[0] ^= 1;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::HashMismatch
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    header->inline_payload_addr = reinterpret_cast<uint64_t>(blob.data() + header->header_size);
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::DevicePatched),
        HbgLaunchBlobStatus::Ok
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    ++header->identity.argument_snapshot_hash;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::HashMismatch
    );

    blob = make_blob();
    header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    ++header->identity.host_total_tasks;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched),
        HbgLaunchBlobStatus::HashMismatch
    );
}

TEST(HbgLaunchBlob, ExpectedExecutionBindingRejectsStaleSlotAndAddress) {
    Sources sources;
    std::vector<uint8_t> blob = make_blob(&sources);
    const auto *header = reinterpret_cast<const HbgLaunchBlobHeader *>(blob.data());
    HbgExecutionBinding expected = header->binding;
    ASSERT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched, &expected),
        HbgLaunchBlobStatus::Ok
    );

    ++expected.slot_generation;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched, &expected),
        HbgLaunchBlobStatus::InvalidGeneration
    );

    expected = header->binding;
    expected.runtime_arena_base += 0x1000;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched, &expected),
        HbgLaunchBlobStatus::InvalidBinding
    );
}

TEST(HbgLaunchBlob, ExpectedInvocationIdentityRejectsAnotherCallSnapshot) {
    std::vector<uint8_t> blob = make_blob();
    const auto *header = reinterpret_cast<const HbgLaunchBlobHeader *>(blob.data());
    HbgInvocationIdentity expected = header->identity;
    ASSERT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched, nullptr, &expected),
        HbgLaunchBlobStatus::Ok
    );

    ++expected.argument_snapshot_hash;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched, nullptr, &expected),
        HbgLaunchBlobStatus::IdentityMismatch
    );

    expected = header->identity;
    ++expected.function_binding_hash;
    EXPECT_EQ(
        validate_hbg_launch_blob(blob.data(), blob.size(), HbgLaunchBlobAddressMode::HostUnpatched, nullptr, &expected),
        HbgLaunchBlobStatus::IdentityMismatch
    );
}

TEST(HbgLaunchBlob, SupportsNonOverlappingHeapInitializersAndRejectsOverlap) {
    Sources sources;
    std::array<uint8_t, 8> first{};
    std::array<uint8_t, 8> second{};
    std::vector<HbgHostRegionInput> inputs = make_inputs(sources);
    inputs.push_back({HbgLaunchRegionKind::GmHeapInitializer, kRegionFlags, first.data(), first.size(), 0});
    inputs.push_back({HbgLaunchRegionKind::GmHeapInitializer, kRegionFlags, second.data(), second.size(), 16});

    std::vector<uint8_t> blob;
    EXPECT_EQ(
        build_hbg_launch_blob(make_binding(sources, true), make_identity(), 3, inputs, &blob), HbgLaunchBlobStatus::Ok
    );
    const std::vector<uint8_t> before = blob;

    inputs.back().destination_offset = 4;
    EXPECT_EQ(
        build_hbg_launch_blob(make_binding(sources, true), make_identity(), 3, inputs, &blob),
        HbgLaunchBlobStatus::Overlap
    );
    EXPECT_EQ(blob, before);
}

TEST(HbgLaunchBlob, BuilderRejectsMissingFullImagesAndInvalidInputs) {
    Sources sources;
    std::vector<uint8_t> blob;
    std::vector<HbgHostRegionInput> inputs = make_inputs(sources);
    inputs.pop_back();
    EXPECT_EQ(
        build_hbg_launch_blob(make_binding(sources), make_identity(), 1, inputs, &blob),
        HbgLaunchBlobStatus::InvalidHeader
    );

    inputs = make_inputs(sources);
    inputs[0].data = nullptr;
    EXPECT_EQ(
        build_hbg_launch_blob(make_binding(sources), make_identity(), 1, inputs, &blob),
        HbgLaunchBlobStatus::InvalidRegion
    );

    inputs = make_inputs(sources);
    EXPECT_EQ(
        build_hbg_launch_blob(make_binding(sources), make_identity(), 0, inputs, &blob),
        HbgLaunchBlobStatus::InvalidGeneration
    );
    HbgInvocationIdentity invalid_identity = make_identity();
    invalid_identity.scalar_count = CHIP_MAX_SCALAR_ARGS + 1;
    EXPECT_EQ(
        build_hbg_launch_blob(make_binding(sources), invalid_identity, 1, inputs, &blob),
        HbgLaunchBlobStatus::InvalidIdentity
    );
    EXPECT_EQ(
        build_hbg_launch_blob(make_binding(sources), make_identity(), 1, inputs, nullptr),
        HbgLaunchBlobStatus::NullArgument
    );
}

TEST(HbgLaunchBlob, FailedBuildPreservesThePreviousCanonicalBlob) {
    Sources sources;
    std::vector<uint8_t> blob = make_blob(&sources);
    const std::vector<uint8_t> before = blob;
    std::vector<HbgHostRegionInput> invalid = make_inputs(sources);
    invalid[0].data = nullptr;

    EXPECT_EQ(
        build_hbg_launch_blob(make_binding(sources), make_identity(), 11, invalid, &blob),
        HbgLaunchBlobStatus::InvalidRegion
    );
    EXPECT_EQ(blob, before);
}

TEST(HbgLaunchBlob, RestoresThePristineWorkingImageOnEveryReplay) {
    Sources sources;
    for (size_t i = 0; i < sources.sm.size(); ++i)
        sources.sm[i] = static_cast<uint8_t>(0x20 + i);
    for (size_t i = 0; i < sources.arena.size(); ++i)
        sources.arena[i] = static_cast<uint8_t>(0x60 + i);
    RestoreHarness harness;
    const HbgExecutionBinding binding = make_restore_binding(harness);
    const HbgInvocationIdentity identity = make_identity();
    std::vector<uint8_t> blob = make_restore_blob(sources, binding, identity, 31);
    const HbgExecutionSlotRegistration registration = make_restore_registration(harness, binding, blob.size());
    HbgRestoreCommit commit{};
    const HbgRestoreOps ops{&harness, restore_copy, restore_publish};

    harness.working_sm.fill(0xee);
    harness.working_arena.fill(0xdd);
    auto result = restore_hbg_launch_blob(blob.data(), blob.size(), registration, identity, ops, &commit);
    ASSERT_EQ(result.status, HbgRestoreStatus::Ok);
    EXPECT_EQ(harness.working_sm, sources.sm);
    EXPECT_EQ(harness.working_arena, sources.arena);
    EXPECT_EQ(commit.slot_generation, binding.slot_generation);
    EXPECT_EQ(commit.plan_generation, 31u);
    EXPECT_EQ(commit.identity.argument_snapshot_hash, identity.argument_snapshot_hash);

    harness.working_sm.fill(0x11);
    harness.working_arena.fill(0x22);
    result = restore_hbg_launch_blob(blob.data(), blob.size(), registration, identity, ops, &commit);
    ASSERT_EQ(result.status, HbgRestoreStatus::Ok);
    EXPECT_EQ(harness.working_sm, sources.sm);
    EXPECT_EQ(harness.working_arena, sources.arena);
    EXPECT_EQ(harness.copy_calls, 4);
    EXPECT_EQ(harness.publish_calls, 4);
}

TEST(HbgLaunchBlob, EveryReplayRestoresFirstMiddleAndTailAcrossMultipleCacheLines) {
    constexpr size_t sm_size = 5 * 64 + 13;
    constexpr size_t arena_size = 7 * 64 + 31;
    std::vector<uint8_t> pristine_sm(sm_size);
    std::vector<uint8_t> pristine_arena(arena_size);
    for (size_t index = 0; index < pristine_sm.size(); ++index) {
        pristine_sm[index] = static_cast<uint8_t>((index * 17U + 0x31U) & 0xffU);
    }
    for (size_t index = 0; index < pristine_arena.size(); ++index) {
        pristine_arena[index] = static_cast<uint8_t>((index * 29U + 0x72U) & 0xffU);
    }

    MultiLineRestoreHarness harness;
    harness.working_sm.resize(sm_size, 0xee);
    harness.working_arena.resize(arena_size, 0xdd);
    const HbgExecutionBinding binding = make_multi_line_binding(harness);
    const HbgInvocationIdentity identity = make_identity();
    const std::vector<HbgHostRegionInput> inputs{
        {HbgLaunchRegionKind::SharedMemoryImage, kRegionFlags, pristine_sm.data(), pristine_sm.size(), 0},
        {HbgLaunchRegionKind::RuntimeArenaImage, kRegionFlags, pristine_arena.data(), pristine_arena.size(), 0},
    };
    std::vector<uint8_t> blob;
    ASSERT_EQ(build_hbg_launch_blob(binding, identity, 61, inputs, &blob), HbgLaunchBlobStatus::Ok);
    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(blob.data());
    header->inline_payload_addr = reinterpret_cast<uint64_t>(blob.data()) + header->header_size;
    const HbgExecutionSlotRegistration registration = make_multi_line_registration(harness, binding, blob.size());
    const HbgRestoreOps ops{&harness, multi_line_restore_copy, multi_line_restore_publish};
    HbgRestoreCommit commit{};

    ASSERT_EQ(
        restore_hbg_launch_blob(blob.data(), blob.size(), registration, identity, ops, &commit).status,
        HbgRestoreStatus::Ok
    );
    EXPECT_EQ(harness.working_sm, pristine_sm);
    EXPECT_EQ(harness.working_arena, pristine_arena);
    EXPECT_EQ(commit.plan_generation, 61u);

    const std::array<size_t, 3> sm_poison_offsets{0, pristine_sm.size() / 2U, pristine_sm.size() - 1U};
    const std::array<size_t, 3> arena_poison_offsets{0, pristine_arena.size() / 2U, pristine_arena.size() - 1U};
    for (const size_t offset : sm_poison_offsets)
        harness.working_sm[offset] ^= 0xffU;
    for (const size_t offset : arena_poison_offsets)
        harness.working_arena[offset] ^= 0xffU;

    ASSERT_EQ(
        restore_hbg_launch_blob(blob.data(), blob.size(), registration, identity, ops, &commit).status,
        HbgRestoreStatus::Ok
    );
    EXPECT_EQ(harness.working_sm, pristine_sm);
    EXPECT_EQ(harness.working_arena, pristine_arena);
    EXPECT_EQ(harness.copy_calls, 4);
    EXPECT_EQ(harness.publish_calls, 4);
    EXPECT_EQ(harness.published_sizes, (std::array<size_t, 4>{sm_size, arena_size, sm_size, arena_size}));
}

TEST(HbgLaunchBlob, AlternatingCapturedPackagesRestoreTheirOwnSnapshotIntoOneSlot) {
    Sources sources_a;
    Sources sources_b;
    sources_a.sm.fill(0xa1);
    sources_a.arena.fill(0xa2);
    sources_b.sm.fill(0xb1);
    sources_b.arena.fill(0xb2);
    RestoreHarness harness;
    const HbgExecutionBinding binding = make_restore_binding(harness);
    const HbgInvocationIdentity identity_a = make_identity();
    HbgInvocationIdentity identity_b = identity_a;
    ++identity_b.argument_snapshot_hash;
    std::vector<uint8_t> blob_a = make_restore_blob(sources_a, binding, identity_a, 41);
    std::vector<uint8_t> blob_b = make_restore_blob(sources_b, binding, identity_b, 42);
    const HbgExecutionSlotRegistration registration =
        make_restore_registration(harness, binding, std::max(blob_a.size(), blob_b.size()));
    const HbgRestoreOps ops{&harness, restore_copy, restore_publish};
    HbgRestoreCommit commit{};

    ASSERT_EQ(
        restore_hbg_launch_blob(blob_a.data(), blob_a.size(), registration, identity_a, ops, &commit).status,
        HbgRestoreStatus::Ok
    );
    EXPECT_EQ(harness.working_sm, sources_a.sm);
    EXPECT_EQ(commit.plan_generation, 41u);

    ASSERT_EQ(
        restore_hbg_launch_blob(blob_b.data(), blob_b.size(), registration, identity_b, ops, &commit).status,
        HbgRestoreStatus::Ok
    );
    EXPECT_EQ(harness.working_sm, sources_b.sm);
    EXPECT_EQ(harness.working_arena, sources_b.arena);
    EXPECT_EQ(commit.plan_generation, 42u);

    ASSERT_EQ(
        restore_hbg_launch_blob(blob_a.data(), blob_a.size(), registration, identity_a, ops, &commit).status,
        HbgRestoreStatus::Ok
    );
    EXPECT_EQ(harness.working_sm, sources_a.sm);
    EXPECT_EQ(harness.working_arena, sources_a.arena);
    EXPECT_EQ(harness.copy_calls, 6);
}

TEST(HbgLaunchBlob, FailedRestoreNeverPublishesAReadyCommit) {
    Sources sources;
    sources.sm.fill(0xc1);
    sources.arena.fill(0xc2);
    RestoreHarness harness;
    const HbgExecutionBinding binding = make_restore_binding(harness);
    const HbgInvocationIdentity identity = make_identity();
    std::vector<uint8_t> blob = make_restore_blob(sources, binding, identity, 51);
    const HbgExecutionSlotRegistration registration = make_restore_registration(harness, binding, blob.size());
    const HbgRestoreOps ops{&harness, restore_copy, restore_publish};
    HbgRestoreCommit commit{91, 92, 93, identity};
    const HbgRestoreCommit sentinel = commit;

    HbgExecutionSlotRegistration corrupted_registration = registration;
    ++corrupted_registration.binary_generation;
    auto result = restore_hbg_launch_blob(blob.data(), blob.size(), corrupted_registration, identity, ops, &commit);
    EXPECT_EQ(result.status, HbgRestoreStatus::SlotRejected);
    EXPECT_EQ(result.slot_status, HbgExecutionSlotStatus::HashMismatch);
    EXPECT_EQ(harness.copy_calls, 0);
    EXPECT_EQ(std::memcmp(&commit, &sentinel, sizeof(commit)), 0);

    HbgExecutionSlotRegistration stale_registration = registration;
    ++stale_registration.binding.slot_generation;
    ASSERT_EQ(seal_hbg_execution_slot_registration(&stale_registration, 1), HbgExecutionSlotStatus::Ok);
    result = restore_hbg_launch_blob(blob.data(), blob.size(), stale_registration, identity, ops, &commit);
    EXPECT_EQ(result.status, HbgRestoreStatus::BlobRejected);
    EXPECT_EQ(result.blob_status, HbgLaunchBlobStatus::InvalidGeneration);
    EXPECT_EQ(harness.copy_calls, 0);
    EXPECT_EQ(std::memcmp(&commit, &sentinel, sizeof(commit)), 0);

    harness.working_sm.fill(0xee);
    harness.working_arena.fill(0xdd);
    harness.fail_copy_at = 1;
    result = restore_hbg_launch_blob(blob.data(), blob.size(), registration, identity, ops, &commit);
    EXPECT_EQ(result.status, HbgRestoreStatus::CopyFailed);
    EXPECT_EQ(result.region_index, 1u);
    EXPECT_EQ(result.callback_error, -91);
    EXPECT_EQ(harness.working_sm, sources.sm);
    EXPECT_NE(harness.working_arena, sources.arena);
    EXPECT_EQ(std::memcmp(&commit, &sentinel, sizeof(commit)), 0);

    harness.fail_copy_at = -1;
    harness.fail_publish_at = harness.publish_calls;
    result = restore_hbg_launch_blob(blob.data(), blob.size(), registration, identity, ops, &commit);
    EXPECT_EQ(result.status, HbgRestoreStatus::PublishFailed);
    EXPECT_EQ(result.region_index, 0u);
    EXPECT_EQ(result.callback_error, -92);
    EXPECT_EQ(std::memcmp(&commit, &sentinel, sizeof(commit)), 0);

    // A failed attempt may have partially modified the mutable slot, but it
    // never publishes a commit. A subsequent complete restore is the only
    // supported in-context repair and must overwrite every pristine byte.
    harness.fail_publish_at = -1;
    harness.working_sm.fill(0x19);
    harness.working_arena.fill(0x2a);
    result = restore_hbg_launch_blob(blob.data(), blob.size(), registration, identity, ops, &commit);
    ASSERT_EQ(result.status, HbgRestoreStatus::Ok);
    EXPECT_EQ(harness.working_sm, sources.sm);
    EXPECT_EQ(harness.working_arena, sources.arena);
    EXPECT_EQ(commit.plan_generation, 51u);
    EXPECT_EQ(commit.plan_hash, reinterpret_cast<const HbgLaunchBlobHeader *>(blob.data())->plan_hash);
}

}  // namespace

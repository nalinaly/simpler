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

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "hbg_launch_blob.h"
#include "hbg_launch_blob_builder.h"

namespace {

using simpler::hbg::build_hbg_launch_blob;
using simpler::hbg::hbg_inline_payload;
using simpler::hbg::hbg_launch_regions;
using simpler::hbg::HBG_REGION_IMMUTABLE_SOURCE;
using simpler::hbg::HBG_REGION_REQUIRED;
using simpler::hbg::HbgExecutionBinding;
using simpler::hbg::HbgHostRegionInput;
using simpler::hbg::HbgInvocationIdentity;
using simpler::hbg::HbgLaunchBlobAddressMode;
using simpler::hbg::HbgLaunchBlobHeader;
using simpler::hbg::HbgLaunchBlobStatus;
using simpler::hbg::HbgLaunchRegion;
using simpler::hbg::HbgLaunchRegionKind;
using simpler::hbg::validate_hbg_launch_blob;

constexpr uint32_t kRegionFlags = HBG_REGION_REQUIRED | HBG_REGION_IMMUTABLE_SOURCE;

struct Sources {
    std::array<uint8_t, 16> sm{};
    std::array<uint8_t, 24> arena{};
};

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

HbgInvocationIdentity make_identity() {
    HbgInvocationIdentity identity;
    identity.callable_hash = 0x1111222233334444ULL;
    identity.argument_snapshot_hash = 0x5555666677778888ULL;
    identity.function_binding_hash = 0x9999aaaabbbbccccULL;
    identity.tensor_count = 2;
    identity.scalar_count = 1;
    return identity;
}

std::vector<HbgHostRegionInput> make_inputs(const Sources &sources) {
    return {
        {HbgLaunchRegionKind::SharedMemoryImage, kRegionFlags, sources.sm.data(), sources.sm.size(), 0},
        {HbgLaunchRegionKind::RuntimeArenaImage, kRegionFlags, sources.arena.data(), sources.arena.size(), 0},
    };
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

}  // namespace

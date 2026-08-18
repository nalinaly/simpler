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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "hbg_launch_blob.h"

namespace simpler::hbg {

struct HbgHostRegionInput {
    HbgLaunchRegionKind kind;
    uint32_t flags;
    const void *data;
    uint64_t size;
    uint64_t destination_offset;
};

inline bool hbg_checked_align_up_size(size_t value, size_t alignment, size_t *out) noexcept {
    if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return false;
    }
    *out = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

/**
 * Serialize immutable graph bytes into a writable per-launch host blob.
 *
 * The returned vector owns a deep copy.  CANN may later patch the copy's
 * inline_payload_addr, while the caller's canonical plan and source buffers
 * remain unchanged.  No CANN limit is encoded here beyond the public uint32
 * args-size carrier; device capability/large-args limits remain Phase-H0
 * measurements and must not be inferred from this host-only builder.
 */
inline HbgLaunchBlobStatus build_hbg_launch_blob(
    const HbgExecutionBinding &binding, const HbgInvocationIdentity &identity, uint64_t plan_generation,
    const std::vector<HbgHostRegionInput> &inputs, std::vector<uint8_t> *out
) noexcept {
    if (out == nullptr) return HbgLaunchBlobStatus::NullArgument;
    if (inputs.size() < 2 || inputs.size() > HBG_LAUNCH_BLOB_MAX_REGIONS) return HbgLaunchBlobStatus::InvalidHeader;
    if (plan_generation == 0) return HbgLaunchBlobStatus::InvalidGeneration;
    if (!hbg_valid_invocation_identity(identity)) return HbgLaunchBlobStatus::InvalidIdentity;

    size_t descriptor_bytes = 0;
    if (inputs.size() > std::numeric_limits<size_t>::max() / sizeof(HbgLaunchRegion)) {
        return HbgLaunchBlobStatus::OutOfBounds;
    }
    descriptor_bytes = inputs.size() * sizeof(HbgLaunchRegion);
    size_t descriptor_end = 0;
    if (descriptor_bytes > std::numeric_limits<size_t>::max() - sizeof(HbgLaunchBlobHeader)) {
        return HbgLaunchBlobStatus::OutOfBounds;
    }
    descriptor_end = sizeof(HbgLaunchBlobHeader) + descriptor_bytes;

    size_t header_size = 0;
    if (!hbg_checked_align_up_size(descriptor_end, HBG_LAUNCH_BLOB_ALIGNMENT, &header_size)) {
        return HbgLaunchBlobStatus::OutOfBounds;
    }

    std::vector<uint64_t> source_offsets;
    try {
        source_offsets.resize(inputs.size());
    } catch (...) {
        return HbgLaunchBlobStatus::AllocationFailure;
    }

    size_t payload_size = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const HbgHostRegionInput &input = inputs[i];
        if (input.data == nullptr || input.size == 0 || input.size > std::numeric_limits<size_t>::max()) {
            return HbgLaunchBlobStatus::InvalidRegion;
        }
        if (!hbg_checked_align_up_size(payload_size, HBG_LAUNCH_BLOB_ALIGNMENT, &payload_size)) {
            return HbgLaunchBlobStatus::OutOfBounds;
        }
        source_offsets[i] = payload_size;
        if (static_cast<size_t>(input.size) > std::numeric_limits<size_t>::max() - payload_size) {
            return HbgLaunchBlobStatus::OutOfBounds;
        }
        payload_size += static_cast<size_t>(input.size);
    }

    if (payload_size > std::numeric_limits<size_t>::max() - header_size) {
        return HbgLaunchBlobStatus::OutOfBounds;
    }
    const size_t total_size = header_size + payload_size;
    if (total_size > std::numeric_limits<uint32_t>::max()) return HbgLaunchBlobStatus::OutOfBounds;

    std::vector<uint8_t> candidate;
    try {
        candidate.assign(total_size, 0);
    } catch (...) {
        return HbgLaunchBlobStatus::AllocationFailure;
    }

    auto *header = reinterpret_cast<HbgLaunchBlobHeader *>(candidate.data());
    *header = HbgLaunchBlobHeader{};
    header->header_size = static_cast<uint32_t>(header_size);
    header->total_size = static_cast<uint32_t>(total_size);
    header->region_count = static_cast<uint32_t>(inputs.size());
    header->plan_generation = plan_generation;
    header->inline_payload_size = payload_size;
    header->binding = binding;
    header->identity = identity;

    auto *regions = reinterpret_cast<HbgLaunchRegion *>(candidate.data() + sizeof(*header));
    uint8_t *payload = candidate.data() + header_size;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const HbgHostRegionInput &input = inputs[i];
        regions[i].kind = input.kind;
        regions[i].flags = input.flags;
        regions[i].source_offset = source_offsets[i];
        regions[i].size = input.size;
        regions[i].destination_offset = input.destination_offset;
        std::memcpy(payload + source_offsets[i], input.data, static_cast<size_t>(input.size));
    }
    header->plan_hash = hbg_plan_hash(identity, regions, header->region_count, payload, header->inline_payload_size);

    const HbgLaunchBlobStatus status =
        validate_hbg_launch_blob(candidate.data(), candidate.size(), HbgLaunchBlobAddressMode::HostUnpatched);
    if (status != HbgLaunchBlobStatus::Ok) return status;
    *out = std::move(candidate);
    return HbgLaunchBlobStatus::Ok;
}

}  // namespace simpler::hbg

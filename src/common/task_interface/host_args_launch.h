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
#include <limits>
#include <type_traits>

namespace simpler::host_args {

/**
 * CANN-independent mirror of one aclrtPlaceHolderInfo entry.
 *
 * addr_offset identifies an eight-byte pointer field inside the host argument
 * image. data_offset identifies inline bytes in the same image.  The onboard
 * bridge locks this layout against the installed CANN definition before it
 * passes the array to aclrtLaunchKernelWithHostArgs.
 */
struct HostArgsPlaceholder {
    uint32_t addr_offset{0};
    uint32_t data_offset{0};
};

static_assert(sizeof(HostArgsPlaceholder) == 8, "host-args placeholder ABI changed");
static_assert(alignof(HostArgsPlaceholder) == 4, "host-args placeholder alignment changed");
static_assert(std::is_standard_layout_v<HostArgsPlaceholder>, "host-args placeholder must be standard-layout");
static_assert(std::is_trivially_copyable_v<HostArgsPlaceholder>, "host-args placeholder must be byte-copyable");

enum class HostArgsLaunchStatus : uint32_t {
    Ok = 0,
    NullArguments,
    EmptyArguments,
    ArgumentsTooLarge,
    PlaceholderPointerMismatch,
    TooManyPlaceholders,
    MisalignedAddressField,
    AddressFieldOutOfBounds,
    DataOffsetOutOfBounds,
    OverlappingAddressFields,
};

/**
 * Validate the lossy size carriers and every placeholder write before CANN.
 *
 * The public ACL API accepts size_t, but the installed runtime forwards the
 * argument byte count through uint32_t and the placeholder count through a
 * uint16_t field.  HBG uses one placeholder; the explicit checks prevent a
 * future caller from relying on silent narrowing.  These carrier limits are
 * not exposed as HBG product capacities and do not replace onboard large-args
 * probes.
 */
inline HostArgsLaunchStatus validate_host_args_launch_layout(
    const void *host_args, size_t args_size, const HostArgsPlaceholder *placeholders, size_t placeholder_count
) noexcept {
    if (host_args == nullptr) return HostArgsLaunchStatus::NullArguments;
    if (args_size == 0) return HostArgsLaunchStatus::EmptyArguments;
    if (args_size > std::numeric_limits<uint32_t>::max()) return HostArgsLaunchStatus::ArgumentsTooLarge;
    if ((placeholders == nullptr) != (placeholder_count == 0)) {
        return HostArgsLaunchStatus::PlaceholderPointerMismatch;
    }
    if (placeholder_count > std::numeric_limits<uint16_t>::max()) {
        return HostArgsLaunchStatus::TooManyPlaceholders;
    }

    for (size_t index = 0; index < placeholder_count; ++index) {
        const HostArgsPlaceholder &placeholder = placeholders[index];
        if (placeholder.addr_offset % alignof(uint64_t) != 0) {
            return HostArgsLaunchStatus::MisalignedAddressField;
        }
        if (placeholder.addr_offset > args_size || sizeof(uint64_t) > args_size - placeholder.addr_offset) {
            return HostArgsLaunchStatus::AddressFieldOutOfBounds;
        }
        if (placeholder.data_offset >= args_size) return HostArgsLaunchStatus::DataOffsetOutOfBounds;

        for (size_t previous = 0; previous < index; ++previous) {
            const uint32_t previous_offset = placeholders[previous].addr_offset;
            const uint32_t current_offset = placeholder.addr_offset;
            const uint32_t lower = previous_offset < current_offset ? previous_offset : current_offset;
            const uint32_t upper = previous_offset < current_offset ? current_offset : previous_offset;
            if (upper - lower < sizeof(uint64_t)) {
                return HostArgsLaunchStatus::OverlappingAddressFields;
            }
        }
    }
    return HostArgsLaunchStatus::Ok;
}

}  // namespace simpler::host_args

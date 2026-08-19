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
#include <type_traits>

namespace simpler::hbg {

inline constexpr uint32_t HBG_L1_DIRECT_AIV_MAGIC = UINT32_C(0x48444156);  // "HDAV"
inline constexpr uint32_t HBG_L1_DIRECT_AIV_ABI_VERSION = 1;
inline constexpr uint32_t HBG_L1_DIRECT_AIV_TENSOR_BYTES = 128;
inline constexpr uint32_t HBG_L1_DIRECT_AIV_LANE_SCRATCH_BYTES = 8192;
inline constexpr uint32_t HBG_L1_DIRECT_AIV_LAUNCH_PREFIX_BYTES = 64;
// The dedicated direct-AIV entry declares exactly one native pointer argument.
// CANN patches it to the inline package copied into runtime-owned task args.
inline constexpr uint32_t HBG_L1_DIRECT_AIV_PACKAGE_POINTER_OFFSET = 0;
inline constexpr uint32_t HBG_L1_DIRECT_AIV_HEADER_BYTES = 128;

/**
 * Immutable header for the A2/A3 homogeneous, dependency-free HBG L1 path.
 *
 * CANN owns one device copy of the complete package for each enqueued task or
 * captured graph node.  The bytes before lane_scratch_offset are immutable;
 * per-task descriptors are immutable. ``lane_scratch_device_addr`` names a
 * context-owned, 64-byte-aligned working area in the frozen HBG runtime slot;
 * it is not part of CANN's possibly under-aligned HostArgs allocation.
 */
struct HbgL1DirectAivPackageHeader {
    uint32_t magic{HBG_L1_DIRECT_AIV_MAGIC};
    uint32_t abi_version{HBG_L1_DIRECT_AIV_ABI_VERSION};
    uint32_t header_size{HBG_L1_DIRECT_AIV_HEADER_BYTES};
    uint32_t total_size{0};

    uint32_t task_count{0};
    uint32_t logical_block_num{0};
    uint32_t work_count{0};
    uint32_t tensor_count{0};

    uint32_t scalar_count{0};
    uint32_t task_record_stride{0};
    uint32_t task_records_offset{HBG_L1_DIRECT_AIV_HEADER_BYTES};
    uint32_t lane_count{0};

    uint32_t lane_scratch_stride{HBG_L1_DIRECT_AIV_LANE_SCRATCH_BYTES};
    uint32_t lane_scratch_offset{0};
    uint32_t immutable_size{0};
    uint32_t reserved{0};

    uint64_t function_bin_addr{0};
    uint64_t lane_scratch_device_addr{0};
    uint64_t reserved_u64[6]{};
};

static_assert(
    sizeof(HbgL1DirectAivPackageHeader) == HBG_L1_DIRECT_AIV_HEADER_BYTES,
    "direct HBG package header must be two cache lines"
);
static_assert(alignof(HbgL1DirectAivPackageHeader) == alignof(uint64_t));
static_assert(std::is_standard_layout_v<HbgL1DirectAivPackageHeader>);
static_assert(std::is_trivially_copyable_v<HbgL1DirectAivPackageHeader>);

inline constexpr bool hbg_l1_direct_align_up(uint64_t value, uint64_t alignment, uint64_t *out) noexcept {
    if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) return false;
    *out = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

inline constexpr bool
hbg_l1_direct_record_stride(uint32_t tensor_count, uint32_t scalar_count, uint32_t *out) noexcept {
    if (out == nullptr) return false;
    const uint64_t tensor_bytes = static_cast<uint64_t>(tensor_count) * HBG_L1_DIRECT_AIV_TENSOR_BYTES;
    const uint64_t scalar_bytes = static_cast<uint64_t>(scalar_count) * sizeof(uint64_t);
    uint64_t aligned = 0;
    if (tensor_bytes > std::numeric_limits<uint64_t>::max() - scalar_bytes ||
        !hbg_l1_direct_align_up(tensor_bytes + scalar_bytes, 64, &aligned) ||
        aligned > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *out = static_cast<uint32_t>(aligned);
    return true;
}

inline constexpr bool
hbg_l1_direct_package_layout_valid(const HbgL1DirectAivPackageHeader &header, size_t package_size) noexcept {
    if (header.magic != HBG_L1_DIRECT_AIV_MAGIC || header.abi_version != HBG_L1_DIRECT_AIV_ABI_VERSION ||
        header.header_size != HBG_L1_DIRECT_AIV_HEADER_BYTES || header.total_size != package_size ||
        header.task_count == 0 || header.logical_block_num == 0 || header.lane_count == 0 ||
        header.function_bin_addr == 0 || header.task_records_offset != HBG_L1_DIRECT_AIV_HEADER_BYTES ||
        header.lane_scratch_stride != HBG_L1_DIRECT_AIV_LANE_SCRATCH_BYTES || header.lane_scratch_offset != 0 ||
        header.immutable_size != package_size || header.lane_scratch_device_addr == 0 ||
        (header.lane_scratch_device_addr & 63u) != 0 || header.reserved != 0) {
        return false;
    }
    for (uint64_t value : header.reserved_u64) {
        if (value != 0) return false;
    }

    const uint64_t expected_work_count =
        static_cast<uint64_t>(header.task_count) * static_cast<uint64_t>(header.logical_block_num);
    if (expected_work_count != header.work_count || expected_work_count > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    uint32_t expected_record_stride = 0;
    if (!hbg_l1_direct_record_stride(header.tensor_count, header.scalar_count, &expected_record_stride) ||
        expected_record_stride != header.task_record_stride) {
        return false;
    }

    const uint64_t records_end = static_cast<uint64_t>(header.task_records_offset) +
                                 static_cast<uint64_t>(header.task_count) * header.task_record_stride;
    uint64_t expected_total = 0;
    if (!hbg_l1_direct_align_up(records_end, 64, &expected_total)) {
        return false;
    }
    return expected_total == package_size && expected_total <= std::numeric_limits<uint32_t>::max();
}

inline bool validate_hbg_l1_direct_package(const void *package, size_t package_size) noexcept {
    if (package == nullptr || package_size < sizeof(HbgL1DirectAivPackageHeader)) return false;
    HbgL1DirectAivPackageHeader header{};
    std::memcpy(&header, package, sizeof(header));
    return hbg_l1_direct_package_layout_valid(header, package_size);
}

}  // namespace simpler::hbg

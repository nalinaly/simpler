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
#include <type_traits>

namespace simpler::test::host_args_probe {

constexpr uint64_t HOST_ARGS_PROBE_MAGIC = 0x4853544152475331ULL;         // "HSTARGS1"
constexpr uint64_t HOST_ARGS_PROBE_RESULT_MAGIC = 0x48535452534c5431ULL;  // "HSTRSLT1"
constexpr uint32_t HOST_ARGS_PROBE_ABI_MAJOR = 1;
constexpr uint32_t HOST_ARGS_PROBE_ABI_MINOR = 0;
constexpr size_t HOST_ARGS_PROBE_PAYLOAD_COUNT = 3;

enum HostArgsProbeStatus : uint32_t {
    HOST_ARGS_PROBE_OK = 0,
    HOST_ARGS_PROBE_BAD_MAGIC = 1U << 0,
    HOST_ARGS_PROBE_BAD_VERSION = 1U << 1,
    HOST_ARGS_PROBE_BAD_HEADER_SIZE = 1U << 2,
    HOST_ARGS_PROBE_BAD_TOTAL_SIZE = 1U << 3,
    HOST_ARGS_PROBE_BAD_RESULT_ADDRESS = 1U << 4,
    HOST_ARGS_PROBE_BAD_REGION_LAYOUT = 1U << 5,
    HOST_ARGS_PROBE_BAD_PLACEHOLDER_0 = 1U << 6,
    HOST_ARGS_PROBE_BAD_PLACEHOLDER_1 = 1U << 7,
    HOST_ARGS_PROBE_BAD_PLACEHOLDER_2 = 1U << 8,
    HOST_ARGS_PROBE_BAD_CHECKSUM_0 = 1U << 9,
    HOST_ARGS_PROBE_BAD_CHECKSUM_1 = 1U << 10,
    HOST_ARGS_PROBE_BAD_CHECKSUM_2 = 1U << 11,
    HOST_ARGS_PROBE_BAD_SAMPLE_0 = 1U << 12,
    HOST_ARGS_PROBE_BAD_SAMPLE_1 = 1U << 13,
    HOST_ARGS_PROBE_BAD_SAMPLE_2 = 1U << 14,
};

struct HostArgsProbeHeader {
    uint64_t magic{HOST_ARGS_PROBE_MAGIC};
    uint32_t abi_major{HOST_ARGS_PROBE_ABI_MAJOR};
    uint32_t abi_minor{HOST_ARGS_PROBE_ABI_MINOR};
    uint32_t header_size{sizeof(HostArgsProbeHeader)};
    uint32_t total_size{0};
    uint64_t invocation_id{0};
    uint64_t result_addr{0};
    uint64_t payload_addr[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint32_t payload_offset[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint32_t payload_size[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint64_t expected_checksum[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint8_t expected_first[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint8_t expected_middle[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint8_t expected_tail[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint8_t reserved[7]{};
};

static_assert(sizeof(HostArgsProbeHeader) == 128, "host-args probe header ABI changed");
static_assert(alignof(HostArgsProbeHeader) == 8, "host-args probe header alignment changed");
static_assert(offsetof(HostArgsProbeHeader, payload_addr) == 40, "placeholder field offset changed");
static_assert(std::is_standard_layout_v<HostArgsProbeHeader>, "probe header must be standard-layout");
static_assert(std::is_trivially_copyable_v<HostArgsProbeHeader>, "probe header must be byte-copyable");

struct HostArgsProbeResult {
    uint64_t magic{HOST_ARGS_PROBE_RESULT_MAGIC};
    uint64_t invocation_id{0};
    uint64_t args_base{0};
    uint32_t args_base_mod_64{0};
    uint32_t total_size{0};
    uint32_t status{HOST_ARGS_PROBE_OK};
    uint32_t reserved0{0};
    uint64_t observed_addr[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint64_t expected_addr[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint64_t observed_checksum[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint64_t expected_checksum[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint8_t observed_first[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint8_t observed_middle[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint8_t observed_tail[HOST_ARGS_PROBE_PAYLOAD_COUNT]{};
    uint8_t reserved1[7]{};
};

static_assert(sizeof(HostArgsProbeResult) == 152, "host-args probe result ABI changed");
static_assert(alignof(HostArgsProbeResult) == 8, "host-args probe result alignment changed");
static_assert(std::is_standard_layout_v<HostArgsProbeResult>, "probe result must be standard-layout");
static_assert(std::is_trivially_copyable_v<HostArgsProbeResult>, "probe result must be byte-copyable");

constexpr uint64_t HOST_ARGS_PROBE_FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr uint64_t HOST_ARGS_PROBE_FNV_PRIME = 0x100000001b3ULL;

inline uint64_t host_args_probe_checksum(const uint8_t *bytes, size_t size) noexcept {
    uint64_t value = HOST_ARGS_PROBE_FNV_OFFSET;
    for (size_t index = 0; index < size; ++index) {
        value ^= bytes[index];
        value *= HOST_ARGS_PROBE_FNV_PRIME;
    }
    return value;
}

}  // namespace simpler::test::host_args_probe

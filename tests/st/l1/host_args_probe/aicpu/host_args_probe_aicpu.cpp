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

#include <cstddef>
#include <cstdint>

#include "host_args_probe_abi.h"

namespace {

using simpler::test::host_args_probe::HOST_ARGS_PROBE_ABI_MAJOR;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_ABI_MINOR;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_CHECKSUM_0;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_HEADER_SIZE;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_MAGIC;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_PLACEHOLDER_0;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_REGION_LAYOUT;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_RESULT_ADDRESS;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_SAMPLE_0;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_TOTAL_SIZE;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_VERSION;
using simpler::test::host_args_probe::host_args_probe_checksum;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_MAGIC;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_PAYLOAD_COUNT;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_RESULT_MAGIC;
using simpler::test::host_args_probe::HostArgsProbeHeader;
using simpler::test::host_args_probe::HostArgsProbeResult;

bool region_is_in_bounds(const HostArgsProbeHeader &header, size_t index) {
    const uint32_t offset = header.payload_offset[index];
    const uint32_t size = header.payload_size[index];
    return size != 0 && offset >= header.header_size && offset <= header.total_size &&
           size <= header.total_size - offset;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int simpler_host_args_probe_init(void *args) {
    (void)args;
    return 0;
}

__attribute__((visibility("default"))) int simpler_host_args_probe_run(void *args) {
    if (args == nullptr) return 1;

    // CANN only promises a byte image.  Copy the fixed prefix before typed
    // access so a runtime args allocation with less than 64-byte alignment
    // never creates a misaligned HostArgsProbeHeader lvalue.
    alignas(16) HostArgsProbeHeader header{};
    __builtin_memcpy(&header, args, sizeof(header));
    if (header.result_addr == 0) return 2;

    HostArgsProbeResult result{};
    result.magic = HOST_ARGS_PROBE_RESULT_MAGIC;
    result.invocation_id = header.invocation_id;
    result.args_base = reinterpret_cast<uint64_t>(args);
    result.args_base_mod_64 = static_cast<uint32_t>(result.args_base % 64U);
    result.total_size = header.total_size;

    if (header.magic != HOST_ARGS_PROBE_MAGIC) result.status |= HOST_ARGS_PROBE_BAD_MAGIC;
    if (header.abi_major != HOST_ARGS_PROBE_ABI_MAJOR || header.abi_minor != HOST_ARGS_PROBE_ABI_MINOR) {
        result.status |= HOST_ARGS_PROBE_BAD_VERSION;
    }
    if (header.header_size != sizeof(HostArgsProbeHeader)) result.status |= HOST_ARGS_PROBE_BAD_HEADER_SIZE;
    if (header.total_size < sizeof(HostArgsProbeHeader)) result.status |= HOST_ARGS_PROBE_BAD_TOTAL_SIZE;
    if (header.result_addr == 0) result.status |= HOST_ARGS_PROBE_BAD_RESULT_ADDRESS;

    const auto *args_bytes = reinterpret_cast<const uint8_t *>(args);
    for (size_t index = 0; index < HOST_ARGS_PROBE_PAYLOAD_COUNT; ++index) {
        result.observed_addr[index] = header.payload_addr[index];
        result.expected_checksum[index] = header.expected_checksum[index];
        if (!region_is_in_bounds(header, index)) {
            result.status |= HOST_ARGS_PROBE_BAD_REGION_LAYOUT;
            continue;
        }
        result.expected_addr[index] = reinterpret_cast<uint64_t>(args_bytes + header.payload_offset[index]);
        if (result.observed_addr[index] != result.expected_addr[index]) {
            result.status |= HOST_ARGS_PROBE_BAD_PLACEHOLDER_0 << index;
            continue;
        }

        const auto *payload = reinterpret_cast<const uint8_t *>(header.payload_addr[index]);
        const size_t size = header.payload_size[index];
        result.observed_first[index] = payload[0];
        result.observed_middle[index] = payload[size / 2];
        result.observed_tail[index] = payload[size - 1];
        result.observed_checksum[index] = host_args_probe_checksum(payload, size);
        if (result.observed_checksum[index] != header.expected_checksum[index]) {
            result.status |= HOST_ARGS_PROBE_BAD_CHECKSUM_0 << index;
        }
        if (result.observed_first[index] != header.expected_first[index] ||
            result.observed_middle[index] != header.expected_middle[index] ||
            result.observed_tail[index] != header.expected_tail[index]) {
            result.status |= HOST_ARGS_PROBE_BAD_SAMPLE_0 << index;
        }
    }

    auto *device_result = reinterpret_cast<HostArgsProbeResult *>(header.result_addr);
    __builtin_memcpy(device_result, &result, sizeof(result));
    // Keep the runtime task successful so Host can inspect an exact status
    // bitmap instead of losing evidence to a generic AICPU task failure.
    return 0;
}

}  // extern "C"

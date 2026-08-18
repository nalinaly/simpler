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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "host_args_probe_abi.h"
#include "host_args_probe_loader_contract.h"

extern "C" int simpler_host_args_probe_run(void *args);

namespace {

using simpler::test::host_args_probe::HOST_ARGS_PROBE_BAD_PLACEHOLDER_1;
using simpler::test::host_args_probe::host_args_probe_checksum;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_MAGIC;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_OK;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_PAYLOAD_COUNT;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_RESULT_MAGIC;
using simpler::test::host_args_probe::HostArgsProbeHeader;
using simpler::test::host_args_probe::HostArgsProbeResult;

constexpr size_t kArgsSize = 4096;

uint8_t Pattern(size_t region, size_t index) {
    return static_cast<uint8_t>((region * 73U + index * 29U + (index >> 5U)) & 0xffU);
}

bool BuildPatchedImage(uint8_t *args, HostArgsProbeResult *result) {
    HostArgsProbeHeader header{};
    header.magic = HOST_ARGS_PROBE_MAGIC;
    header.total_size = kArgsSize;
    header.invocation_id = 77;
    header.result_addr = reinterpret_cast<uint64_t>(result);
    constexpr std::array<uint32_t, HOST_ARGS_PROBE_PAYLOAD_COUNT> offsets{128, 1024, 2560};
    constexpr std::array<uint32_t, HOST_ARGS_PROBE_PAYLOAD_COUNT> sizes{896, 1536, 1536};
    for (size_t region = 0; region < HOST_ARGS_PROBE_PAYLOAD_COUNT; ++region) {
        header.payload_offset[region] = offsets[region];
        header.payload_size[region] = sizes[region];
        header.payload_addr[region] = reinterpret_cast<uint64_t>(args + offsets[region]);
        uint8_t *payload = args + offsets[region];
        for (size_t index = 0; index < sizes[region]; ++index)
            payload[index] = Pattern(region, index);
        header.expected_checksum[region] = host_args_probe_checksum(payload, sizes[region]);
        header.expected_first[region] = payload[0];
        header.expected_middle[region] = payload[sizes[region] / 2U];
        header.expected_tail[region] = payload[sizes[region] - 1U];
    }
    std::memcpy(args, &header, sizeof(header));
    return true;
}

}  // namespace

int main() {
    char basename[64]{};
    if (!simpler::test::host_args_probe::format_dispatcher_inner_so_basename(
            0x0123456789abcdefULL, 1, basename, sizeof(basename)
        ) ||
        std::strcmp(basename, "simpler_inner_0123456789abcdef_1.so") != 0 ||
        simpler::test::host_args_probe::format_dispatcher_inner_so_basename(
            0x0123456789abcdefULL, -1, basename, sizeof(basename)
        )) {
        std::fprintf(stderr, "dispatcher inner SO basename contract failed: %s\n", basename);
        return 1;
    }

    // Deliberately offset the argument image by one byte.  The probe kernel
    // must copy its prefix into an aligned local object before typed access.
    std::vector<uint8_t> storage(kArgsSize + 1U, 0);
    uint8_t *const args = storage.data() + 1U;
    HostArgsProbeResult result{};
    BuildPatchedImage(args, &result);
    if (simpler_host_args_probe_run(args) != 0 || result.magic != HOST_ARGS_PROBE_RESULT_MAGIC ||
        result.status != HOST_ARGS_PROBE_OK || result.invocation_id != 77 || result.args_base % 2U == 0) {
        std::fprintf(
            stderr, "unaligned valid image failed: magic=%016llx status=0x%08x args=%016llx\n",
            static_cast<unsigned long long>(result.magic), result.status,
            static_cast<unsigned long long>(result.args_base)
        );
        return 2;
    }

    HostArgsProbeHeader header{};
    std::memcpy(&header, args, sizeof(header));
    ++header.payload_addr[1];
    std::memcpy(args, &header, sizeof(header));
    result = HostArgsProbeResult{};
    if (simpler_host_args_probe_run(args) != 0 || (result.status & HOST_ARGS_PROBE_BAD_PLACEHOLDER_1) == 0) {
        std::fprintf(stderr, "bad placeholder was not diagnosed: status=0x%08x\n", result.status);
        return 3;
    }

    std::printf("HOST_ARGS_PROBE_SELF_TEST PASS basename_unaligned_prefix_and_placeholder_diagnostic\n");
    return 0;
}

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

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace simpler::test::host_args_probe {

/**
 * Format the content-addressed basename written by the shared dispatcher.
 *
 * The name identifies the bundled inner DSO, not the symbols exported by that
 * DSO.  The ACL descriptor and dispatcher must therefore use the same
 * `simpler_inner_` prefix even for this standalone probe.
 */
inline bool format_dispatcher_inner_so_basename(
    uint64_t fingerprint, int device_id, char *output, size_t output_capacity
) noexcept {
    if (device_id < 0 || output == nullptr || output_capacity == 0) return false;
    const int written =
        std::snprintf(output, output_capacity, "simpler_inner_%016" PRIx64 "_%d.so", fingerprint, device_id);
    return written > 0 && static_cast<size_t>(written) < output_capacity;
}

}  // namespace simpler::test::host_args_probe

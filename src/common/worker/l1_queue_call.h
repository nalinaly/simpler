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

#include <cstdint>

inline constexpr uint32_t SIMPLER_L1_QUEUE_CALL_ABI_VERSION = 1;
inline constexpr const char *SIMPLER_L1_QUEUE_CALL_CAPSULE_NAME = "simpler.l1.queue_call.v1";

/**
 * Torch/taskQueue-neutral deferred L1 call ABI.
 *
 * The producer owns an immutable prepare/launch argument snapshot behind
 * `opaque`. A consumer such as the optional torch_npu adapter retains this
 * object while its callback is queued, then invokes it with the raw stream
 * supplied by that queue. No Python object or torch type crosses this ABI.
 */
struct SimplerL1QueueCall {
    uint32_t abi_version;
    uint32_t struct_size;
    void *opaque;
    void (*retain)(void *opaque) noexcept;
    void (*release)(void *opaque) noexcept;
    int (*invoke)(void *opaque, uint64_t caller_stream) noexcept;
};

static_assert(sizeof(SimplerL1QueueCall) == 40, "SimplerL1QueueCall ABI size changed");

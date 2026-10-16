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
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

// 每个核使用独立 cache line，避免标量写回造成核间覆盖。
// 输入 0 的值携带实际 Tensor 数，最后一个 Tensor 为输出。
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *first = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *first_data = reinterpret_cast<__gm__ float *>(first->buffer.addr) + first->start_offset;
    int32_t count = static_cast<int32_t>(first_data[0]);
    float value = 0;
    for (int32_t i = 0; i < count - 1; ++i) {
        auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[i]);
        auto *data = reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset;
        value += data[0] * static_cast<float>(i + 1);
    }
    // 刻意读取全部 16 个标量；count=256 时最后一项位于槽 271。
    for (int32_t i = 0; i < CORE_MAX_SCALAR_ARGS; ++i) {
        value += static_cast<float>(args[count + i] * (i + 1));
    }
    int32_t lane = 0;
#if CAPACITY_AIV
    lane = 1 + get_sub_block_id(args);
#endif
    int32_t block = get_block_idx(args);
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[count - 1]);
    auto *out = reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset;
    int32_t offset = (block * 3 + lane) * 16;
    out[offset] = value + block * 1000 + lane * 100 + get_block_num(args);
#ifndef PTO_CPUSTUB_HPP
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
}

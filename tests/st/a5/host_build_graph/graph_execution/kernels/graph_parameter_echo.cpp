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

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "inner_kernel.h"
#include "scheduler/scheduler_memory.h"
#include "tensor.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *left = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *right = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *a = reinterpret_cast<__gm__ int64_t *>(left->buffer.addr) + left->start_offset;
    auto *b = reinterpret_cast<__gm__ int64_t *>(right->buffer.addr) + right->start_offset;
    const int64_t count = args[2] & 31;
    if (count >= 1 && count <= 16) {
        for (int64_t index = 0; index < count; ++index) {
            a[index] = args[index + 2];
            b[index] = args[index + 2] ^ 0x5555;
        }
    }
    a[31] = static_cast<int64_t>(left->start_offset);
    b[31] = left->buffer.addr == right->buffer.addr && right->start_offset == left->start_offset + 32 ?
                static_cast<int64_t>(right->start_offset) :
                -1;
    for (int32_t index = 0; index < 32; index += 8) {
        scheduler_writeback_cache_line(a + index);
        scheduler_writeback_cache_line(b + index);
    }
    scheduler_cache_barrier();
}

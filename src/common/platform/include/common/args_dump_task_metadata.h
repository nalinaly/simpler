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
#include <type_traits>

#include "arg_direction.h"

// 按实际参数槽编号选择 dump：Tensor 在前，标量紧随实际 Tensor 数量。
// 使用定长 POD 多字掩码，保证跨 Host/AICPU memcpy 时不携带指针，
// 并避免参数编号 >= 64 时移位溢出或静默丢失选择信息。
constexpr uint32_t ARGS_DUMP_ARG_MASK_BITS = CORE_MAX_TENSOR_ARGS + CORE_MAX_SCALAR_ARGS;
constexpr uint32_t ARGS_DUMP_ARG_MASK_WORDS = (ARGS_DUMP_ARG_MASK_BITS + 63) / 64;

struct ArgsDumpArgMask {
    uint64_t words[ARGS_DUMP_ARG_MASK_WORDS]{};

    constexpr ArgsDumpArgMask() = default;
    // 保留低 64 位掩码的构造方式；高位参数通过 set() 指定。
    constexpr ArgsDumpArgMask(uint64_t low_word) :
        words{low_word} {}

    constexpr void set(int32_t index) {
        if (index >= 0 && index < static_cast<int32_t>(ARGS_DUMP_ARG_MASK_BITS)) {
            words[index / 64] |= uint64_t{1} << (index % 64);
        }
    }
    constexpr bool test(int32_t index) const {
        return index >= 0 && index < static_cast<int32_t>(ARGS_DUMP_ARG_MASK_BITS) &&
               (words[index / 64] & (uint64_t{1} << (index % 64))) != 0;
    }
    constexpr bool operator==(const ArgsDumpArgMask &other) const {
        for (uint32_t i = 0; i < ARGS_DUMP_ARG_MASK_WORDS; ++i) {
            if (words[i] != other.words[i]) return false;
        }
        return true;
    }
    constexpr bool operator!=(const ArgsDumpArgMask &other) const { return !(*this == other); }
};

constexpr ArgsDumpArgMask ARGS_DUMP_ARG_MASK_NONE{};
constexpr uint8_t ARGS_DUMP_RECORD_FLAG_ARG_INDEX_AMBIGUOUS = 1u << 0;

struct ArgsDumpTaskMetadata {
    ArgsDumpArgMask dump_arg_mask{ARGS_DUMP_ARG_MASK_NONE};
    ArgsDumpArgMask dump_arg_flags{ARGS_DUMP_ARG_MASK_NONE};
    uint8_t scalar_dtypes[CORE_MAX_SCALAR_ARGS]{};
};

static_assert(std::is_trivially_copyable_v<ArgsDumpTaskMetadata>);
static_assert(std::is_standard_layout_v<ArgsDumpTaskMetadata>);
static_assert(std::is_trivially_copyable_v<ArgsDumpArgMask>);
static_assert(std::is_standard_layout_v<ArgsDumpArgMask>);
static_assert(sizeof(ArgsDumpTaskMetadata) == 2 * ARGS_DUMP_ARG_MASK_WORDS * sizeof(uint64_t) + CORE_MAX_SCALAR_ARGS);

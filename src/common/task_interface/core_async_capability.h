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

enum class CoreAsyncMode : uint8_t { SYNCHRONOUS = 1, DEFERRED = 2 };

// A kernel must declare its registration contract. Zero-filled storage is
// invalid; SYNCHRONOUS proves only that the kernel registers no defer_wait.
struct CoreAsyncCapability {
    uint32_t version{0};
    CoreAsyncMode mode{};
    uint8_t max_conditions{0};
    uint8_t condition_types{0};
    uint8_t reserved{0};
};

inline constexpr bool core_async_capability_valid(const CoreAsyncCapability &value) {
    if (value.version != 1 || value.reserved != 0) return false;
    if (value.mode == CoreAsyncMode::DEFERRED)
        return value.max_conditions >= 1 && value.max_conditions <= 64 && value.condition_types >= 1 &&
               value.condition_types <= 3;
    return value.mode == CoreAsyncMode::SYNCHRONOUS && value.max_conditions == 0 && value.condition_types == 0;
}

static_assert(sizeof(CoreAsyncCapability) == 8);
static_assert(std::is_trivially_copyable_v<CoreAsyncCapability> && std::is_standard_layout_v<CoreAsyncCapability>);

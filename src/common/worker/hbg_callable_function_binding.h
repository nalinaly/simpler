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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "task_interface/hbg_prebuilt_invocation.h"

namespace simpler::hbg {

enum class HbgCallableFunctionBindingStatus : uint32_t {
    Ok = 0,
    NullArgument,
    InvalidCapacity,
    InvalidFunctionId,
    InvalidAddress,
    DuplicateFunctionId,
    InvalidHash,
};

/**
 * Construct a complete callable-local function table transactionally.
 *
 * Every build starts from 1024 zeros. This is what permits two independent
 * callables to both own func_id 0 without stale addresses leaking from the
 * previously launched callable. The caller's output table/hash are committed
 * only after all ids and addresses have been validated.
 */
inline HbgCallableFunctionBindingStatus build_hbg_callable_function_binding(
    const std::vector<std::pair<int, uint64_t>> &kernel_addrs, uint64_t *out_table, size_t table_size,
    uint64_t *out_hash
) noexcept {
    if (out_table == nullptr || out_hash == nullptr) return HbgCallableFunctionBindingStatus::NullArgument;
    if (table_size != HBG_PREBUILT_FUNC_ID_COUNT) return HbgCallableFunctionBindingStatus::InvalidCapacity;

    std::array<uint64_t, HBG_PREBUILT_FUNC_ID_COUNT> candidate{};
    for (const auto &binding : kernel_addrs) {
        if (binding.first < 0 || binding.first >= static_cast<int>(candidate.size())) {
            return HbgCallableFunctionBindingStatus::InvalidFunctionId;
        }
        if (binding.second == 0) return HbgCallableFunctionBindingStatus::InvalidAddress;
        uint64_t &slot = candidate[static_cast<size_t>(binding.first)];
        if (slot != 0) return HbgCallableFunctionBindingStatus::DuplicateFunctionId;
        slot = binding.second;
    }

    const uint64_t hash = hbg_function_binding_hash(candidate.data(), candidate.size());
    if (hash == 0) return HbgCallableFunctionBindingStatus::InvalidHash;
    std::copy(candidate.begin(), candidate.end(), out_table);
    *out_hash = hash;
    return HbgCallableFunctionBindingStatus::Ok;
}

}  // namespace simpler::hbg

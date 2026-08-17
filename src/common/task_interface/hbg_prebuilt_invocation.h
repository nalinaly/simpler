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

#include "utils/fnv1a_64.h"

namespace simpler::hbg {

inline constexpr uint32_t HBG_PREBUILT_INVOCATION_MAGIC = 0x49474248U;  // "HBGI"
inline constexpr uint32_t HBG_PREBUILT_INVOCATION_ABI_VERSION = 2;
inline constexpr size_t HBG_PREBUILT_FUNC_ID_COUNT = 1024;

/**
 * Invocation-local scheduler inputs carried by the pristine runtime image.
 *
 * The complete table is data, not merely identity: the scheduler dereferences
 * these addresses after every restore. `function_binding_hash` is stored next
 * to that data so the HBG L1 entry can compare it with the launch header before
 * publishing a restored generation. The hash consumes the old eight-byte
 * reserved area, preserving the table offset and total ABI size.
 */
struct alignas(8) HbgPrebuiltInvocationState {
    uint32_t magic{0};
    uint32_t abi_version{0};
    uint32_t func_id_count{0};
    int32_t host_total_tasks{-1};
    uint64_t function_binding_hash{0};
    uint64_t func_id_to_addr[HBG_PREBUILT_FUNC_ID_COUNT]{};
};

static_assert(sizeof(HbgPrebuiltInvocationState) == 8216, "HBG prebuilt invocation ABI changed");
static_assert(
    offsetof(HbgPrebuiltInvocationState, function_binding_hash) == 16, "HBG function binding hash offset changed"
);
static_assert(offsetof(HbgPrebuiltInvocationState, func_id_to_addr) == 24, "HBG function table offset changed");
static_assert(
    std::is_standard_layout_v<HbgPrebuiltInvocationState>, "HBG prebuilt invocation state must be standard-layout"
);
static_assert(
    std::is_trivially_copyable_v<HbgPrebuiltInvocationState>, "HBG prebuilt invocation state must be byte-copyable"
);

inline uint64_t hbg_function_binding_hash(const uint64_t *func_id_to_addr, size_t func_id_count) noexcept {
    if (func_id_to_addr == nullptr || func_id_count != HBG_PREBUILT_FUNC_ID_COUNT) return 0;
    const uint64_t stable_func_id_count = HBG_PREBUILT_FUNC_ID_COUNT;
    uint64_t hash = common::utils::fnv1a_64(&stable_func_id_count, sizeof(stable_func_id_count));
    return common::utils::fnv1a_64_append(hash, func_id_to_addr, HBG_PREBUILT_FUNC_ID_COUNT * sizeof(*func_id_to_addr));
}

inline bool hbg_has_valid_prebuilt_invocation_state(const HbgPrebuiltInvocationState *state) noexcept {
    if (state == nullptr || state->magic != HBG_PREBUILT_INVOCATION_MAGIC ||
        state->abi_version != HBG_PREBUILT_INVOCATION_ABI_VERSION ||
        state->func_id_count != HBG_PREBUILT_FUNC_ID_COUNT || state->host_total_tasks < 0 ||
        state->function_binding_hash == 0) {
        return false;
    }
    return state->function_binding_hash ==
           hbg_function_binding_hash(state->func_id_to_addr, HBG_PREBUILT_FUNC_ID_COUNT);
}

/**
 * Replace one complete invocation snapshot.
 *
 * Invalid arguments do not mutate the prior state. On success the magic is
 * published last, so readers never accept a half-replaced table as a valid
 * generation. The helper performs no allocation and owns no source pointer.
 */
inline bool hbg_set_prebuilt_invocation_state(
    HbgPrebuiltInvocationState *state, const uint64_t *func_id_to_addr, size_t func_id_count, int32_t host_total_tasks
) noexcept {
    const uint64_t function_binding_hash = hbg_function_binding_hash(func_id_to_addr, func_id_count);
    if (state == nullptr || host_total_tasks < 0 || function_binding_hash == 0) return false;

    state->magic = 0;
    for (size_t i = 0; i < HBG_PREBUILT_FUNC_ID_COUNT; ++i) {
        state->func_id_to_addr[i] = func_id_to_addr[i];
    }
    state->host_total_tasks = host_total_tasks;
    state->func_id_count = static_cast<uint32_t>(HBG_PREBUILT_FUNC_ID_COUNT);
    state->function_binding_hash = function_binding_hash;
    state->abi_version = HBG_PREBUILT_INVOCATION_ABI_VERSION;
    state->magic = HBG_PREBUILT_INVOCATION_MAGIC;
    return true;
}

inline bool hbg_prebuilt_invocation_matches(
    const HbgPrebuiltInvocationState *state, uint64_t expected_function_binding_hash, int32_t expected_host_total_tasks
) noexcept {
    return hbg_has_valid_prebuilt_invocation_state(state) && expected_function_binding_hash != 0 &&
           state->function_binding_hash == expected_function_binding_hash &&
           state->host_total_tasks == expected_host_total_tasks;
}

}  // namespace simpler::hbg

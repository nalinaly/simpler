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
#include <cstring>
#include <type_traits>

#include "common/kernel_args.h"
#include "l1_limits.h"
#include "task_args.h"

inline constexpr uint32_t L1_AICPU_INVOCATION_ABI_VERSION = 1;
inline constexpr uint32_t L1_AICPU_REGISTER_ABI_VERSION = 2;

/** One callable-local func_id -> device-entry-address binding. */
struct L1CallableKernelAddr {
    int32_t func_id{-1};
    uint32_t reserved{0};
    uint64_t device_addr{0};
};

/**
 * L1-only callable registration image copied by
 * aclrtLaunchKernelWithHostArgs.
 *
 * Unlike the legacy RegisterCallableArgs used by L2, this payload also owns
 * the callable-local AICore address table.  PyPTO code generation numbers
 * func_id from zero independently for every @pl.program, so a context-global
 * func_id table cannot represent two programs.  The AICPU keeps this snapshot
 * beside the callable's orchestration SO and replays the selected mapping onto
 * the serialized Runtime at invocation time.
 */
struct L1RegisterCallableArgs {
    uint32_t abi_version{L1_AICPU_REGISTER_ABI_VERSION};
    uint32_t struct_size{0};
    int32_t callable_id{-1};
    uint32_t kernel_count{0};
    uint64_t callable_hash{0};
    uint64_t dev_orch_so_addr{0};
    uint64_t dev_orch_so_size{0};
    char device_orch_func_name[INIT_ARGS_MAX_ORCH_SYMBOL_NAME]{};
    char device_orch_config_name[INIT_ARGS_MAX_ORCH_SYMBOL_NAME]{};
    L1CallableKernelAddr kernel_addrs[L1_MAX_KERNELS_PER_CALLABLE]{};
};

static_assert(std::is_standard_layout_v<L1RegisterCallableArgs>, "L1 registration must be standard-layout");
static_assert(std::is_trivially_copyable_v<L1RegisterCallableArgs>, "L1 registration must be byte-copyable");
static_assert(
    sizeof(L1RegisterCallableArgs) <= 64U * 1024U, "L1 registration must fit the runtime host-args allocation contract"
);

inline bool IsValidL1RegisterCallable(const L1RegisterCallableArgs &args) {
    return args.abi_version == L1_AICPU_REGISTER_ABI_VERSION && args.struct_size == sizeof(args) &&
           args.callable_id >= 0 && args.callable_hash != 0 && args.kernel_count <= L1_MAX_KERNELS_PER_CALLABLE;
}

/**
 * Per-launch AICPU argument image copied by aclrtLaunchKernelWithHostArgs.
 *
 * kernel_args contains context-lifetime device addresses and platform state.
 * callable_id and orch_args are the invocation-specific snapshot. AICore uses
 * a separate context-lifetime device KernelArgs image and never retains a
 * pointer into this runtime-owned AICPU task argument image.
 */
struct alignas(64) L1AicpuInvocationArgs {
    uint32_t abi_version{L1_AICPU_INVOCATION_ABI_VERSION};
    uint32_t struct_size{0};
    int32_t callable_id{-1};
    uint32_t reserved{0};
    KernelArgs kernel_args{};
    ChipStorageTaskArgs orch_args{};
};

static_assert(std::is_standard_layout_v<L1AicpuInvocationArgs>, "L1 AICPU invocation must be standard-layout");
static_assert(std::is_trivially_copyable_v<L1AicpuInvocationArgs>, "L1 AICPU invocation must be byte-copyable");
static_assert(alignof(L1AicpuInvocationArgs) == 64, "L1 AICPU invocation must preserve Tensor alignment");
static_assert(
    sizeof(L1AicpuInvocationArgs) <= 64U * 1024U,
    "L1 AICPU invocation must fit the runtime host-args allocation contract"
);

inline L1AicpuInvocationArgs
MakeL1AicpuInvocationArgs(const KernelArgs &kernel_args, int32_t callable_id, const ChipStorageTaskArgs &orch_args) {
    L1AicpuInvocationArgs invocation;
    invocation.struct_size = sizeof(invocation);
    invocation.callable_id = callable_id;
    invocation.kernel_args = kernel_args;
    invocation.orch_args = orch_args;
    return invocation;
}

inline bool IsValidL1AicpuInvocation(const L1AicpuInvocationArgs &invocation) {
    return invocation.abi_version == L1_AICPU_INVOCATION_ABI_VERSION &&
           invocation.struct_size == sizeof(L1AicpuInvocationArgs) && invocation.reserved == 0;
}

/**
 * Naturally-aligned front matter of L1AicpuInvocationArgs.
 *
 * aclrtLaunchKernelWithHostArgs owns and copies the complete argument byte
 * image, but its AICPU task-argument pool does not promise the 64-byte
 * alignment required by ChipStorageTaskArgs/Tensor.  A platform entry must
 * therefore not form a L1AicpuInvocationArgs pointer directly over that
 * image.  Copy this small prefix first, then hand the raw orch-args bytes to
 * the TRB executor; its unique orchestrator worker copies them into one
 * context-lifetime aligned snapshot before typed access.
 */
struct L1AicpuInvocationPrefix {
    uint32_t abi_version{0};
    uint32_t struct_size{0};
    int32_t callable_id{-1};
    uint32_t reserved{0};
    KernelArgs kernel_args{};
};

static_assert(std::is_standard_layout_v<L1AicpuInvocationPrefix>, "L1 AICPU prefix must be standard-layout");
static_assert(std::is_trivially_copyable_v<L1AicpuInvocationPrefix>, "L1 AICPU prefix must be byte-copyable");
static_assert(
    offsetof(L1AicpuInvocationPrefix, abi_version) == offsetof(L1AicpuInvocationArgs, abi_version) &&
        offsetof(L1AicpuInvocationPrefix, struct_size) == offsetof(L1AicpuInvocationArgs, struct_size) &&
        offsetof(L1AicpuInvocationPrefix, callable_id) == offsetof(L1AicpuInvocationArgs, callable_id) &&
        offsetof(L1AicpuInvocationPrefix, reserved) == offsetof(L1AicpuInvocationArgs, reserved) &&
        offsetof(L1AicpuInvocationPrefix, kernel_args) == offsetof(L1AicpuInvocationArgs, kernel_args),
    "L1 AICPU prefix must match the invocation wire front matter"
);
static_assert(
    sizeof(L1AicpuInvocationPrefix) <= offsetof(L1AicpuInvocationArgs, orch_args),
    "L1 AICPU prefix must not overlap orch args"
);
static_assert(sizeof(L1AicpuInvocationPrefix) <= 256, "L1 AICPU prefix must remain stack-friendly");

inline bool ReadL1AicpuInvocationPrefix(const void *raw_args, L1AicpuInvocationPrefix *prefix) noexcept {
    if (raw_args == nullptr || prefix == nullptr) return false;
    std::memcpy(prefix, raw_args, sizeof(*prefix));
    return true;
}

inline bool IsValidL1AicpuInvocationPrefix(const L1AicpuInvocationPrefix &prefix) noexcept {
    return prefix.abi_version == L1_AICPU_INVOCATION_ABI_VERSION &&
           prefix.struct_size == sizeof(L1AicpuInvocationArgs) && prefix.reserved == 0;
}

inline const void *L1AicpuInvocationOrchArgsBytes(const void *raw_args) noexcept {
    if (raw_args == nullptr) return nullptr;
    return static_cast<const uint8_t *>(raw_args) + offsetof(L1AicpuInvocationArgs, orch_args);
}

inline bool HasValidL1OrchArgCounts(const ChipStorageTaskArgs &args) noexcept {
    return args.tensor_count_ >= 0 && args.tensor_count_ <= CHIP_MAX_TENSOR_ARGS && args.scalar_count_ >= 0 &&
           args.scalar_count_ <= CHIP_MAX_SCALAR_ARGS;
}

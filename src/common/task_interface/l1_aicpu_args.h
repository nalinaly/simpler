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

#include "common/kernel_args.h"
#include "l1_limits.h"
#include "task_args.h"

inline constexpr uint32_t L1_AICPU_INVOCATION_ABI_VERSION = 1;
inline constexpr uint32_t L1_AICPU_REGISTER_ABI_VERSION = 1;

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
           args.callable_id >= 0 && args.kernel_count <= L1_MAX_KERNELS_PER_CALLABLE;
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

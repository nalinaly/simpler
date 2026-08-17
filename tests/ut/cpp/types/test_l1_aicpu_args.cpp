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

#include <gtest/gtest.h>

#include <cstdint>

#include "l1_aicpu_args.h"

namespace {

TEST(L1AicpuInvocationArgs, BuilderCreatesAnIndependentVersionedSnapshot) {
    KernelArgs kernel_args;
    kernel_args.runtime_args = reinterpret_cast<Runtime *>(static_cast<uintptr_t>(0x12340000));
    kernel_args.regs = 0x22340000;

    ChipStorageTaskArgs orch_args;
    orch_args.tensor_count_ = 0;
    orch_args.scalar_count_ = 2;
    orch_args.scalars_[0] = 17;
    orch_args.scalars_[1] = 29;

    L1AicpuInvocationArgs invocation = MakeL1AicpuInvocationArgs(kernel_args, 7, orch_args);
    kernel_args.runtime_args = nullptr;
    orch_args.scalars_[0] = 99;

    EXPECT_TRUE(IsValidL1AicpuInvocation(invocation));
    EXPECT_EQ(invocation.abi_version, L1_AICPU_INVOCATION_ABI_VERSION);
    EXPECT_EQ(invocation.struct_size, sizeof(L1AicpuInvocationArgs));
    EXPECT_EQ(invocation.callable_id, 7);
    EXPECT_EQ(invocation.kernel_args.runtime_args, reinterpret_cast<Runtime *>(static_cast<uintptr_t>(0x12340000)));
    EXPECT_EQ(invocation.orch_args.scalar_count(), 2);
    EXPECT_EQ(invocation.orch_args.scalar(0), 17u);
    EXPECT_EQ(invocation.orch_args.scalar(1), 29u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&invocation) % alignof(L1AicpuInvocationArgs), 0u);
}

TEST(L1AicpuInvocationArgs, RejectsEveryHeaderMismatch) {
    L1AicpuInvocationArgs invocation;
    invocation.struct_size = sizeof(invocation);
    ASSERT_TRUE(IsValidL1AicpuInvocation(invocation));

    invocation.abi_version += 1;
    EXPECT_FALSE(IsValidL1AicpuInvocation(invocation));
    invocation.abi_version = L1_AICPU_INVOCATION_ABI_VERSION;

    invocation.struct_size -= 1;
    EXPECT_FALSE(IsValidL1AicpuInvocation(invocation));
    invocation.struct_size = sizeof(invocation);

    invocation.reserved = 1;
    EXPECT_FALSE(IsValidL1AicpuInvocation(invocation));
}

TEST(L1RegisterCallableArgs, VersionedCallableLocalKernelSnapshot) {
    L1RegisterCallableArgs args{};
    args.struct_size = sizeof(args);
    args.callable_id = 3;
    args.kernel_count = 2;
    args.kernel_addrs[0].func_id = 0;
    args.kernel_addrs[0].device_addr = 0x1000;
    args.kernel_addrs[1].func_id = 7;
    args.kernel_addrs[1].device_addr = 0x2000;

    EXPECT_TRUE(IsValidL1RegisterCallable(args));
    EXPECT_EQ(args.kernel_addrs[0].func_id, 0);
    EXPECT_EQ(args.kernel_addrs[1].device_addr, 0x2000u);
    EXPECT_LE(sizeof(args), 64U * 1024U);
}

TEST(L1RegisterCallableArgs, RejectsHeaderAndCapacityMismatch) {
    L1RegisterCallableArgs args{};
    args.struct_size = sizeof(args);
    args.callable_id = 0;
    ASSERT_TRUE(IsValidL1RegisterCallable(args));

    args.struct_size -= 1;
    EXPECT_FALSE(IsValidL1RegisterCallable(args));
    args.struct_size = sizeof(args);
    args.kernel_count = L1_MAX_KERNELS_PER_CALLABLE + 1;
    EXPECT_FALSE(IsValidL1RegisterCallable(args));
    args.kernel_count = 0;
    args.callable_id = -1;
    EXPECT_FALSE(IsValidL1RegisterCallable(args));
}

}  // namespace

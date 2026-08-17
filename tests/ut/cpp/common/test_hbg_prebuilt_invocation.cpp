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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "pto_runtime2.h"

namespace {

using FunctionTable = std::array<uint64_t, PTO2_PREBUILT_FUNC_ID_COUNT>;

TEST(HbgPrebuiltInvocationTest, InvalidInputDoesNotMutateExistingSnapshot) {
    PTO2Runtime runtime{};
    FunctionTable functions{};
    functions[0] = 0x12340000ULL;

    ASSERT_TRUE(runtime_set_prebuilt_invocation_state(&runtime, functions.data(), functions.size(), 7));
    const PTO2PrebuiltInvocationState before = runtime.prebuilt_invocation;

    EXPECT_FALSE(runtime_set_prebuilt_invocation_state(nullptr, functions.data(), functions.size(), 7));
    EXPECT_FALSE(runtime_set_prebuilt_invocation_state(&runtime, nullptr, functions.size(), 7));
    EXPECT_FALSE(runtime_set_prebuilt_invocation_state(&runtime, functions.data(), functions.size() - 1, 7));
    EXPECT_FALSE(runtime_set_prebuilt_invocation_state(&runtime, functions.data(), functions.size(), -1));
    EXPECT_EQ(std::memcmp(&runtime.prebuilt_invocation, &before, sizeof(before)), 0);
}

TEST(HbgPrebuiltInvocationTest, SnapshotOwnsFullCallableLocalFunctionTableAndTaskCount) {
    PTO2Runtime runtime{};
    FunctionTable first{};
    first[0] = 0x10000000ULL;
    first[17] = 0x17000000ULL;
    first.back() = 0xffff0000ULL;

    ASSERT_TRUE(runtime_set_prebuilt_invocation_state(&runtime, first.data(), first.size(), 37));
    ASSERT_TRUE(runtime_has_valid_prebuilt_invocation_state(&runtime));
    EXPECT_EQ(runtime.prebuilt_invocation.host_total_tasks, 37);
    EXPECT_EQ(runtime.prebuilt_invocation.func_id_to_addr[0], 0x10000000ULL);
    EXPECT_EQ(runtime.prebuilt_invocation.func_id_to_addr[17], 0x17000000ULL);
    EXPECT_EQ(runtime.prebuilt_invocation.func_id_to_addr[PTO2_PREBUILT_FUNC_ID_COUNT - 1], 0xffff0000ULL);

    first[0] = 0;
    first[17] = 0;
    EXPECT_EQ(runtime.prebuilt_invocation.func_id_to_addr[0], 0x10000000ULL);
    EXPECT_EQ(runtime.prebuilt_invocation.func_id_to_addr[17], 0x17000000ULL);
}

TEST(HbgPrebuiltInvocationTest, NewGenerationReplacesTheEntireFunctionTable) {
    PTO2Runtime runtime{};
    FunctionTable first{};
    first[0] = 0x11110000ULL;
    first[19] = 0x19190000ULL;
    ASSERT_TRUE(runtime_set_prebuilt_invocation_state(&runtime, first.data(), first.size(), 5));

    FunctionTable second{};
    second[0] = 0x22220000ULL;
    ASSERT_TRUE(runtime_set_prebuilt_invocation_state(&runtime, second.data(), second.size(), 9));

    EXPECT_EQ(runtime.prebuilt_invocation.func_id_to_addr[0], 0x22220000ULL);
    EXPECT_EQ(runtime.prebuilt_invocation.func_id_to_addr[19], 0);
    EXPECT_EQ(runtime.prebuilt_invocation.host_total_tasks, 9);
    EXPECT_TRUE(runtime_has_valid_prebuilt_invocation_state(&runtime));
}

TEST(HbgPrebuiltInvocationTest, ValidationRejectsCorruptMetadata) {
    PTO2Runtime runtime{};
    FunctionTable functions{};
    ASSERT_TRUE(runtime_set_prebuilt_invocation_state(&runtime, functions.data(), functions.size(), 0));

    ++runtime.prebuilt_invocation.abi_version;
    EXPECT_FALSE(runtime_has_valid_prebuilt_invocation_state(&runtime));
    --runtime.prebuilt_invocation.abi_version;

    --runtime.prebuilt_invocation.func_id_count;
    EXPECT_FALSE(runtime_has_valid_prebuilt_invocation_state(&runtime));
    ++runtime.prebuilt_invocation.func_id_count;

    runtime.prebuilt_invocation.host_total_tasks = -1;
    EXPECT_FALSE(runtime_has_valid_prebuilt_invocation_state(&runtime));
}

}  // namespace

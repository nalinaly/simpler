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

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "hbg_callable_function_binding.h"

namespace {

using simpler::hbg::build_hbg_callable_function_binding;
using simpler::hbg::HBG_PREBUILT_FUNC_ID_COUNT;
using simpler::hbg::HbgCallableFunctionBindingStatus;

TEST(HbgCallableFunctionBinding, RebuildsOneCallableLocalNamespaceFromZero) {
    std::array<uint64_t, HBG_PREBUILT_FUNC_ID_COUNT> table{};
    uint64_t first_hash = 0;
    const std::vector<std::pair<int, uint64_t>> first{{0, 0x1000}, {7, 0x7000}};
    ASSERT_EQ(
        build_hbg_callable_function_binding(first, table.data(), table.size(), &first_hash),
        HbgCallableFunctionBindingStatus::Ok
    );
    EXPECT_EQ(table[0], 0x1000u);
    EXPECT_EQ(table[7], 0x7000u);
    EXPECT_NE(first_hash, 0u);

    uint64_t second_hash = 0;
    const std::vector<std::pair<int, uint64_t>> second{{0, 0x9000}};
    ASSERT_EQ(
        build_hbg_callable_function_binding(second, table.data(), table.size(), &second_hash),
        HbgCallableFunctionBindingStatus::Ok
    );
    EXPECT_EQ(table[0], 0x9000u);
    EXPECT_EQ(table[7], 0u);
    EXPECT_NE(second_hash, first_hash);
}

TEST(HbgCallableFunctionBinding, InvalidInputLeavesPriorTableAndHashUntouched) {
    std::array<uint64_t, HBG_PREBUILT_FUNC_ID_COUNT> table{};
    table[2] = 0x2222;
    const auto before = table;
    uint64_t hash = 0xabcdef;

    EXPECT_EQ(
        build_hbg_callable_function_binding({{2, 0x2000}, {2, 0x3000}}, table.data(), table.size(), &hash),
        HbgCallableFunctionBindingStatus::DuplicateFunctionId
    );
    EXPECT_EQ(table, before);
    EXPECT_EQ(hash, 0xabcdefu);

    EXPECT_EQ(
        build_hbg_callable_function_binding({{-1, 0x2000}}, table.data(), table.size(), &hash),
        HbgCallableFunctionBindingStatus::InvalidFunctionId
    );
    EXPECT_EQ(
        build_hbg_callable_function_binding({{0, 0}}, table.data(), table.size(), &hash),
        HbgCallableFunctionBindingStatus::InvalidAddress
    );
    EXPECT_EQ(
        build_hbg_callable_function_binding({}, table.data(), table.size() - 1, &hash),
        HbgCallableFunctionBindingStatus::InvalidCapacity
    );
    EXPECT_EQ(table, before);
    EXPECT_EQ(hash, 0xabcdefu);
}

}  // namespace

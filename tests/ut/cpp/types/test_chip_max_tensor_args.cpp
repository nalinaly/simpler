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
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "callable.h"
#include "task_args.h"

namespace {

ChipTensor make_tensor(uint64_t addr) {
    ChipTensor t{};
    t.buffer.addr = addr;
    t.shapes[0] = 1;
    t.ndims = 1;
    t.dtype = DataType::FLOAT32;
    return t;
}

}  // namespace

TEST(ChipMaxTensorArgs, CapIsAtLeast256) {
    // The cap is the contract — the storage struct and the chip callable
    // signature_[] array both size off it.
    static_assert(CHIP_MAX_TENSOR_ARGS >= 256, "CHIP_MAX_TENSOR_ARGS must support 256 chip-level tensors");
    EXPECT_GE(CHIP_MAX_TENSOR_ARGS, 256);
}

TEST(ChipMaxTensorArgs, ChipStorageHoldsCapacity) {
    ChipStorageTaskArgs args;
    for (int i = 0; i < 256; ++i) {
        ASSERT_NO_THROW(args.add_tensor(make_tensor(static_cast<uint64_t>(0x1000 + i))));
    }
    EXPECT_EQ(args.tensor_count(), 256);
    EXPECT_EQ(args.tensor(0).buffer.addr, 0x1000u);
    EXPECT_EQ(args.tensor(255).buffer.addr, 0x1000u + 255);
}

TEST(ChipMaxTensorArgs, ChipCallableAcceptsCapacity) {
    std::vector<ArgDirection> signature(256, ArgDirection::IN);
    auto buffer = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        signature.data(), static_cast<int32_t>(signature.size()), 0, "composed", nullptr, 0, nullptr, nullptr, 0, ""
    );

    const auto &callable = *reinterpret_cast<const ChipCallable *>(buffer.data());
    EXPECT_EQ(callable.sig_count(), 256);
}

TEST(ChipMaxTensorArgs, ChipCallableCarriesScalarAritySeparatelyFromTensorSignature) {
    ArgDirection signature[] = {ArgDirection::IN, ArgDirection::OUT};
    auto buffer = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        signature, 2, 3, "composed", nullptr, 0, nullptr, nullptr, 0, ""
    );

    const auto &callable = *reinterpret_cast<const ChipCallable *>(buffer.data());
    EXPECT_EQ(callable.sig_count(), 2);
    EXPECT_EQ(callable.scalar_count(), 3);
}

TEST(ChipMaxTensorArgs, LegacyFactoryAndZeroedTailPaddingDecodeAsZeroScalars) {
    ArgDirection signature[] = {ArgDirection::IN};
    // This overload is the pre-L1 source API. Its byte layout and every old
    // field offset remain unchanged; the historical zero-filled tail padding
    // now carries scalar_count == 0.
    auto buffer = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        signature, 1, "legacy", nullptr, 0, nullptr, nullptr, 0, "legacy_config"
    );

    const auto &callable = *reinterpret_cast<const ChipCallable *>(buffer.data());
    EXPECT_EQ(callable.sig_count(), 1);
    EXPECT_EQ(callable.binary_size(), 0u);
    EXPECT_STREQ(callable.func_name(), "legacy");
    EXPECT_STREQ(callable.config_name(), "legacy_config");
    EXPECT_EQ(callable.child_count(), 0);
    EXPECT_EQ(callable.scalar_count(), 0);
    EXPECT_EQ(buffer.size(), offsetof(ChipCallable, storage_));
}

TEST(ChipMaxTensorArgs, ChipCallableOverflowReportsRequestedAndSupportedCounts) {
    const int32_t requested = CHIP_MAX_TENSOR_ARGS + 1;
    std::vector<ArgDirection> signature(static_cast<size_t>(requested), ArgDirection::IN);

    try {
        (void)make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
            signature.data(), requested, 0, "overflow", nullptr, 0, nullptr, nullptr, 0, ""
        );
        FAIL() << "expected signature capacity validation to fail";
    } catch (const std::invalid_argument &error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(std::to_string(requested)), std::string::npos);
        EXPECT_NE(message.find(std::to_string(CHIP_MAX_TENSOR_ARGS)), std::string::npos);
    }
}

TEST(ChipMaxTensorArgs, ChipStorageRejectsOverflow) {
    ChipStorageTaskArgs args;
    for (int i = 0; i < CHIP_MAX_TENSOR_ARGS; ++i) {
        args.add_tensor(make_tensor(static_cast<uint64_t>(i)));
    }
    EXPECT_THROW(args.add_tensor(make_tensor(0xDEAD)), std::out_of_range);
}

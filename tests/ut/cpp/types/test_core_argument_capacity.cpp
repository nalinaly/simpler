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

#include "aicpu/dump_arg_selection.h"
#include "kernel_callable_validation.h"
#include "task_args.h"

TEST(CoreArgumentCapacity, LargeCallableAndAllStorageSlotsRoundTrip) {
    static_assert(CORE_MAX_TENSOR_ARGS == 256);
    // 包含 CSA 的 132/133 参数以及新容量上界；检查真实序列化布局。
    for (int count : {32, 64, 132, 133, 256}) {
        std::vector<ArgDirection> signature(count, ArgDirection::IN);
        signature.back() = ArgDirection::INOUT;
        const uint64_t code = 0x123456789abcdef0;
        auto child = make_callable<CORE_MAX_TENSOR_ARGS>(signature.data(), count, &code, sizeof(code));
        const auto &core = *reinterpret_cast<const CoreCallable *>(child.data());
        EXPECT_EQ(core.sig_count(), count);
        EXPECT_EQ(core.sig(count - 1), ArgDirection::INOUT);
        EXPECT_EQ(std::memcmp(core.binary_data(), &code, sizeof(code)), 0);
        const int32_t id = 0;
        auto parent = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
            signature.data(), count, "large_core", nullptr, 0, &id, &child, 1, ""
        );
        EXPECT_TRUE(simpler::kernel::valid_kernel_callable_image(parent.data(), parent.size()));
    }
    TaskArgsTpl<ChipTensor, uint64_t, CORE_MAX_TENSOR_ARGS, CORE_MAX_SCALAR_ARGS> args;
    for (int i = 0; i < CORE_MAX_TENSOR_ARGS; ++i) {
        ChipTensor tensor{};
        tensor.buffer.addr = 0x1000 + i * 128;
        args.add_tensor(tensor);
    }
    EXPECT_THROW(args.add_tensor(ChipTensor{}), std::out_of_range);
    for (int i = 0; i < CORE_MAX_SCALAR_ARGS; ++i)
        args.add_scalar(0x2000 + i);
    EXPECT_THROW(args.add_scalar(0), std::out_of_range);
    for (int i = 0; i < CORE_MAX_TENSOR_ARGS; ++i)
        EXPECT_EQ(args.tensor(i).buffer.addr, 0x1000 + i * 128);
    EXPECT_EQ(args.scalar(CORE_MAX_SCALAR_ARGS - 1), 0x2000 + CORE_MAX_SCALAR_ARGS - 1);
    std::vector<ArgDirection> too_many(CORE_MAX_TENSOR_ARGS + 1, ArgDirection::IN);
    EXPECT_THROW(
        make_callable<CORE_MAX_TENSOR_ARGS>(too_many.data(), too_many.size(), nullptr, 0), std::invalid_argument
    );
}

TEST(CoreArgumentCapacity, DumpSelectionCoversWordBoundariesAndLastScalar) {
    DumpArgSelection selection;
    for (int index : {0, 31, 32, 63, 64, 127, 128, 132, 255, 256, 271})
        selection.mark_index(index);
    auto mask = selection.dump_arg_mask();
    for (int index : {0, 31, 32, 63, 64, 127, 128, 132, 255, 256, 271})
        EXPECT_TRUE(mask.test(index));
    for (int index : {-1, 1, 65, 129, 257, 270, 272, 320})
        EXPECT_FALSE(mask.test(index));
    selection.mark_index(-1);
    selection.mark_index(272);
    EXPECT_EQ(selection.dump_arg_mask(), mask);
    selection.clear();
    EXPECT_EQ(selection.dump_arg_mask(), ARGS_DUMP_ARG_MASK_NONE);
    selection.mark_all(256, 16);
    for (int i = 0; i < 272; ++i)
        EXPECT_TRUE(selection.dump_arg_mask().test(i));
    EXPECT_FALSE(selection.dump_arg_mask().test(272));

    // 共享标量源的歧义标记也要覆盖高字，clear 后不得残留上一次选择。
    selection.clear();
    selection.record_scalar_source(14, 0x1000, 3);
    selection.record_scalar_source(15, 0x1000, 3);
    EXPECT_TRUE(selection.mark_scalar_by_ptr(0x1000, 16, 256));
    EXPECT_TRUE(selection.dump_arg_mask().test(270));
    EXPECT_TRUE(selection.dump_arg_index_ambiguous_mask().test(270));
    EXPECT_FALSE(selection.dump_arg_mask().test(271));
    selection.clear();
    EXPECT_EQ(selection.dump_arg_index_ambiguous_mask(), ARGS_DUMP_ARG_MASK_NONE);
}

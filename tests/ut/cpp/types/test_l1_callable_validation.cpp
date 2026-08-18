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

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "l1_callable_validation.h"

namespace {

std::vector<uint8_t> make_valid_callable() {
    const ArgDirection child_signature[] = {ArgDirection::IN, ArgDirection::SCALAR};
    const uint8_t child_binary[] = {0x11, 0x22, 0x33, 0x44};
    std::vector<uint8_t> children[] = {make_callable<CORE_MAX_TENSOR_ARGS>(
        child_signature, 2, child_binary, static_cast<uint32_t>(sizeof(child_binary))
    )};

    const ArgDirection chip_signature[] = {ArgDirection::IN, ArgDirection::OUT};
    const uint8_t orchestration_binary[] = {0x7f, 'E', 'L', 'F'};
    const int32_t child_ids[] = {7};
    return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        chip_signature, 2, 1, "test_orchestration", orchestration_binary,
        static_cast<uint32_t>(sizeof(orchestration_binary)), child_ids, children, 1, "test_config"
    );
}

ChipCallable *as_callable(std::vector<uint8_t> &bytes) { return reinterpret_cast<ChipCallable *>(bytes.data()); }

}  // namespace

TEST(L1CallableValidation, AcceptsCanonicalFactoryImage) {
    std::vector<uint8_t> bytes = make_valid_callable();
    EXPECT_TRUE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));
}

TEST(L1CallableValidation, AcceptsLegacyZeroScalarPadding) {
    std::vector<uint8_t> bytes = make_valid_callable();
    as_callable(bytes)->scalar_count_ = 0;
    EXPECT_TRUE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));
}

TEST(L1CallableValidation, RejectsTruncationAndTrailingBytes) {
    std::vector<uint8_t> bytes = make_valid_callable();
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size() - 1));
    bytes.push_back(0);
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));
}

TEST(L1CallableValidation, RejectsInvalidCountsAndChipScalarSignature) {
    std::vector<uint8_t> bytes = make_valid_callable();
    as_callable(bytes)->sig_count_ = CHIP_MAX_TENSOR_ARGS + 1;
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));

    bytes = make_valid_callable();
    as_callable(bytes)->scalar_count_ = CHIP_MAX_SCALAR_ARGS + 1;
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));

    bytes = make_valid_callable();
    as_callable(bytes)->signature_[0] = ArgDirection::SCALAR;
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));
}

TEST(L1CallableValidation, RejectsUnterminatedOrInconsistentNames) {
    std::vector<uint8_t> bytes = make_valid_callable();
    as_callable(bytes)->func_name_len_ = CALLABLE_FUNC_NAME_MAX;
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));

    bytes = make_valid_callable();
    as_callable(bytes)->func_name_[1] = '\0';
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));
}

TEST(L1CallableValidation, RejectsNonCanonicalChildOffset) {
    std::vector<uint8_t> bytes = make_valid_callable();
    ++as_callable(bytes)->child_offsets_[0];
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));
}

TEST(L1CallableValidation, RejectsOutOfRangeAndDuplicateFunctionIds) {
    std::vector<uint8_t> bytes = make_valid_callable();
    as_callable(bytes)->child_func_ids_[0] = static_cast<int32_t>(L1_MAX_KERNELS_PER_CALLABLE);
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));

    const ArgDirection child_signature[] = {ArgDirection::IN};
    const uint8_t child_binary[] = {1, 2};
    std::vector<uint8_t> children[] = {
        make_callable<CORE_MAX_TENSOR_ARGS>(child_signature, 1, child_binary, sizeof(child_binary)),
        make_callable<CORE_MAX_TENSOR_ARGS>(child_signature, 1, child_binary, sizeof(child_binary)),
    };
    const ArgDirection chip_signature[] = {ArgDirection::IN};
    const uint8_t orchestration_binary[] = {1};
    const int32_t duplicate_ids[] = {3, 3};
    bytes = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        chip_signature, 1, 0, "orch", orchestration_binary, sizeof(orchestration_binary), duplicate_ids, children, 2,
        "cfg"
    );
    EXPECT_FALSE(simpler::l1::valid_callable_blob(as_callable(bytes), bytes.size()));
}

TEST(L1CallableValidation, RejectsChildBinaryOutsideBlob) {
    std::vector<uint8_t> bytes = make_valid_callable();
    ChipCallable *callable = as_callable(bytes);
    CoreCallable *child = reinterpret_cast<CoreCallable *>(callable->storage_ + callable->child_offsets_[0]);
    child->binary_size_ = UINT32_MAX;
    EXPECT_FALSE(simpler::l1::valid_callable_blob(callable, bytes.size()));
}

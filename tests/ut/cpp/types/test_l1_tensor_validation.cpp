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
#include <limits>

#include <gtest/gtest.h>

#include "l1_tensor_validation.h"

namespace {

Tensor make_contiguous_tensor() {
    Tensor tensor{};
    tensor.buffer = {0x1000, 24};
    tensor.ndims = 2;
    tensor.dtype = DataType::FLOAT32;
    tensor.shapes[0] = 2;
    tensor.shapes[1] = 3;
    tensor.strides[0] = 3;
    tensor.strides[1] = 1;
    tensor.extent_elem_cache = 6;
    tensor.is_contiguous = true;
    return tensor;
}

}  // namespace

TEST(L1TensorValidation, AcceptsRecomputedContiguousMetadata) {
    EXPECT_TRUE(valid_l1_tensor_metadata(make_contiguous_tensor()));
}

TEST(L1TensorValidation, AcceptsRecomputedPositiveStridedMetadata) {
    Tensor tensor = make_contiguous_tensor();
    tensor.buffer.size = 28;
    tensor.strides[0] = 4;
    tensor.extent_elem_cache = 7;
    tensor.is_contiguous = false;
    EXPECT_TRUE(valid_l1_tensor_metadata(tensor));
}

TEST(L1TensorValidation, RejectsForgedExtentCache) {
    Tensor tensor = make_contiguous_tensor();
    tensor.extent_elem_cache = 1;
    EXPECT_FALSE(valid_l1_tensor_metadata(tensor));
}

TEST(L1TensorValidation, RejectsForgedContiguousFlag) {
    Tensor tensor = make_contiguous_tensor();
    tensor.buffer.size = 28;
    tensor.strides[0] = 4;
    tensor.extent_elem_cache = 7;
    EXPECT_FALSE(valid_l1_tensor_metadata(tensor));
}

TEST(L1TensorValidation, RejectsZeroSizedTensorInV1) {
    Tensor tensor = make_contiguous_tensor();
    tensor.shapes[0] = 0;
    tensor.extent_elem_cache = 0;
    EXPECT_FALSE(valid_l1_tensor_metadata(tensor));
}

TEST(L1TensorValidation, RejectsNullAddress) {
    Tensor tensor = make_contiguous_tensor();
    tensor.buffer.addr = 0;
    EXPECT_FALSE(valid_l1_tensor_metadata(tensor));
}

TEST(L1TensorValidation, RejectsExtentOutsideBuffer) {
    Tensor tensor = make_contiguous_tensor();
    tensor.buffer.size = 20;
    EXPECT_FALSE(valid_l1_tensor_metadata(tensor));
}

TEST(L1TensorValidation, RejectsStartOffsetOverflow) {
    Tensor tensor = make_contiguous_tensor();
    tensor.start_offset = std::numeric_limits<uint64_t>::max() - 5;
    EXPECT_FALSE(valid_l1_tensor_metadata(tensor));
}

TEST(L1TensorValidation, RejectsExtentArithmeticOverflow) {
    Tensor tensor{};
    tensor.buffer = {0x1000, std::numeric_limits<uint64_t>::max()};
    tensor.ndims = 2;
    tensor.dtype = DataType::UINT8;
    tensor.shapes[0] = std::numeric_limits<uint32_t>::max();
    tensor.shapes[1] = std::numeric_limits<uint32_t>::max();
    tensor.strides[0] = std::numeric_limits<uint32_t>::max();
    tensor.strides[1] = std::numeric_limits<uint32_t>::max();
    tensor.extent_elem_cache = std::numeric_limits<uint64_t>::max();
    tensor.is_contiguous = false;
    EXPECT_FALSE(valid_l1_tensor_metadata(tensor));
}

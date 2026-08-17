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

#include <cstdint>
#include <limits>

#include "tensor.h"

inline bool valid_l1_tensor_metadata(const Tensor &tensor) noexcept {
    if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS || tensor.buffer.addr == 0 ||
        static_cast<uint8_t>(tensor.dtype) >= static_cast<uint8_t>(DataType::DATA_TYPE_NUM)) {
        return false;
    }

    uint64_t computed_extent = 1;
    uint64_t expected_stride = 1;
    bool expected_stride_overflowed = false;
    bool computed_contiguous = true;
    for (uint32_t reverse = tensor.ndims; reverse > 0; --reverse) {
        const uint32_t dim = reverse - 1;
        const uint64_t shape = tensor.shapes[dim];
        const uint64_t stride = tensor.strides[dim];
        if (shape == 0 || stride == 0) return false;

        if (expected_stride_overflowed || stride != expected_stride) computed_contiguous = false;

        const uint64_t shape_span = shape - 1;
        if (shape_span != 0 && stride > std::numeric_limits<uint64_t>::max() / shape_span) return false;
        const uint64_t dim_extent = shape_span * stride;
        if (dim_extent > std::numeric_limits<uint64_t>::max() - computed_extent) return false;
        computed_extent += dim_extent;

        if (!expected_stride_overflowed) {
            if (expected_stride > std::numeric_limits<uint64_t>::max() / shape) {
                expected_stride_overflowed = true;
                computed_contiguous = false;
            } else {
                expected_stride *= shape;
            }
        }
    }

    if (tensor.extent_elem_cache != computed_extent || tensor.is_contiguous != computed_contiguous) return false;

    const uint64_t element_size = get_element_size(tensor.dtype);
    if (element_size == 0 || tensor.start_offset > std::numeric_limits<uint64_t>::max() - computed_extent) {
        return false;
    }
    const uint64_t extent_end = tensor.start_offset + computed_extent;
    return extent_end <= std::numeric_limits<uint64_t>::max() / element_size &&
           extent_end * element_size <= tensor.buffer.size;
}

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

#include "task_interface/task_args.h"
#include "utils/fnv1a_64.h"

namespace simpler::hbg {

template <typename T>
inline uint64_t hbg_hash_value(uint64_t hash, const T &value) noexcept {
    return common::utils::fnv1a_64_append(hash, &value, sizeof(value));
}

/**
 * Hash only the semantic tensor/scalar state baked into one HBG graph image.
 *
 * Raw Tensor bytes are deliberately not hashed: reserved padding and unused
 * dimensions are not part of an invocation identity.  Every field that can
 * change an address, dependency, shape, layout, dtype, or scalar value is
 * appended with a fixed-width representation and in call order.
 */
inline uint64_t hbg_argument_snapshot_hash(const ChipStorageTaskArgs &args) noexcept {
    if (args.tensor_count_ < 0 || args.tensor_count_ > CHIP_MAX_TENSOR_ARGS || args.scalar_count_ < 0 ||
        args.scalar_count_ > CHIP_MAX_SCALAR_ARGS) {
        return 0;
    }

    uint64_t hash = common::utils::fnv1a_64(&args.tensor_count_, sizeof(args.tensor_count_));
    hash = hbg_hash_value(hash, args.scalar_count_);
    for (int32_t index = 0; index < args.tensor_count_; ++index) {
        const ChipTensor &tensor = args.tensor(index);
        if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS) return 0;

        hash = hbg_hash_value(hash, tensor.buffer.addr);
        hash = hbg_hash_value(hash, tensor.buffer.size);
        hash = hbg_hash_value(hash, tensor.owner_task_id.raw);
        hash = hbg_hash_value(hash, tensor.start_offset);
        hash = hbg_hash_value(hash, tensor.version);
        hash = hbg_hash_value(hash, tensor.ndims);
        const uint8_t dtype = static_cast<uint8_t>(tensor.dtype);
        const uint8_t manual_dep = tensor.manual_dep ? 1U : 0U;
        const uint8_t is_contiguous = tensor.is_contiguous ? 1U : 0U;
        hash = hbg_hash_value(hash, dtype);
        hash = hbg_hash_value(hash, manual_dep);
        hash = hbg_hash_value(hash, is_contiguous);
        hash = hbg_hash_value(hash, tensor.address_space);
        for (uint32_t dim = 0; dim < tensor.ndims; ++dim) {
            hash = hbg_hash_value(hash, tensor.shapes[dim]);
        }
        hash = hbg_hash_value(hash, tensor.extent_elem_cache);
        for (uint32_t dim = 0; dim < tensor.ndims; ++dim) {
            hash = hbg_hash_value(hash, tensor.strides[dim]);
        }
    }
    for (int32_t index = 0; index < args.scalar_count_; ++index) {
        hash = hbg_hash_value(hash, args.scalar(index));
    }
    return hash;
}

}  // namespace simpler::hbg

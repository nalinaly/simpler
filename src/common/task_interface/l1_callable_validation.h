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
/**
 * Bounds-first validation for an untrusted L1 ChipCallable byte image.
 *
 * The historical L2/L3 registration path receives an in-process object and
 * computes its size by following the object's flexible-array offsets.  L1 is
 * a public C ABI, so a caller can provide arbitrary bytes.  Its prepare entry
 * therefore carries the exact byte count and must validate the complete
 * canonical layout before hashing, uploading, or dereferencing a child.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "callable.h"
#include "l1_limits.h"

namespace simpler::l1 {

inline bool valid_callable_name(const char *name, uint32_t length) noexcept {
    if (name == nullptr || length >= CALLABLE_FUNC_NAME_MAX || name[length] != '\0') return false;
    return length == 0 || std::memchr(name, '\0', length) == nullptr;
}

inline bool valid_arg_direction(ArgDirection direction, bool scalar_allowed) noexcept {
    switch (direction) {
    case ArgDirection::SCALAR:
        return scalar_allowed;
    case ArgDirection::IN:
    case ArgDirection::OUT:
    case ArgDirection::INOUT:
        return true;
    default:
        return false;
    }
}

inline bool checked_align_up(size_t value, size_t alignment, size_t *result) noexcept {
    if (result == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return false;
    }
    *result = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

/**
 * Validate a canonical ChipCallable image for the L1 prepare ABI.
 *
 * `callable_size` must be the exact vector/blob length, not merely a lower
 * bound.  Requiring the canonical make_callable() layout rejects overlapping
 * children, hidden trailing payloads, and ambiguous hashes.  The pointer must
 * be naturally aligned because even reading the fixed header through a
 * ChipCallable pointer would otherwise be undefined behaviour.
 */
inline bool valid_callable_blob(const ChipCallable *callable, size_t callable_size) noexcept {
    constexpr size_t kHeaderSize = offsetof(ChipCallable, storage_);
    constexpr size_t kChildHeaderSize = CoreCallable::binary_data_offset();

    if (callable == nullptr || callable_size < kHeaderSize ||
        reinterpret_cast<uintptr_t>(callable) % alignof(ChipCallable) != 0) {
        return false;
    }
    if (callable->sig_count_ < 0 || callable->sig_count_ > CHIP_MAX_TENSOR_ARGS || callable->scalar_count_ < 0 ||
        callable->scalar_count_ > CHIP_MAX_SCALAR_ARGS || callable->child_count_ < 0 ||
        static_cast<uint32_t>(callable->child_count_) > L1_MAX_KERNELS_PER_CALLABLE) {
        return false;
    }
    if (!valid_callable_name(callable->func_name_, callable->func_name_len_) ||
        !valid_callable_name(callable->config_name_, callable->config_name_len_)) {
        return false;
    }
    for (int32_t i = 0; i < callable->sig_count_; ++i) {
        // ChipStorageTaskArgs stores tensors and scalars in separate arrays.
        // The ChipCallable signature is therefore tensor-only in L1.
        if (!valid_arg_direction(callable->signature_[i], false)) return false;
    }

    const size_t storage_size = callable_size - kHeaderSize;
    if (callable->binary_size_ == 0 || static_cast<size_t>(callable->binary_size_) > storage_size) return false;

    size_t storage_used = static_cast<size_t>(callable->binary_size_);
    std::array<bool, L1_MAX_KERNELS_PER_CALLABLE> seen_func_ids{};
    for (int32_t i = 0; i < callable->child_count_; ++i) {
        const int32_t func_id = callable->child_func_ids_[i];
        if (func_id < 0 || static_cast<uint32_t>(func_id) >= L1_MAX_KERNELS_PER_CALLABLE || seen_func_ids[func_id]) {
            return false;
        }
        seen_func_ids[func_id] = true;

        size_t expected_offset = 0;
        if (!checked_align_up(storage_used, CALLABLE_ALIGN, &expected_offset) ||
            static_cast<size_t>(callable->child_offsets_[i]) != expected_offset || expected_offset > storage_size ||
            kChildHeaderSize > storage_size - expected_offset) {
            return false;
        }

        const auto *child_bytes = reinterpret_cast<const uint8_t *>(callable->storage_) + expected_offset;
        if (reinterpret_cast<uintptr_t>(child_bytes) % alignof(CoreCallable) != 0) return false;
        const auto *child = reinterpret_cast<const CoreCallable *>(child_bytes);
        if (child->sig_count_ < 0 || child->sig_count_ > CORE_MAX_TENSOR_ARGS || child->binary_size_ == 0) {
            return false;
        }
        for (int32_t sig = 0; sig < child->sig_count_; ++sig) {
            if (!valid_arg_direction(child->signature_[sig], true)) return false;
        }
        const size_t child_binary_size = static_cast<size_t>(child->binary_size_);
        if (child_binary_size > storage_size - expected_offset - kChildHeaderSize) return false;
        storage_used = expected_offset + kChildHeaderSize + child_binary_size;
    }

    return storage_used == storage_size;
}

}  // namespace simpler::l1

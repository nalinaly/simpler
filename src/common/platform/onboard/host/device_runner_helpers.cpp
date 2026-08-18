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
 * Onboard host common helpers — `KernelArgsHelper` implementation.
 *
 * Linked into both a2a3 and a5 `libhost_runtime.so`. The arch-specific
 * `KernelArgs` layout is brought in via `common/kernel_args.h` on the
 * include path (each arch CMake adds the right one).
 */

#include "device_runner_helpers.h"

#include <runtime/rt.h>

#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "common/unified_log.h"

namespace {

size_t first_different_byte(const void *lhs, const void *rhs, size_t bytes) {
    const auto *left = static_cast<const unsigned char *>(lhs);
    const auto *right = static_cast<const unsigned char *>(rhs);
    for (size_t i = 0; i < bytes; ++i) {
        if (left[i] != right[i]) return i;
    }
    return bytes;
}

}  // namespace

int KernelArgsHelper::init_runtime_args(const Runtime &host_runtime, MemoryAllocator &allocator) {
    allocator_ = &allocator;

    // Only the device-read prefix of Runtime crosses to the device: trb copies
    // its `dev` descriptor (offset 0), hbg copies the whole object. Both start
    // at &host_runtime; runtime_device_copy_size() picks the right length per
    // runtime variant so this shared path stays runtime-agnostic.
    const uint64_t runtime_size = runtime_device_copy_size(host_runtime);
    if (args.runtime_args == nullptr) {
        void *runtime_dev = allocator_->alloc(runtime_size);
        if (runtime_dev == nullptr) {
            LOG_ERROR("Alloc for runtime_args failed");
            return -1;
        }
        args.runtime_args = reinterpret_cast<Runtime *>(runtime_dev);
    }
    int rc = rtMemcpy(args.runtime_args, runtime_size, &host_runtime, runtime_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy for runtime failed: %d", rc);
        if (allocator_->free(args.runtime_args) == 0) args.runtime_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::finalize_runtime_args() {
    if (args.runtime_args != nullptr && allocator_ != nullptr) {
        int rc = allocator_->free(args.runtime_args);
        if (rc == 0) args.runtime_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::init_device_kernel_args(MemoryAllocator &allocator) {
    allocator_ = &allocator;
    if (device_k_args_ == nullptr) {
        void *dev_ptr = allocator_->alloc(sizeof(KernelArgs));
        if (dev_ptr == nullptr) {
            LOG_ERROR("Alloc for device KernelArgs failed");
            return -1;
        }
        device_k_args_ = reinterpret_cast<KernelArgs *>(dev_ptr);
    }
    int rc = rtMemcpy(device_k_args_, sizeof(KernelArgs), &args, sizeof(KernelArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy for KernelArgs failed: %d", rc);
        if (allocator_->free(device_k_args_) == 0) device_k_args_ = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::validate_device_copies(const Runtime &host_runtime) const {
    if (args.runtime_args == nullptr || device_k_args_ == nullptr) {
        LOG_ERROR("L1 device-copy validation requires prepared Runtime and KernelArgs images");
        return -1;
    }

    const uint64_t runtime_bytes_u64 = runtime_device_copy_size(host_runtime);
    if (runtime_bytes_u64 == 0 || runtime_bytes_u64 > std::numeric_limits<size_t>::max()) {
        LOG_ERROR(
            "L1 Runtime device-copy validation has invalid byte size: %llu",
            static_cast<unsigned long long>(runtime_bytes_u64)
        );
        return -1;
    }
    const size_t runtime_bytes = static_cast<size_t>(runtime_bytes_u64);
    std::vector<unsigned char> runtime_shadow;
    try {
        runtime_shadow.resize(runtime_bytes);
    } catch (...) {
        LOG_ERROR("L1 Runtime device-copy validation could not allocate %zu host bytes", runtime_bytes);
        return -1;
    }

    int rc = rtMemcpy(runtime_shadow.data(), runtime_bytes, args.runtime_args, runtime_bytes, RT_MEMCPY_DEVICE_TO_HOST);
    if (rc != 0) {
        LOG_ERROR("L1 Runtime D2H validation copy failed: %d", rc);
        return rc;
    }
    const size_t runtime_diff = first_different_byte(runtime_shadow.data(), &host_runtime, runtime_bytes);
    if (runtime_diff != runtime_bytes) {
        LOG_ERROR(
            "L1 Runtime device image differs at byte %zu (device Runtime=%p, bytes=%zu)", runtime_diff,
            static_cast<void *>(args.runtime_args), runtime_bytes
        );
        return -1;
    }

    KernelArgs kernel_args_shadow{};
    rc = rtMemcpy(
        &kernel_args_shadow, sizeof(kernel_args_shadow), device_k_args_, sizeof(kernel_args_shadow),
        RT_MEMCPY_DEVICE_TO_HOST
    );
    if (rc != 0) {
        LOG_ERROR("L1 KernelArgs D2H validation copy failed: %d", rc);
        return rc;
    }
    const size_t kernel_args_diff = first_different_byte(&kernel_args_shadow, &args, sizeof(args));
    if (kernel_args_diff != sizeof(args)) {
        LOG_ERROR(
            "L1 KernelArgs device image differs at byte %zu (device KernelArgs=%p, host Runtime=%p, "
            "device Runtime=%p, host regs=0x%llx, device regs=0x%llx)",
            kernel_args_diff, static_cast<void *>(device_k_args_), static_cast<void *>(args.runtime_args),
            static_cast<void *>(kernel_args_shadow.runtime_args), static_cast<unsigned long long>(args.regs),
            static_cast<unsigned long long>(kernel_args_shadow.regs)
        );
        return -1;
    }

    LOG_INFO(
        "L1 persistent device images verified: Runtime=%p bytes=%zu KernelArgs=%p bytes=%zu regs=0x%llx",
        static_cast<void *>(args.runtime_args), runtime_bytes, static_cast<void *>(device_k_args_), sizeof(args),
        static_cast<unsigned long long>(args.regs)
    );
    return 0;
}

int KernelArgsHelper::finalize_device_kernel_args() {
    if (device_k_args_ != nullptr && allocator_ != nullptr) {
        int rc = allocator_->free(device_k_args_);
        if (rc == 0) device_k_args_ = nullptr;
        return rc;
    }
    return 0;
}

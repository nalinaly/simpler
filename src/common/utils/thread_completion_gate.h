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

#include <atomic>
#include <cstdint>

namespace simpler {

class ThreadCompletionGate {
public:
    template <typename Finalize>
    void arrive_and_finalize_if_last(int32_t thread_count, Finalize finalize) {
        int32_t previous = arrived_.fetch_add(1, std::memory_order_acq_rel);
        if (previous + 1 != thread_count) {
            return;
        }

        finalize();
        finalization_ready_.store(true, std::memory_order_release);
    }

    /** Wait until the last arriver has completed the per-run finalizer. */
    void wait_for_finalization() const {
        while (!finalization_ready_.load(std::memory_order_acquire)) {}
    }

    /**
     * Publish that this participant has taken every post-finalization snapshot.
     * Exactly the last departing participant owns shared-state cleanup/deinit.
     */
    bool depart_and_claim_cleanup_if_last(int32_t thread_count) {
        int32_t previous = departed_.fetch_add(1, std::memory_order_acq_rel);
        return previous + 1 == thread_count;
    }

    void reset() {
        arrived_.store(0, std::memory_order_release);
        departed_.store(0, std::memory_order_release);
        finalization_ready_.store(false, std::memory_order_release);
    }

private:
    std::atomic<int32_t> arrived_{0};
    std::atomic<int32_t> departed_{0};
    std::atomic<bool> finalization_ready_{false};
};

}  // namespace simpler

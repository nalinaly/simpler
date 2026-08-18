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

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "pto_runtime_c_api.h"

/**
 * A DeviceRunner has exactly one ownership model for its whole lifetime.
 * L2Owned retains the historical PyPTO-owned-device/reset semantics;
 * L1Borrowed borrows the caller's already-current device and may only release
 * resources created by the L1 context itself.
 */
enum class DeviceExecutionMode : uint8_t {
    Uninitialized = 0,
    L2Owned,
    L1Borrowed,
    Closed,
};

class DeviceExecutionModeState {
public:
    int claim_l2_owned();
    int claim_l1_borrowed();
    int abort_l1_initialization();
    int mark_closed();

    DeviceExecutionMode mode() const;
    bool accepts_l2_calls() const;
    bool accepts_l1_calls() const;
    bool requires_explicit_l1_close() const;

private:
    int claim(DeviceExecutionMode requested);

    mutable std::mutex mutex_;
    DeviceExecutionMode mode_{DeviceExecutionMode::Uninitialized};
};

/**
 * Runtime operations owned by the L1 context lifecycle.
 *
 * Deliberately absent from this table: device/stream synchronization, device
 * reset, ACL finalization, capture queries, model handles, and stream-to-model
 * attachment. Keeping those operations unrepresentable makes the host-only
 * lifecycle tests an architectural guard instead of a mock that can silently
 * exercise forbidden behavior.
 */
struct L1RuntimeOps {
    void *context{nullptr};
    int (*get_current_device)(void *context, int *device_id){nullptr};
    int (*create_hidden_stream)(void *context, void **stream){nullptr};
    int (*destroy_hidden_stream)(void *context, void *stream){nullptr};
    int (*create_event)(void *context, void **event){nullptr};
    int (*destroy_event)(void *context, void *event){nullptr};

    bool valid() const {
        return get_current_device != nullptr && create_hidden_stream != nullptr && destroy_hidden_stream != nullptr &&
               create_event != nullptr && destroy_event != nullptr;
    }
};

enum class L1ContextPhase : uint8_t {
    New = 0,
    Initializing,
    Collecting,
    ReadyEnqueued,
    Sealed,
    Poisoned,
    Closing,
    Closed,
};

enum class L1EventKind : size_t {
    PrepareTail = 0,
    Start,
    AicoreDone,
    SerialTail,
    Count,
};

/**
 * Context-lifetime state for the borrowed L1 execution model.
 *
 * This object owns the persistent hidden AICore stream and event set. Every
 * graph-visible persistent execution resource belongs here rather than in a
 * per-invocation object. The caller stream is never stored or destroyed;
 * every launch receives it as a borrowed argument.
 *
 * The destructor intentionally performs no runtime calls. If a caller skips
 * explicit close while an ACLGraph can still reference these handles, freeing
 * them would create a use-after-free. The public C API therefore refuses to
 * destroy an unclosed L1 DeviceRunner and conservatively leaves it alive.
 */
class L1ExecutionState {
public:
    L1ExecutionState() = default;
    ~L1ExecutionState() = default;
    L1ExecutionState(const L1ExecutionState &) = delete;
    L1ExecutionState &operator=(const L1ExecutionState &) = delete;

    int initialize(int requested_device_id, const L1RuntimeOps &ops);
    int mark_ready_enqueued();
    int seal();
    int begin_close();
    void poison(int runtime_error);
    int close();

    L1ContextPhase phase() const;
    bool accepts_dispatch() const;
    int device_id() const;
    int last_runtime_error() const;
    bool has_live_resources() const;
    void *hidden_aicore_stream() const;
    void *event(L1EventKind kind) const;

private:
    int fail_runtime_call_locked(int runtime_error);
    int cleanup_owned_resources_locked();
    bool has_live_resources_locked() const;

    mutable std::mutex mutex_;
    L1ContextPhase phase_{L1ContextPhase::New};
    int device_id_{-1};
    int last_runtime_error_{0};
    bool device_claimed_{false};
    L1RuntimeOps ops_{};
    void *hidden_aicore_stream_{nullptr};
    std::array<void *, static_cast<size_t>(L1EventKind::Count)> events_{};
};

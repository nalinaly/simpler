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

#include "l1_execution_state.h"

#include <unordered_set>

namespace {

std::mutex &l1_device_claim_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_set<int> &l1_claimed_devices() {
    static std::unordered_set<int> devices;
    return devices;
}

int claim_l1_device(int device_id) noexcept {
    std::lock_guard<std::mutex> lock(l1_device_claim_mutex());
    try {
        return l1_claimed_devices().insert(device_id).second ? 0 : PTO_RUNTIME_ERR_INVALID_STATE;
    } catch (...) {
        return PTO_RUNTIME_ERR_RUNTIME_FAILURE;
    }
}

void release_l1_device(int device_id) noexcept {
    std::lock_guard<std::mutex> lock(l1_device_claim_mutex());
    l1_claimed_devices().erase(device_id);
}

}  // namespace

int DeviceExecutionModeState::claim(DeviceExecutionMode requested) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != DeviceExecutionMode::Uninitialized) {
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    mode_ = requested;
    return 0;
}

int DeviceExecutionModeState::claim_l2_owned() { return claim(DeviceExecutionMode::L2Owned); }

int DeviceExecutionModeState::claim_l1_borrowed() { return claim(DeviceExecutionMode::L1Borrowed); }

int DeviceExecutionModeState::abort_l1_initialization() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != DeviceExecutionMode::L1Borrowed) {
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    mode_ = DeviceExecutionMode::Uninitialized;
    return 0;
}

int DeviceExecutionModeState::mark_closed() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ == DeviceExecutionMode::Closed) {
        return 0;
    }
    mode_ = DeviceExecutionMode::Closed;
    return 0;
}

DeviceExecutionMode DeviceExecutionModeState::mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

bool DeviceExecutionModeState::accepts_l2_calls() const { return mode() == DeviceExecutionMode::L2Owned; }

bool DeviceExecutionModeState::accepts_l1_calls() const { return mode() == DeviceExecutionMode::L1Borrowed; }

bool DeviceExecutionModeState::requires_explicit_l1_close() const { return mode() == DeviceExecutionMode::L1Borrowed; }

int L1ExecutionState::initialize(int requested_device_id, const L1RuntimeOps &ops) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ != L1ContextPhase::New || requested_device_id < 0 || !ops.valid()) {
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    }

    phase_ = L1ContextPhase::Initializing;
    ops_ = ops;
    last_runtime_error_ = 0;

    int current_device_id = -1;
    int rc = ops_.get_current_device(ops_.context, &current_device_id);
    if (rc != 0) {
        return fail_runtime_call_locked(rc);
    }
    if (current_device_id != requested_device_id) {
        phase_ = L1ContextPhase::New;
        ops_ = {};
        return PTO_RUNTIME_ERR_DEVICE_MISMATCH;
    }
    device_id_ = requested_device_id;

    rc = claim_l1_device(device_id_);
    if (rc != 0) {
        phase_ = L1ContextPhase::New;
        device_id_ = -1;
        ops_ = {};
        return rc;
    }
    device_claimed_ = true;

    rc = ops_.create_hidden_stream(ops_.context, &hidden_aicore_stream_);
    if (rc != 0 || hidden_aicore_stream_ == nullptr) {
        return fail_runtime_call_locked(rc != 0 ? rc : PTO_RUNTIME_ERR_RUNTIME_FAILURE);
    }

    for (void *&event_handle : events_) {
        rc = ops_.create_event(ops_.context, &event_handle);
        if (rc != 0 || event_handle == nullptr) {
            return fail_runtime_call_locked(rc != 0 ? rc : PTO_RUNTIME_ERR_RUNTIME_FAILURE);
        }
    }

    phase_ = L1ContextPhase::Collecting;
    return 0;
}

int L1ExecutionState::mark_ready_enqueued() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ != L1ContextPhase::Collecting && phase_ != L1ContextPhase::ReadyEnqueued) {
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    phase_ = L1ContextPhase::ReadyEnqueued;
    return 0;
}

int L1ExecutionState::seal() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ != L1ContextPhase::Collecting && phase_ != L1ContextPhase::ReadyEnqueued &&
        phase_ != L1ContextPhase::Sealed) {
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    // ReadyEnqueued preserves append-only callable admission. Sealed remains
    // an accepted compatibility input state.
    phase_ = L1ContextPhase::ReadyEnqueued;
    return 0;
}

int L1ExecutionState::begin_close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ == L1ContextPhase::Closing || phase_ == L1ContextPhase::Closed) {
        return 0;
    }
    if (phase_ != L1ContextPhase::Collecting && phase_ != L1ContextPhase::ReadyEnqueued &&
        phase_ != L1ContextPhase::Sealed && phase_ != L1ContextPhase::Poisoned) {
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    phase_ = L1ContextPhase::Closing;
    return 0;
}

void L1ExecutionState::poison(int runtime_error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ == L1ContextPhase::Closed) {
        return;
    }
    last_runtime_error_ = runtime_error;
    if (phase_ != L1ContextPhase::Closing) {
        phase_ = L1ContextPhase::Poisoned;
    }
}

int L1ExecutionState::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ == L1ContextPhase::Closed) {
        return 0;
    }
    if (phase_ == L1ContextPhase::Initializing) {
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    const bool close_intent_latched = phase_ == L1ContextPhase::Closing;
    if (!has_live_resources_locked()) {
        if (device_claimed_) {
            release_l1_device(device_id_);
            device_claimed_ = false;
        }
        phase_ = L1ContextPhase::Closed;
        device_id_ = -1;
        ops_ = {};
        return 0;
    }

    int current_device_id = -1;
    int rc = ops_.get_current_device(ops_.context, &current_device_id);
    if (rc != 0) {
        last_runtime_error_ = rc;
        phase_ = close_intent_latched ? L1ContextPhase::Closing : L1ContextPhase::Poisoned;
        return PTO_RUNTIME_ERR_RUNTIME_FAILURE;
    }
    if (current_device_id != device_id_) {
        return PTO_RUNTIME_ERR_DEVICE_MISMATCH;
    }

    rc = cleanup_owned_resources_locked();
    if (has_live_resources_locked()) {
        phase_ = close_intent_latched ? L1ContextPhase::Closing : L1ContextPhase::Poisoned;
        return rc != 0 ? rc : PTO_RUNTIME_ERR_RUNTIME_FAILURE;
    }

    if (device_claimed_) {
        release_l1_device(device_id_);
        device_claimed_ = false;
    }

    phase_ = L1ContextPhase::Closed;
    device_id_ = -1;
    ops_ = {};
    return rc;
}

L1ContextPhase L1ExecutionState::phase() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return phase_;
}

bool L1ExecutionState::accepts_dispatch() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return phase_ == L1ContextPhase::Collecting || phase_ == L1ContextPhase::ReadyEnqueued ||
           phase_ == L1ContextPhase::Sealed;
}

int L1ExecutionState::device_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return device_id_;
}

int L1ExecutionState::last_runtime_error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_runtime_error_;
}

bool L1ExecutionState::has_live_resources() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return has_live_resources_locked();
}

void *L1ExecutionState::hidden_aicore_stream() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hidden_aicore_stream_;
}

void *L1ExecutionState::event(L1EventKind kind) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t index = static_cast<size_t>(kind);
    return index < events_.size() ? events_[index] : nullptr;
}

int L1ExecutionState::fail_runtime_call_locked(int runtime_error) {
    last_runtime_error_ = runtime_error;
    const int cleanup_rc = cleanup_owned_resources_locked();
    if (has_live_resources_locked()) {
        phase_ = L1ContextPhase::Poisoned;
    } else {
        if (device_claimed_) {
            release_l1_device(device_id_);
            device_claimed_ = false;
        }
        phase_ = L1ContextPhase::New;
        device_id_ = -1;
        ops_ = {};
    }
    if (cleanup_rc != 0) {
        return cleanup_rc;
    }
    return PTO_RUNTIME_ERR_RUNTIME_FAILURE;
}

int L1ExecutionState::cleanup_owned_resources_locked() {
    int first_error = 0;
    auto record_error = [this, &first_error](int rc) {
        if (rc != 0) {
            last_runtime_error_ = rc;
            if (first_error == 0) {
                first_error = PTO_RUNTIME_ERR_RUNTIME_FAILURE;
            }
        }
    };

    for (size_t i = events_.size(); i > 0; --i) {
        void *&event_handle = events_[i - 1];
        if (event_handle == nullptr) {
            continue;
        }
        const int rc = ops_.destroy_event(ops_.context, event_handle);
        record_error(rc);
        if (rc == 0) {
            event_handle = nullptr;
        }
    }

    if (hidden_aicore_stream_ != nullptr) {
        const int rc = ops_.destroy_hidden_stream(ops_.context, hidden_aicore_stream_);
        record_error(rc);
        if (rc == 0) {
            hidden_aicore_stream_ = nullptr;
        }
    }
    return first_error;
}

bool L1ExecutionState::has_live_resources_locked() const {
    if (hidden_aicore_stream_ != nullptr) {
        return true;
    }
    for (void *event_handle : events_) {
        if (event_handle != nullptr) {
            return true;
        }
    }
    return false;
}

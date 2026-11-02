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

#include "scheduler_memory.h"

// Core-local view of a single engine's pool. The host owns initialization;
// metadata, task identities and epoch are immutable until all workers drain.
// Reusable Graph instances cannot use this view without a new reuse protocol.
struct SchedulerOrdinaryPoolView {
    __gm__ SchedulerOrdinaryPoolWord *words{nullptr};
    __gm__ SchedulerTaskControl *controls{nullptr};
    __gm__ const SchedulerTaskMetadata *metadata{nullptr};
    uint64_t task_count{0};
    uint64_t run_epoch{0};
    uint32_t core_type{UINT32_MAX};
};

enum class SchedulerOrdinaryPoolPhase : uint8_t { EMPTY, RESERVED, CONFIRMED, COMPLETED };

// This token stays with its owner, including while a cooperative task parks
// its dispatch slot. Rebuilding a payload never obtains a replacement ticket.
struct SchedulerOrdinaryPoolReservation {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t block_idx{0};
    uint16_t block_num{0};
    SchedulerOrdinaryPoolPhase phase{SchedulerOrdinaryPoolPhase::EMPTY};
};
static_assert(sizeof(SchedulerOrdinaryPoolReservation) == 24);

enum class SchedulerOrdinaryPoolResult : uint8_t { EMPTY, FOUND, RESERVED, CONFIRMED, COMPLETED, TASK_LAST, ERROR };

// Unlike the admission lock's temporary value 1, this ownership marker is
// retained until the run drains. It forbids a second publication of this task.
inline constexpr uint64_t SCHEDULER_ORDINARY_POOL_OWNER = 2;

inline __aicore__ bool scheduler_ordinary_pool_view_valid(const SchedulerOrdinaryPoolView &pool) {
    return pool.words != nullptr && pool.controls != nullptr && pool.metadata != nullptr && pool.task_count != 0 &&
           pool.task_count <= INT32_MAX && pool.run_epoch != 0 && pool.core_type < 2;
}

inline __aicore__ bool scheduler_ordinary_pool_task_valid(const SchedulerOrdinaryPoolView &pool, int64_t task) {
    if (!scheduler_ordinary_pool_view_valid(pool) || task < 0 || static_cast<uint64_t>(task) >= pool.task_count)
        return false;
    const auto *metadata = &pool.metadata[task];
    const LaneMask lanes = scheduler_task_lanes(*metadata);
    return scheduler_ordinary_pool_shape(metadata->flags, lanes, metadata->logical_block_num) &&
           (lanes.has_aic() ? 0U : 1U) == pool.core_type;
}

inline __aicore__ bool scheduler_ordinary_pool_token_valid(
    const SchedulerOrdinaryPoolView &pool, const SchedulerOrdinaryPoolReservation &token,
    SchedulerOrdinaryPoolPhase phase
) {
    return token.phase == phase && token.run_epoch == pool.run_epoch &&
           scheduler_ordinary_pool_task_valid(pool, token.task_ref) && token.block_idx < token.block_num &&
           token.block_num == pool.metadata[token.task_ref].logical_block_num;
}

// The caller owns a dependency-ready task obtained from Ready or direct resolve.
// This helper does not itself prove dependency readiness or grant kernel entry.
inline __aicore__ bool scheduler_ordinary_pool_publish(const SchedulerOrdinaryPoolView &pool, int64_t task) {
    if (!scheduler_ordinary_pool_task_valid(pool, task)) return false;
    auto *control = &pool.controls[task];
    if (scheduler_gm_load(control->state) != static_cast<int64_t>(SchedulerTaskState::BLOCKED) ||
        scheduler_gm_fetch_add(control->admission_state, UINT64_C(0)) != 0 ||
        scheduler_gm_fetch_add(control->next_block, UINT64_C(0)) != 0 ||
        scheduler_gm_fetch_add(control->completed_blocks, UINT64_C(0)) != 0 ||
        scheduler_gm_fetch_add(control->claim_tickets, UINT64_C(0)) != 0 ||
        scheduler_gm_compare_exchange(control->admission_claim_lock, UINT64_C(0), SCHEDULER_ORDINARY_POOL_OWNER) != 0)
        return false;
    scheduler_cache_barrier();
    const uint64_t bit = scheduler_ordinary_pool_task_bit(task);
    // Unique publication makes this addition an OR, without a CAS retry loop.
    // Other publishers/retirers alter different bits; no carry crosses a bit.
    const uint64_t previous = scheduler_gm_fetch_add(pool.words[scheduler_ordinary_pool_word_index(task)].bits, bit);
    scheduler_cache_barrier();
    return (previous & bit) == 0;
}

// A fill owns this observation; it never survives the resident-loop boundary.
// Only an atomic read can populate it. A positive bit proves publication in
// this non-reusable run, not that a block remains available for reservation.
class SchedulerOrdinaryPoolDiscovery {
public:
    inline __aicore__ uint64_t observe(__gm__ SchedulerOrdinaryPoolWord *word, uint64_t epoch) {
        if (word_ != word || run_epoch_ != epoch) {
            bits_ = scheduler_gm_fetch_add(word->bits, UINT64_C(0));
            word_ = word;
            run_epoch_ = epoch;
        }
        return bits_;
    }

    inline __aicore__ bool published(__gm__ SchedulerOrdinaryPoolWord *word, uint64_t epoch, uint64_t bit) const {
        return word_ == word && run_epoch_ == epoch && (bits_ & bit) != 0;
    }

private:
    __gm__ SchedulerOrdinaryPoolWord *word_{nullptr};
    uint64_t run_epoch_{0};
    uint64_t bits_{0};
};

// The private cursor advances even on an empty suffix. A publication absent
// from this fill's hint is observed by a later finite fill, not by a cached permit.
inline __aicore__ SchedulerOrdinaryPoolResult scheduler_ordinary_pool_discover(
    const SchedulerOrdinaryPoolView &pool, uint64_t *cursor, int64_t *task,
    SchedulerOrdinaryPoolDiscovery *observation = nullptr
) {
    if (task != nullptr) *task = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_ordinary_pool_view_valid(pool) || cursor == nullptr || task == nullptr || *cursor >= pool.task_count)
        return SchedulerOrdinaryPoolResult::ERROR;
    constexpr uint64_t capacity = SCHEDULER_ORDINARY_POOL_WORD_TASK_CAPACITY;
    const uint64_t word = scheduler_ordinary_pool_word_index(*cursor);
    const uint64_t begin = word * capacity;
    const uint64_t remaining = pool.task_count - begin;
    const uint64_t valid = (UINT64_C(1) << (remaining < capacity ? remaining : capacity)) - 1;
    auto *shared = &pool.words[word];
    const uint64_t observed = observation != nullptr ? observation->observe(shared, pool.run_epoch) :
                                                       scheduler_gm_fetch_add(shared->bits, UINT64_C(0));
    if ((observed & ~valid) != 0) return SchedulerOrdinaryPoolResult::ERROR;
    const uint64_t candidates = observed & (UINT64_MAX << (*cursor % capacity));
    if (candidates == 0) {
        *cursor = remaining <= capacity ? 0 : begin + capacity;
        return SchedulerOrdinaryPoolResult::EMPTY;
    }
    const uint64_t found = begin + static_cast<uint64_t>(__builtin_ctzll(candidates));
    if (!scheduler_ordinary_pool_task_valid(pool, static_cast<int64_t>(found)))
        return SchedulerOrdinaryPoolResult::ERROR;
    *task = static_cast<int64_t>(found);
    *cursor = found + 1 == pool.task_count ? 0 : found + 1;
    return SchedulerOrdinaryPoolResult::FOUND;
}

inline __aicore__ SchedulerOrdinaryPoolResult scheduler_ordinary_pool_reserve(
    const SchedulerOrdinaryPoolView &pool, int64_t task, SchedulerOrdinaryPoolReservation *token,
    const SchedulerOrdinaryPoolDiscovery *publication = nullptr
) {
    if (token == nullptr || token->phase != SchedulerOrdinaryPoolPhase::EMPTY ||
        !scheduler_ordinary_pool_task_valid(pool, task))
        return SchedulerOrdinaryPoolResult::ERROR;
    const uint64_t bit = scheduler_ordinary_pool_task_bit(task);
    auto *word = &pool.words[scheduler_ordinary_pool_word_index(task)];
    if (publication != nullptr && !publication->published(word, pool.run_epoch, bit))
        return SchedulerOrdinaryPoolResult::ERROR;
    if (publication == nullptr && (scheduler_gm_fetch_add(word->bits, UINT64_C(0)) & bit) == 0)
        return SchedulerOrdinaryPoolResult::EMPTY;
    auto *control = &pool.controls[task];
    if (scheduler_gm_fetch_add(control->admission_claim_lock, UINT64_C(0)) != SCHEDULER_ORDINARY_POOL_OWNER)
        return SchedulerOrdinaryPoolResult::ERROR;
    const uint32_t blocks = pool.metadata[task].logical_block_num;
    // A retired bit implies all valid tickets were already reserved and
    // confirmed. Fresh cursor checks reject a stale positive observation.
    if (scheduler_gm_fetch_add(control->claim_tickets, UINT64_C(0)) >= blocks)
        return SchedulerOrdinaryPoolResult::EMPTY;
    const uint64_t ticket = scheduler_gm_fetch_add(control->claim_tickets, UINT64_C(1));
    if (ticket >= blocks) return SchedulerOrdinaryPoolResult::EMPTY;
    *token = {
        pool.run_epoch, task, static_cast<uint32_t>(ticket), static_cast<uint16_t>(blocks),
        SchedulerOrdinaryPoolPhase::RESERVED
    };
    return SchedulerOrdinaryPoolResult::RESERVED;
}

struct SchedulerOrdinaryPoolNoopObserver {
    inline __aicore__ void confirmed(uint64_t) const {}
    inline __aicore__ void retiring() const {}
    inline __aicore__ void retired(uint64_t) const {}
};

template <typename Observer = SchedulerOrdinaryPoolNoopObserver>
inline __aicore__ SchedulerOrdinaryPoolResult scheduler_ordinary_pool_confirm(
    const SchedulerOrdinaryPoolView &pool, SchedulerOrdinaryPoolReservation *token, uint64_t *previous_count,
    const Observer &observer = {}
) {
    if (token == nullptr || previous_count == nullptr ||
        !scheduler_ordinary_pool_token_valid(pool, *token, SchedulerOrdinaryPoolPhase::RESERVED))
        return SchedulerOrdinaryPoolResult::ERROR;
    auto *control = &pool.controls[token->task_ref];
    *previous_count = scheduler_gm_fetch_add(control->next_block, UINT64_C(1));
    if (*previous_count >= token->block_num) return SchedulerOrdinaryPoolResult::ERROR;
    observer.confirmed(*previous_count + 1);
    if (*previous_count + 1 == token->block_num) {
        const uint64_t bit = scheduler_ordinary_pool_task_bit(token->task_ref);
        // Only the last valid confirmer retires the bit. Subtraction is an AND
        // with its complement because this bit is still set exactly once.
        observer.retiring();
        const uint64_t previous = scheduler_gm_fetch_add(
            pool.words[scheduler_ordinary_pool_word_index(token->task_ref)].bits, UINT64_C(0) - bit
        );
        observer.retired(previous);
        if ((previous & bit) == 0) return SchedulerOrdinaryPoolResult::ERROR;
    }
    scheduler_cache_barrier();
    token->phase = SchedulerOrdinaryPoolPhase::CONFIRMED;
    return SchedulerOrdinaryPoolResult::CONFIRMED;
}

inline __aicore__ SchedulerOrdinaryPoolResult scheduler_ordinary_pool_finish(
    const SchedulerOrdinaryPoolView &pool, SchedulerOrdinaryPoolReservation *token, uint64_t *previous_count
) {
    if (token == nullptr || previous_count == nullptr ||
        !scheduler_ordinary_pool_token_valid(pool, *token, SchedulerOrdinaryPoolPhase::CONFIRMED))
        return SchedulerOrdinaryPoolResult::ERROR;
    scheduler_cache_barrier();
    *previous_count = scheduler_gm_fetch_add(pool.controls[token->task_ref].completed_blocks, UINT64_C(1));
    if (*previous_count >= token->block_num) return SchedulerOrdinaryPoolResult::ERROR;
    token->phase = SchedulerOrdinaryPoolPhase::COMPLETED;
    return *previous_count + 1 == token->block_num ? SchedulerOrdinaryPoolResult::TASK_LAST :
                                                     SchedulerOrdinaryPoolResult::COMPLETED;
}

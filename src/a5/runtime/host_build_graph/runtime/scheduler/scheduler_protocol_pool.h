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

#include "scheduler_ready.h"

inline __aicore__ void
scheduler_protocol_pool_publication(__gm__ void *base, const SchedulerLocalState *local, int64_t task, bool before) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (layout->ordinary_pool_task_count == 0 || layout->pool_publications_offset == 0 ||
        !scheduler_ordinary_pool_task_valid(scheduler_ordinary_pool_view(base, local), task)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *actor = scheduler_state_at<SchedulerProtocolTaskActor>(base, layout->pool_publications_offset) + task;
    if (before) {
        scheduler_protocol_observe_record(actor);
        if (!scheduler_protocol_stamp(base, local, layout, &actor->before)) return;
        actor->run_epoch = layout->run_epoch;
        actor->task_ref = task;
        actor->order_key = UINT64_MAX;
        actor->worker_index = static_cast<uint32_t>(local->worker_id());
        actor->flags = scheduler_task_metadata_at(base, local, task)->flags;
    } else {
        if (actor->before.sequence == 0 || actor->task_ref != task || actor->run_epoch != layout->run_epoch ||
            actor->worker_index != local->worker_id() || actor->order_key != UINT64_MAX) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return;
        }
        if (!scheduler_protocol_stamp(base, local, layout, &actor->after)) return;
    }
    scheduler_protocol_publish_record(actor);
}

inline __aicore__ __gm__ SchedulerProtocolBlockTrace *scheduler_protocol_pool_block_at(
    __gm__ void *base, const SchedulerLocalState *local, __gm__ const SchedulerProtocolLayout *layout,
    const SchedulerOrdinaryPoolReservation &token
) {
    if (layout->ordinary_pool_task_count == 0 || token.run_epoch != layout->run_epoch ||
        !scheduler_ordinary_pool_task_valid(scheduler_ordinary_pool_view(base, local), token.task_ref) ||
        token.block_idx >= token.block_num) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    const auto *metadata = scheduler_task_metadata_at(base, local, token.task_ref);
    const uint64_t index = static_cast<uint64_t>(metadata->trace_index_base) + token.block_idx;
    if (metadata->logical_block_num != token.block_num || index >= layout->trace_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    return scheduler_state_at<SchedulerProtocolBlockTrace>(base, layout->blocks_offset) + index;
}

inline __aicore__ bool scheduler_protocol_pool_owner_matches(
    const SchedulerLocalState *local, __gm__ const SchedulerProtocolBlockOwner *owner,
    const SchedulerOrdinaryPoolReservation &token
) {
    return owner->run_epoch == token.run_epoch && owner->task_ref == token.task_ref && owner->order_key == UINT64_MAX &&
           owner->block_idx == token.block_idx && owner->logical_block_num == token.block_num &&
           owner->worker_index == local->worker_id();
}

inline __aicore__ void scheduler_protocol_pool_claim(
    __gm__ void *base, const SchedulerLocalState *local, const SchedulerOrdinaryPoolReservation &token, uint64_t before,
    uint64_t after
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_pool_block_at(base, local, layout, token);
    if (record == nullptr) return;
    auto *owner = &record->owner;
    const auto *metadata = scheduler_task_metadata_at(base, local, token.task_ref);
    scheduler_protocol_observe_record(owner);
    if (!scheduler_protocol_stamp_at(base, local, layout, &owner->claim_before, before)) return;
    owner->run_epoch = token.run_epoch;
    owner->task_ref = token.task_ref;
    owner->order_key = UINT64_MAX;
    owner->worker_index = static_cast<uint32_t>(local->worker_id());
    owner->block_idx = token.block_idx;
    owner->cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    owner->logical_block_num = token.block_num;
    scheduler_lane_mask_store(owner->active_mask, scheduler_task_lanes(*metadata));
    owner->flags = metadata->flags;
    owner->completion_ticket = UINT64_MAX;
    scheduler_protocol_stamp_at(base, local, layout, &owner->claim_after, after);
    scheduler_protocol_publish_record(owner);
}

template <bool RecordProtocol>
struct SchedulerPoolConfirmationObserver {
    __gm__ void *base;
    const SchedulerLocalState *local;
    const SchedulerOrdinaryPoolReservation *token;

    inline __aicore__ void confirmed(uint64_t count) const {
        if constexpr (RecordProtocol) {
            auto *layout = scheduler_protocol_layout_at(base, local);
            if (layout == nullptr) return;
            auto *record = scheduler_protocol_pool_block_at(base, local, layout, *token);
            if (record == nullptr) return;
            if (!scheduler_protocol_pool_owner_matches(local, &record->owner, *token) || count == 0 ||
                count > token->block_num) {
                scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
                return;
            }
            auto *claim = &record->claim;
            scheduler_protocol_observe_record(claim);
            claim->confirmed_count = count;
            claim->worker_index = static_cast<uint32_t>(local->worker_id());
            claim->block_idx = token->block_idx;
            claim->run_epoch = token->run_epoch;
            scheduler_protocol_publish_record(claim);
        }
    }

    inline __aicore__ void retiring() const {
        if constexpr (RecordProtocol) {
            auto *layout = scheduler_protocol_layout_at(base, local);
            if (layout == nullptr) return;
            auto *record = scheduler_protocol_pool_block_at(base, local, layout, *token);
            if (record == nullptr) return;
            if (!scheduler_protocol_pool_owner_matches(local, &record->owner, *token) ||
                record->claim.confirmed_count != token->block_num) {
                scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
                return;
            }
            scheduler_protocol_stamp(base, local, layout, &record->claim.published_before);
            scheduler_protocol_publish_record(&record->claim);
        }
    }

    inline __aicore__ void retired(uint64_t previous_word) const {
        if constexpr (RecordProtocol) {
            auto *layout = scheduler_protocol_layout_at(base, local);
            if (layout == nullptr) return;
            auto *record = scheduler_protocol_pool_block_at(base, local, layout, *token);
            if (record == nullptr) return;
            if (record->claim.published_before.sequence == 0 ||
                (previous_word & scheduler_ordinary_pool_task_bit(token->task_ref)) == 0) {
                scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
                return;
            }
            record->claim.seal_value = previous_word;
            scheduler_protocol_stamp(base, local, layout, &record->claim.published_after);
            scheduler_protocol_publish_record(&record->claim);
        }
    }
};

inline __aicore__ void scheduler_protocol_pool_complete(
    __gm__ void *base, const SchedulerLocalState *local, const SchedulerOrdinaryPoolReservation &token, bool before,
    uint64_t ticket
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_pool_block_at(base, local, layout, token);
    if (record == nullptr) return;
    auto *owner = &record->owner;
    if (!scheduler_protocol_pool_owner_matches(local, owner, token) ||
        (!before && owner->completion_before.sequence == 0)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, before ? &owner->completion_before : &owner->completion_after))
        return;
    if (!before) owner->completion_ticket = ticket;
    scheduler_protocol_publish_record(owner);
}

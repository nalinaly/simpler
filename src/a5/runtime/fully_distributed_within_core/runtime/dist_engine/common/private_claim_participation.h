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

#include "data_type.h"  // PTO_DEVICE_FUNC
#include "dist_engine/common/target.h"

#ifndef PTO_FDWIC_PRIVATE_CLAIM_PARTICIPATION_INTERVAL
#define PTO_FDWIC_PRIVATE_CLAIM_PARTICIPATION_INTERVAL 1
#endif

constexpr uint32_t kFdwicPrivateClaimParticipationInterval =
    PTO_FDWIC_PRIVATE_CLAIM_PARTICIPATION_INTERVAL;

PTO_DEVICE_FUNC constexpr bool fdwic_private_claim_participation_interval_supported(uint32_t interval) {
    return interval == 1U || interval == 2U || interval == 4U || interval == 8U;
}

// Private ownership is a monotonic FetchMax over cursor[task_id % S], not a
// per-task CAS node. Every task sharing one cursor must therefore keep the
// same participant residue: otherwise a later residue class can advance the
// cursor past a task before that task's designated candidates arrive.
PTO_DEVICE_FUNC constexpr uint32_t fdwic_private_claim_participation_residue(
    uint32_t task_id, uint32_t cursor_shards, uint32_t interval
) {
    return cursor_shards == 0U || !fdwic_private_claim_participation_interval_supported(interval) ?
               interval :
               (task_id % cursor_shards) % interval;
}

PTO_DEVICE_FUNC constexpr bool fdwic_private_claim_participates(
    uint32_t task_id, uint32_t role_local_candidate_rank, uint32_t cursor_shards, uint32_t interval
) {
    return cursor_shards != 0U && fdwic_private_claim_participation_interval_supported(interval) &&
           role_local_candidate_rank % interval ==
               fdwic_private_claim_participation_residue(task_id, cursor_shards, interval);
}

static_assert(
    fdwic_private_claim_participation_interval_supported(kFdwicPrivateClaimParticipationInterval),
    "private Claim participation interval must be one of 1, 2, 4, or 8"
);

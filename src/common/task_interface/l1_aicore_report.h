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

#include <cstddef>
#include <cstdint>
#include <type_traits>

/**
 * L1-only AICore -> AICPU startup report.
 *
 * A custom-AICPU task can run outside the AICore HBM-write snoop domain. It
 * must therefore invalidate a cached startup report while polling. The legacy
 * Handshake cache line is unsuitable because it mixes AICPU-owned
 * {aicpu_ready, task} with AICore-owned report fields: `dc civac` could write
 * an AICPU-dirty stale line back over a fresh AICore report.
 *
 * Startup fields have exactly one writer (AICore) and one reader (AICPU).
 * Teardown control occupies a second, atomic-only cache line: AICPU publishes
 * it with a release atomic store and AICore observes it through the A2/A3
 * bypass-load path.  It must not share a line with ordinary scalar fields,
 * otherwise a later DCCI writeback could overwrite the control update.
 * L1 allocates one two-line report per launched block during prepare, clears
 * the array on the caller stream for every invocation/replay, and retains the
 * address until explicit context close. L2/L3 pass nullptr and preserve their
 * historical in-Runtime Handshake protocol.
 */
struct alignas(64) L1AicoreTeardownControl {
    uint32_t post_close_release;
    uint8_t reserved[60];
};

struct alignas(64) L1AicoreReport {
    volatile uint32_t aicore_done;
    volatile uint32_t physical_core_id;
    volatile uint32_t core_type;
    uint32_t reserved0;
    uint8_t reserved[48];
    L1AicoreTeardownControl teardown;
};

static_assert(sizeof(L1AicoreTeardownControl) == 64, "L1 teardown control must occupy one cache line");
static_assert(alignof(L1AicoreTeardownControl) == 64, "L1 teardown control must be cache-line aligned");
static_assert(sizeof(L1AicoreReport) == 128, "one L1 AICore report must occupy exactly two cache lines");
static_assert(alignof(L1AicoreReport) == 64, "L1 AICore reports must be cache-line aligned");
static_assert(std::is_standard_layout_v<L1AicoreReport>, "L1 AICore report must be standard-layout");
static_assert(std::is_trivially_copyable_v<L1AicoreReport>, "L1 AICore report must be byte-copyable");
static_assert(offsetof(L1AicoreReport, aicore_done) == 0, "L1 AICore ready word must lead the cache line");
static_assert(offsetof(L1AicoreReport, teardown) == 64, "L1 teardown control must start a fresh cache line");

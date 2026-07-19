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

#include <stddef.h>
#include <stdint.h>

#include "pto_types.h"

struct FdwicQkSplitSubmitTicket {
    uint64_t submit_trace_start;
    uint32_t task_id;
    int16_t kernel_id;
    uint8_t won;
    uint8_t reserved;
};

static_assert(sizeof(FdwicQkSplitSubmitTicket) == 16, "QK split ticket must remain a 16-byte POD");
static_assert(offsetof(FdwicQkSplitSubmitTicket, task_id) == 8, "QK split ticket task offset mismatch");
static_assert(offsetof(FdwicQkSplitSubmitTicket, kernel_id) == 12, "QK split ticket kernel offset mismatch");
static_assert(offsetof(FdwicQkSplitSubmitTicket, won) == 14, "QK split ticket winner offset mismatch");

#if defined(__CCE_AICORE__)
extern "C" {
#if defined(__DAV_CUBE__)
PTO_DEVICE_FUNC TaskOutputTensors fdwic_qk_split_finish_aic(
    const FdwicQkSplitSubmitTicket *ticket, const L0TaskArgs *args
);
#elif defined(__DAV_VEC__)
PTO_DEVICE_FUNC TaskOutputTensors fdwic_qk_split_finish_aiv(
    const FdwicQkSplitSubmitTicket *ticket, const L0TaskArgs *args
);
#else
#error "QK split finish requires __DAV_CUBE__ or __DAV_VEC__"
#endif
}
#endif

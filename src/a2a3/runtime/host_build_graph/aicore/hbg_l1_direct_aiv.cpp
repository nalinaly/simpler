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

#include <cstddef>
#include <cstdint>

#include "hbg_l1_direct_launch.h"

#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)

#include "intrinsic.h"
#include "common/memory_barrier.h"
#include "pto2_dispatch_payload.h"
#include "tensor.h"

namespace {

using UnifiedKernelFunc = void (*)(__gm__ int64_t *);

struct alignas(64) HbgL1DirectAivLaneScratch {
    ChipTensor tensors[MAX_TENSOR_ARGS];
    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
    LocalContext local_context;
    GlobalContext global_context;
    uint8_t reserved
        [simpler::hbg::HBG_L1_DIRECT_AIV_LANE_SCRATCH_BYTES - sizeof(tensors) - sizeof(args) - sizeof(LocalContext) -
         sizeof(GlobalContext)];
};

static_assert(sizeof(HbgL1DirectAivLaneScratch) == simpler::hbg::HBG_L1_DIRECT_AIV_LANE_SCRATCH_BYTES);
static_assert(offsetof(HbgL1DirectAivLaneScratch, tensors) == 0);
static_assert(offsetof(HbgL1DirectAivLaneScratch, args) % 64 == 0);

__aicore__ __attribute__((always_inline)) bool
valid_direct_header(__gm__ const simpler::hbg::HbgL1DirectAivPackageHeader *header, uint32_t physical_lane_count) {
    if (header == nullptr || header->magic != simpler::hbg::HBG_L1_DIRECT_AIV_MAGIC ||
        header->abi_version != simpler::hbg::HBG_L1_DIRECT_AIV_ABI_VERSION ||
        header->header_size != simpler::hbg::HBG_L1_DIRECT_AIV_HEADER_BYTES || header->task_count == 0 ||
        header->logical_block_num == 0 || header->lane_count == 0 || physical_lane_count == 0 ||
        header->tensor_count > MAX_TENSOR_ARGS || header->scalar_count > MAX_SCALAR_ARGS ||
        header->task_records_offset != simpler::hbg::HBG_L1_DIRECT_AIV_HEADER_BYTES ||
        header->lane_scratch_stride != simpler::hbg::HBG_L1_DIRECT_AIV_LANE_SCRATCH_BYTES ||
        header->lane_scratch_offset != 0 || header->immutable_size != header->total_size ||
        header->function_bin_addr == 0 || header->lane_scratch_device_addr == 0 ||
        (header->lane_scratch_device_addr & 63u) != 0 || header->reserved != 0) {
        return false;
    }
    const uint64_t record_bytes = static_cast<uint64_t>(header->tensor_count) * sizeof(ChipTensor) +
                                  static_cast<uint64_t>(header->scalar_count) * sizeof(uint64_t);
    const uint64_t expected_record_stride = (record_bytes + 63u) & ~UINT64_C(63);
    const uint64_t records_end = static_cast<uint64_t>(header->task_records_offset) +
                                 static_cast<uint64_t>(header->task_count) * header->task_record_stride;
    const uint64_t expected_total = (records_end + 63u) & ~UINT64_C(63);
    return static_cast<uint64_t>(header->task_count) * header->logical_block_num == header->work_count &&
           physical_lane_count == header->lane_count && expected_record_stride == header->task_record_stride &&
           expected_total == header->total_size;
}

__aicore__ __attribute__((always_inline)) void populate_direct_args(
    __gm__ HbgL1DirectAivLaneScratch *scratch, __gm__ const simpler::hbg::HbgL1DirectAivPackageHeader *header,
    uint32_t task_id, uint32_t logical_block, uint32_t sub_block_id
) {
    __gm__ uint8_t *package_bytes =
        reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ simpler::hbg::HbgL1DirectAivPackageHeader *>(header));
    __gm__ uint8_t *record =
        package_bytes + header->task_records_offset + static_cast<uint64_t>(task_id) * header->task_record_stride;
    for (uint32_t index = 0; index < header->tensor_count; ++index) {
        __gm__ const uint64_t *source = reinterpret_cast<__gm__ const uint64_t *>(
            record + static_cast<uint64_t>(index) * simpler::hbg::HBG_L1_DIRECT_AIV_TENSOR_BYTES
        );
        __gm__ uint64_t *destination = reinterpret_cast<__gm__ uint64_t *>(&scratch->tensors[index]);
        for (uint32_t word = 0; word < sizeof(ChipTensor) / sizeof(uint64_t); ++word) {
            destination[word] = source[word];
        }
        scratch->args[index] = reinterpret_cast<uint64_t>(&scratch->tensors[index]);
    }
    __gm__ uint64_t *scalars = reinterpret_cast<__gm__ uint64_t *>(
        record + static_cast<uint64_t>(header->tensor_count) * simpler::hbg::HBG_L1_DIRECT_AIV_TENSOR_BYTES
    );
    for (uint32_t index = 0; index < header->scalar_count; ++index) {
        scratch->args[header->tensor_count + index] = scalars[index];
    }

    scratch->local_context.block_idx = static_cast<int32_t>(logical_block);
    scratch->local_context.block_num = static_cast<int32_t>(header->logical_block_num);
    scratch->local_context.async_ctx.completion_count = nullptr;
    scratch->local_context.async_ctx.completion_error_code = nullptr;
    scratch->local_context.async_ctx.completion_entries = nullptr;
    scratch->local_context.async_ctx.completion_capacity = 0;
    scratch->local_context.async_ctx.task_token.raw = UINT64_MAX;
    scratch->global_context.sub_block_id = static_cast<int32_t>(sub_block_id);
    scratch->args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&scratch->local_context);
    scratch->args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&scratch->global_context);
    OUT_OF_ORDER_STORE_BARRIER();
}

}  // namespace

/*
 * Dedicated A2/A3 vector entry for the homogeneous direct schedule.  Suffix
 * `_1_mix_aiv` gives it a distinct function/tiling entry from the normal
 * `aicore_kernel_0_{mix_aic,mix_aiv}` pair.  It deliberately has one native
 * argument: the CANN-owned immutable package copied by WithHostArgs.
 */
extern "C" __global__ __aicore__ void hbg_l1_direct_aiv_kernel_1_mix_aiv(__gm__ void *package) {
    auto *header = reinterpret_cast<__gm__ const simpler::hbg::HbgL1DirectAivPackageHeader *>(package);
    const uint32_t lane = static_cast<uint32_t>(get_block_idx());
    const uint32_t lane_count = static_cast<uint32_t>(get_block_num());
    if (!valid_direct_header(header, lane_count) || lane >= lane_count) return;

    auto *scratch = reinterpret_cast<__gm__ HbgL1DirectAivLaneScratch *>(
        header->lane_scratch_device_addr + static_cast<uint64_t>(lane) * header->lane_scratch_stride
    );
    UnifiedKernelFunc child = reinterpret_cast<UnifiedKernelFunc>(header->function_bin_addr);
    for (uint32_t work_id = lane; work_id < header->work_count; work_id += lane_count) {
        const uint32_t task_id = work_id / header->logical_block_num;
        const uint32_t logical_block = work_id - task_id * header->logical_block_num;
        populate_direct_args(scratch, header, task_id, logical_block, static_cast<uint32_t>(get_subblockid()));
        child(reinterpret_cast<__gm__ int64_t *>(scratch->args));
        OUT_OF_ORDER_STORE_BARRIER();
        // A lane may execute several child tasks (for example 50 works on 48
        // AIV lanes). Complete every outstanding pipeline operation before its
        // private argument/context scratch and the child's fixed UB are reused.
        pipe_barrier(PIPE_ALL);
    }
}

#endif

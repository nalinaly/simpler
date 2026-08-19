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
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "callable.h"
#include "hbg_l1_direct_launch.h"
#include "pto_shared_memory.h"

namespace simpler::hbg {

enum class HbgL1DirectAivBuildStatus : uint8_t {
    Ok = 0,
    NotEligible,
    InvalidGraph,
    AllocationFailure,
};

inline bool hbg_l1_direct_task_eligible(
    const PTO2TaskDescriptor &task, const PTO2TaskPayload &payload, const PTO2TaskSlotState &slot,
    int32_t expected_local_id, int32_t expected_kernel_id, int32_t expected_tensor_count, int32_t expected_scalar_count,
    int16_t expected_logical_block_num
) noexcept {
    return task.task_id == PTO2TaskId::make(0, static_cast<uint32_t>(expected_local_id)) &&
           slot.task_kind == TaskKind::KERNEL && slot.active_mask.raw() == PTO2_SUBTASK_MASK_AIV0 &&
           slot.task_attrs.raw() == 0 && slot.logical_block_num == expected_logical_block_num &&
           slot.total_required_subtasks == expected_logical_block_num && payload.fanin_count == 0 &&
           payload.predicate.op == PredicateOp::NONE && payload.dump_metadata.dump_arg_mask == 0 &&
           payload.dump_metadata.dump_arg_flags == 0 && payload.tensor_count == expected_tensor_count &&
           payload.scalar_count == expected_scalar_count &&
           task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] == INVALID_KERNEL_ID &&
           task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] == expected_kernel_id &&
           task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] == INVALID_KERNEL_ID;
}

/**
 * Try to flatten one strict HBG subset into an A2/A3 direct-AIV package.
 *
 * Ordinary graphs outside the subset return NotEligible and retain the full
 * AICPU scheduler path. Only malformed ring state or host allocation failure
 * is an error. The output is replaced transactionally on success.
 */
inline HbgL1DirectAivBuildStatus try_build_hbg_l1_direct_aiv_package(
    PTO2SharedMemoryRingHeader &ring, int32_t total_tasks, const uint64_t *callable_function_table,
    size_t callable_function_count, uint32_t lane_count, uint64_t lane_scratch_device_addr, std::vector<uint8_t> *out
) noexcept {
    if (total_tasks <= 0) return HbgL1DirectAivBuildStatus::NotEligible;
    if (callable_function_table == nullptr || callable_function_count == 0 || lane_count == 0 ||
        lane_scratch_device_addr == 0 || (lane_scratch_device_addr & 63u) != 0 || out == nullptr) {
        return HbgL1DirectAivBuildStatus::InvalidGraph;
    }

    const PTO2TaskDescriptor &first_task = ring.get_task_by_task_id(0);
    const PTO2TaskPayload &first_payload = ring.get_payload_by_task_id(0);
    const PTO2TaskSlotState &first_slot = ring.get_slot_state_by_task_id(0);
    const int32_t kernel_id = first_task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)];
    const int32_t tensor_count = first_payload.tensor_count;
    const int32_t scalar_count = first_payload.scalar_count;
    const int16_t logical_block_num = first_slot.logical_block_num;
    if (kernel_id < 0 || static_cast<size_t>(kernel_id) >= callable_function_count ||
        callable_function_table[kernel_id] == 0 ||
        callable_function_table[kernel_id] >
            std::numeric_limits<uint64_t>::max() - CoreCallable::binary_data_offset() ||
        tensor_count < 0 || tensor_count > MAX_TENSOR_ARGS || scalar_count < 0 || scalar_count > MAX_SCALAR_ARGS ||
        logical_block_num <= 0) {
        return HbgL1DirectAivBuildStatus::NotEligible;
    }
    for (int32_t task_id = 0; task_id < total_tasks; ++task_id) {
        if (!hbg_l1_direct_task_eligible(
                ring.get_task_by_task_id(task_id), ring.get_payload_by_task_id(task_id),
                ring.get_slot_state_by_task_id(task_id), task_id, kernel_id, tensor_count, scalar_count,
                logical_block_num
            )) {
            return HbgL1DirectAivBuildStatus::NotEligible;
        }
    }

    uint32_t record_stride = 0;
    if (!hbg_l1_direct_record_stride(
            static_cast<uint32_t>(tensor_count), static_cast<uint32_t>(scalar_count), &record_stride
        )) {
        return HbgL1DirectAivBuildStatus::InvalidGraph;
    }
    const uint64_t work_count = static_cast<uint64_t>(total_tasks) * static_cast<uint64_t>(logical_block_num);
    const uint64_t records_end =
        static_cast<uint64_t>(HBG_L1_DIRECT_AIV_HEADER_BYTES) + static_cast<uint64_t>(total_tasks) * record_stride;
    uint64_t package_size = 0;
    if (work_count > std::numeric_limits<uint32_t>::max() || !hbg_l1_direct_align_up(records_end, 64, &package_size)) {
        return HbgL1DirectAivBuildStatus::InvalidGraph;
    }
    if (package_size > std::numeric_limits<uint32_t>::max()) {
        return HbgL1DirectAivBuildStatus::InvalidGraph;
    }

    std::vector<uint8_t> candidate;
    try {
        candidate.assign(static_cast<size_t>(package_size), 0);
    } catch (...) {
        return HbgL1DirectAivBuildStatus::AllocationFailure;
    }

    HbgL1DirectAivPackageHeader header{};
    header.total_size = static_cast<uint32_t>(package_size);
    header.task_count = static_cast<uint32_t>(total_tasks);
    header.logical_block_num = static_cast<uint32_t>(logical_block_num);
    header.work_count = static_cast<uint32_t>(work_count);
    header.tensor_count = static_cast<uint32_t>(tensor_count);
    header.scalar_count = static_cast<uint32_t>(scalar_count);
    header.task_record_stride = record_stride;
    header.lane_count = lane_count;
    header.lane_scratch_offset = 0;
    header.immutable_size = header.total_size;
    // The callable-local binding table stores the device address of the
    // CoreCallable object. The generic scheduler dereferences that object and
    // reads resolved_addr_; upload_chip_callable_buffer() fixes resolved_addr_
    // to exactly object_base + binary_data_offset(). The direct path cannot
    // dereference device memory on Host, so derive the same immutable code
    // address from the wire-layout invariant.
    header.function_bin_addr = callable_function_table[kernel_id] + CoreCallable::binary_data_offset();
    header.lane_scratch_device_addr = lane_scratch_device_addr;
    std::memcpy(candidate.data(), &header, sizeof(header));

    const size_t tensor_bytes = static_cast<size_t>(tensor_count) * sizeof(ChipTensor);
    const size_t scalar_bytes = static_cast<size_t>(scalar_count) * sizeof(uint64_t);
    for (int32_t task_id = 0; task_id < total_tasks; ++task_id) {
        const PTO2TaskPayload &payload = ring.get_payload_by_task_id(task_id);
        uint8_t *record =
            candidate.data() + HBG_L1_DIRECT_AIV_HEADER_BYTES + static_cast<size_t>(task_id) * record_stride;
        std::memcpy(record, payload.tensors, tensor_bytes);
        std::memcpy(record + tensor_bytes, payload.scalars, scalar_bytes);
    }
    if (!validate_hbg_l1_direct_package(candidate.data(), candidate.size())) {
        return HbgL1DirectAivBuildStatus::InvalidGraph;
    }
    *out = std::move(candidate);
    return HbgL1DirectAivBuildStatus::Ok;
}

}  // namespace simpler::hbg

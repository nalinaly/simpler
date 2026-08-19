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

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "hbg_l1_direct_aiv_package.h"

namespace {

using simpler::hbg::HBG_L1_DIRECT_AIV_HEADER_BYTES;
using simpler::hbg::hbg_l1_direct_package_structures_equal;
using simpler::hbg::HbgL1DirectAivBuildStatus;
using simpler::hbg::HbgL1DirectAivPackageHeader;
using simpler::hbg::try_build_hbg_l1_direct_aiv_package;
using simpler::hbg::validate_hbg_l1_direct_package;

constexpr int32_t kTaskCount = 2;
constexpr int16_t kLogicalBlocksPerTask = 25;
constexpr uint32_t kA3AivLanes = 48;
constexpr int32_t kKernelId = 2;
constexpr uint64_t kCallableDeviceBase = UINT64_C(0x300000);
constexpr uint64_t kLaneScratchDeviceBase = UINT64_C(0x400000);

class HbgL1DirectAivPackageTest : public ::testing::Test {
protected:
    PTO2SharedMemoryRingHeader ring{};
    std::array<PTO2TaskDescriptor, kTaskCount> tasks{};
    std::array<PTO2TaskPayload, kTaskCount> payloads{};
    std::array<PTO2TaskSlotState, kTaskCount> slots{};
    std::array<uint64_t, 3> function_table{};

    void SetUp() override {
        ring.task_window_size = kTaskCount;
        ring.task_window_mask = kTaskCount - 1;
        ring.task_descriptors = tasks.data();
        ring.task_payloads = payloads.data();
        ring.slot_states = slots.data();
        function_table[kKernelId] = kCallableDeviceBase;

        constexpr uint32_t shape[2] = {1, 128};
        for (int32_t task_id = 0; task_id < kTaskCount; ++task_id) {
            auto &task = tasks[static_cast<size_t>(task_id)];
            auto &payload = payloads[static_cast<size_t>(task_id)];
            auto &slot = slots[static_cast<size_t>(task_id)];

            task.task_id = PTO2TaskId::make(0, static_cast<uint32_t>(task_id));
            task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
            task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = kKernelId;
            task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;

            payload.tensor_count = 1;
            payload.scalar_count = 1;
            payload.fanin_count = 0;
            payload.predicate.op = PredicateOp::NONE;
            payload.tensors[0] = make_tensor_external(
                reinterpret_cast<void *>(
                    static_cast<uintptr_t>(0x100000) + static_cast<uintptr_t>(task_id) * static_cast<uintptr_t>(0x10000)
                ),
                shape, 2, DataType::FLOAT32
            );
            payload.scalars[0] = UINT64_C(0x11110000) + static_cast<uint64_t>(task_id);

            slot.bind_buffers(&payload, &task);
            slot.task_kind = TaskKind::KERNEL;
            slot.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIV0);
            slot.task_attrs = TaskAttrs{};
            slot.logical_block_num = kLogicalBlocksPerTask;
            slot.total_required_subtasks = kLogicalBlocksPerTask;
        }
    }

    HbgL1DirectAivBuildStatus build(std::vector<uint8_t> *out) {
        return try_build_hbg_l1_direct_aiv_package(
            ring, kTaskCount, function_table.data(), function_table.size(), kA3AivLanes, kLaneScratchDeviceBase, out
        );
    }
};

TEST_F(HbgL1DirectAivPackageTest, FlattensFiftyWorksForFortyEightA3AivLanes) {
    std::vector<uint8_t> package;
    ASSERT_EQ(build(&package), HbgL1DirectAivBuildStatus::Ok);
    ASSERT_TRUE(validate_hbg_l1_direct_package(package.data(), package.size()));

    HbgL1DirectAivPackageHeader header{};
    std::memcpy(&header, package.data(), sizeof(header));
    EXPECT_EQ(header.task_count, 2u);
    EXPECT_EQ(header.logical_block_num, 25u);
    EXPECT_EQ(header.work_count, 50u);
    EXPECT_EQ(header.lane_count, 48u);
    EXPECT_EQ(header.tensor_count, 1u);
    EXPECT_EQ(header.scalar_count, 1u);
    EXPECT_EQ(header.function_bin_addr, kCallableDeviceBase + CoreCallable::binary_data_offset());
    EXPECT_EQ(header.lane_scratch_device_addr, kLaneScratchDeviceBase);

    for (int32_t task_id = 0; task_id < kTaskCount; ++task_id) {
        const uint8_t *record =
            package.data() + header.task_records_offset + static_cast<size_t>(task_id) * header.task_record_stride;
        EXPECT_EQ(std::memcmp(record, &payloads[static_cast<size_t>(task_id)].tensors[0], sizeof(ChipTensor)), 0);
        uint64_t scalar = 0;
        std::memcpy(&scalar, record + sizeof(ChipTensor), sizeof(scalar));
        EXPECT_EQ(scalar, payloads[static_cast<size_t>(task_id)].scalars[0]);
    }
}

TEST_F(HbgL1DirectAivPackageTest, RejectsDependenciesDifferentKernelsPredicatesAndTaskAttrs) {
    const std::vector<uint8_t> sentinel{0xa5};
    std::vector<uint8_t> package = sentinel;

    payloads[1].fanin_count = 1;
    EXPECT_EQ(build(&package), HbgL1DirectAivBuildStatus::NotEligible);
    EXPECT_EQ(package, sentinel);
    payloads[1].fanin_count = 0;

    function_table[1] = UINT64_C(0x500000);
    tasks[1].kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = 1;
    EXPECT_EQ(build(&package), HbgL1DirectAivBuildStatus::NotEligible);
    EXPECT_EQ(package, sentinel);
    tasks[1].kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = kKernelId;

    payloads[1].predicate.op = PredicateOp::EQ;
    EXPECT_EQ(build(&package), HbgL1DirectAivBuildStatus::NotEligible);
    EXPECT_EQ(package, sentinel);
    payloads[1].predicate.op = PredicateOp::NONE;

    slots[1].task_attrs.set_sync_start();
    EXPECT_EQ(build(&package), HbgL1DirectAivBuildStatus::NotEligible);
    EXPECT_EQ(package, sentinel);
}

TEST_F(HbgL1DirectAivPackageTest, RejectsInvalidPlatformResourcesWithoutReplacingOutput) {
    const std::vector<uint8_t> sentinel{0x3c};
    std::vector<uint8_t> package = sentinel;
    EXPECT_EQ(
        try_build_hbg_l1_direct_aiv_package(
            ring, kTaskCount, function_table.data(), function_table.size(), 0, kLaneScratchDeviceBase, &package
        ),
        HbgL1DirectAivBuildStatus::InvalidGraph
    );
    EXPECT_EQ(package, sentinel);
    EXPECT_EQ(
        try_build_hbg_l1_direct_aiv_package(
            ring, kTaskCount, function_table.data(), function_table.size(), kA3AivLanes, kLaneScratchDeviceBase + 1,
            &package
        ),
        HbgL1DirectAivBuildStatus::InvalidGraph
    );
    EXPECT_EQ(package, sentinel);
}

TEST_F(HbgL1DirectAivPackageTest, StructuralComparisonIgnoresTaskArgumentRecords) {
    std::vector<uint8_t> first;
    ASSERT_EQ(build(&first), HbgL1DirectAivBuildStatus::Ok);
    std::vector<uint8_t> rebound = first;
    rebound[HBG_L1_DIRECT_AIV_HEADER_BYTES] ^= 0x5a;
    EXPECT_TRUE(hbg_l1_direct_package_structures_equal(first.data(), first.size(), rebound.data(), rebound.size()));

    HbgL1DirectAivPackageHeader header{};
    std::memcpy(&header, rebound.data(), sizeof(header));
    ++header.logical_block_num;
    std::memcpy(rebound.data(), &header, sizeof(header));
    EXPECT_FALSE(hbg_l1_direct_package_structures_equal(first.data(), first.size(), rebound.data(), rebound.size()));
}

}  // namespace

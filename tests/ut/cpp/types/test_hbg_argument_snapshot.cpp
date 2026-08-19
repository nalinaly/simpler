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

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "hbg_argument_snapshot.h"

namespace {

using simpler::hbg::hbg_argument_snapshot_hash;
using simpler::hbg::hbg_argument_structure_hash;
using simpler::hbg::hbg_argument_structures_equal;

ChipTensor make_tensor(uint64_t address) {
    ChipTensor tensor{};
    tensor.buffer = PTOBufferHandle{address, 3 * 5 * sizeof(float)};
    tensor.owner_task_id = PTO2TaskId::make(1, 8);
    tensor.start_offset = 0;
    tensor.version = 4;
    tensor.ndims = 2;
    tensor.dtype = DataType::FLOAT32;
    tensor.manual_dep = true;
    tensor.is_contiguous = true;
    tensor.address_space = AddressSpace::DEVICE;
    tensor.shapes[0] = 3;
    tensor.shapes[1] = 5;
    tensor.extent_elem_cache = 15;
    tensor.strides[0] = 5;
    tensor.strides[1] = 1;
    return tensor;
}

ChipStorageTaskArgs make_args() {
    ChipStorageTaskArgs args{};
    args.add_tensor(make_tensor(0x100000));
    args.add_tensor(make_tensor(0x200000));
    args.add_scalar(0x1122334455667788ULL);
    return args;
}

TEST(HbgArgumentSnapshot, SameSemanticTensorAndScalarStateHasOneStableHash) {
    ChipStorageTaskArgs first = make_args();
    ChipStorageTaskArgs second = first;
    std::memset(first.tensor(0)._pad_cl2, 0x11, sizeof(first.tensor(0)._pad_cl2));
    std::memset(second.tensor(0)._pad_cl2, 0xee, sizeof(second.tensor(0)._pad_cl2));
    first.tensor(0).shapes[4] = 17;
    second.tensor(0).shapes[4] = 91;
    first.tensor(0).strides[4] = 23;
    second.tensor(0).strides[4] = 97;

    const uint64_t first_hash = hbg_argument_snapshot_hash(first);
    EXPECT_NE(first_hash, 0u);
    EXPECT_EQ(first_hash, hbg_argument_snapshot_hash(second));
    EXPECT_EQ(first_hash, hbg_argument_snapshot_hash(first));
}

TEST(HbgArgumentSnapshot, EveryUsedSemanticDomainChangesTheHash) {
    const ChipStorageTaskArgs baseline = make_args();
    const uint64_t expected = hbg_argument_snapshot_hash(baseline);
    ASSERT_NE(expected, 0u);

    ChipStorageTaskArgs changed = baseline;
    ++changed.tensor(0).buffer.addr;
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    ++changed.tensor(0).buffer.size;
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    ++changed.tensor(0).owner_task_id.raw;
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    ++changed.tensor(0).start_offset;
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    ++changed.tensor(0).version;
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    changed.tensor(0).dtype = DataType::FLOAT16;
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    changed.tensor(0).manual_dep = !changed.tensor(0).manual_dep;
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    ++changed.tensor(0).shapes[1];
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    ++changed.tensor(0).strides[0];
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    ++changed.tensor(0).extent_elem_cache;
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);

    changed = baseline;
    ++changed.scalar(0);
    EXPECT_NE(hbg_argument_snapshot_hash(changed), expected);
}

TEST(HbgArgumentSnapshot, CountsArePartOfTheSnapshotAndInvalidCountsFailClosed) {
    ChipStorageTaskArgs args = make_args();
    const uint64_t full_hash = hbg_argument_snapshot_hash(args);
    --args.scalar_count_;
    EXPECT_NE(hbg_argument_snapshot_hash(args), full_hash);

    args = make_args();
    args.tensor_count_ = -1;
    EXPECT_EQ(hbg_argument_snapshot_hash(args), 0u);

    args = make_args();
    args.scalar_count_ = CHIP_MAX_SCALAR_ARGS + 1;
    EXPECT_EQ(hbg_argument_snapshot_hash(args), 0u);
}

TEST(HbgArgumentSnapshot, StructureIgnoresAddressesAndScalarValuesButRetainsTensorMetadata) {
    const ChipStorageTaskArgs baseline = make_args();
    const uint64_t expected = hbg_argument_structure_hash(baseline);
    ASSERT_NE(expected, 0u);

    ChipStorageTaskArgs changed_values = baseline;
    changed_values.tensor(0).buffer.addr += 0x4000;
    changed_values.tensor(1).buffer.addr += 0x8000;
    changed_values.scalar(0) ^= UINT64_C(0xffff0000ffff0000);
    EXPECT_EQ(hbg_argument_structure_hash(changed_values), expected);
    EXPECT_TRUE(hbg_argument_structures_equal(baseline, changed_values));

    ChipStorageTaskArgs changed_metadata = changed_values;
    ++changed_metadata.tensor(0).strides[0];
    EXPECT_NE(hbg_argument_structure_hash(changed_metadata), expected);
    EXPECT_FALSE(hbg_argument_structures_equal(baseline, changed_metadata));

    changed_metadata = changed_values;
    --changed_metadata.scalar_count_;
    EXPECT_NE(hbg_argument_structure_hash(changed_metadata), expected);
    EXPECT_FALSE(hbg_argument_structures_equal(baseline, changed_metadata));
}

}  // namespace

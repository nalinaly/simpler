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

#include <gtest/gtest.h>

#include "orchestration_requirements.h"

namespace {

using simpler::orchestration::HbgL1RequirementsStatus;
using simpler::orchestration::REQUIREMENT_TENSOR_DATA_READ;
using simpler::orchestration::REQUIREMENT_TENSOR_DATA_WRITE;
using simpler::orchestration::validate_hbg_l1_requirements;

TEST(OrchestrationRequirements, MetadataIsMandatoryForBorrowedHbgL1) {
    EXPECT_EQ(validate_hbg_l1_requirements(false, 0), HbgL1RequirementsStatus::MetadataUnavailable);
}

TEST(OrchestrationRequirements, EmptyKnownMetadataIsAccepted) {
    EXPECT_EQ(validate_hbg_l1_requirements(true, 0), HbgL1RequirementsStatus::Ok);
}

TEST(OrchestrationRequirements, TensorDataReadAndWriteAreRejected) {
    EXPECT_EQ(
        validate_hbg_l1_requirements(true, REQUIREMENT_TENSOR_DATA_READ),
        HbgL1RequirementsStatus::HostTensorDataRequired
    );
    EXPECT_EQ(
        validate_hbg_l1_requirements(true, REQUIREMENT_TENSOR_DATA_WRITE),
        HbgL1RequirementsStatus::HostTensorDataRequired
    );
    EXPECT_EQ(
        validate_hbg_l1_requirements(true, REQUIREMENT_TENSOR_DATA_READ | REQUIREMENT_TENSOR_DATA_WRITE),
        HbgL1RequirementsStatus::HostTensorDataRequired
    );
}

TEST(OrchestrationRequirements, UnknownFutureBitsFailClosed) {
    EXPECT_EQ(validate_hbg_l1_requirements(true, UINT64_C(1) << 63), HbgL1RequirementsStatus::UnknownRequirement);
}

}  // namespace

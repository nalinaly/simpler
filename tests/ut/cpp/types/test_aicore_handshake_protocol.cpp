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

#include <array>
#include <cstddef>
#include <cstdint>

#include "aicore_handshake_protocol.h"
#include "hbg_l1_launch_control.h"
#include "l1_aicore_report.h"

TEST(AicoreHandshakeProtocol, KeepsSuccessAndErrorOnlyControlValuesDistinct) {
    EXPECT_EQ(AICORE_PRE_WINDOW_WAIT, 0U);
    EXPECT_EQ(AICORE_PRE_WINDOW_LEGACY_PROCEED, 1U);
    EXPECT_EQ(AICORE_PRE_WINDOW_CANCEL, 2U);
    EXPECT_EQ(AICORE_PRE_WINDOW_HOST_CANCEL, UINT32_MAX);
    EXPECT_NE(AICORE_PRE_WINDOW_CANCEL, AICORE_PRE_WINDOW_LEGACY_PROCEED);
    EXPECT_TRUE(aicore_pre_window_cancelled(AICORE_PRE_WINDOW_CANCEL));
    EXPECT_TRUE(aicore_pre_window_cancelled(AICORE_PRE_WINDOW_HOST_CANCEL));
    EXPECT_FALSE(aicore_pre_window_cancelled(AICORE_PRE_WINDOW_WAIT));
    EXPECT_FALSE(aicore_pre_window_cancelled(AICORE_PRE_WINDOW_LEGACY_PROCEED));
    EXPECT_EQ(AICORE_PRE_WINDOW_CANCEL_POLL_MASK + 1U, AICORE_PRE_WINDOW_CANCEL_POLL_INTERVAL);
}

TEST(AicoreHandshakeProtocol, UsesAnIndependentCacheLineForPreGenerationCancellation) {
    simpler::hbg::HbgL1LaunchControl control{};

    EXPECT_EQ(sizeof(control), 64u);
    EXPECT_EQ(control.prelaunch_state, simpler::hbg::HBG_L1_PRELAUNCH_CONTINUE);
    EXPECT_FALSE(simpler::hbg::hbg_l1_prelaunch_cancelled(&control));
    control.prelaunch_state = simpler::hbg::HBG_L1_PRELAUNCH_CANCEL;
    EXPECT_TRUE(simpler::hbg::hbg_l1_prelaunch_cancelled(&control));
}

TEST(AicoreHandshakeProtocol, AcceptsOnlyInRangeNonzeroRegisterMappings) {
    constexpr uint32_t max_physical_cores = 24;

    EXPECT_FALSE(aicore_register_mapping_invalid(0, max_physical_cores, 0x1000));
    EXPECT_FALSE(aicore_register_mapping_invalid(max_physical_cores - 1, max_physical_cores, 0x2000));
    EXPECT_TRUE(aicore_register_mapping_invalid(0, max_physical_cores, 0));
    EXPECT_TRUE(aicore_register_mapping_invalid(max_physical_cores - 1, max_physical_cores, 0));
    EXPECT_TRUE(aicore_register_mapping_invalid(max_physical_cores, max_physical_cores, 0x3000));
    EXPECT_TRUE(aicore_register_mapping_invalid(max_physical_cores + 1, max_physical_cores, 0));
}

TEST(AicoreHandshakeProtocol, RejectsOutOfRangeRegisterIndicesBeforeLookup) {
    constexpr uint32_t register_address_count = 108;

    EXPECT_TRUE(aicore_register_index_valid(0, register_address_count));
    EXPECT_TRUE(aicore_register_index_valid(register_address_count - 1, register_address_count));
    EXPECT_FALSE(aicore_register_index_valid(register_address_count, register_address_count));
    EXPECT_FALSE(aicore_register_index_valid(UINT32_MAX, register_address_count));
    EXPECT_FALSE(aicore_register_index_valid(0, 0));
}

TEST(L1AicoreReport, GivesEveryCoreTwoExclusiveCacheLines) {
    alignas(64) std::array<L1AicoreReport, 4> reports{};

    EXPECT_EQ(sizeof(L1AicoreReport), 128U);
    EXPECT_EQ(alignof(L1AicoreReport), 64U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(reports.data()) % 64U, 0U);
    for (size_t i = 1; i < reports.size(); ++i) {
        const auto current = reinterpret_cast<uintptr_t>(&reports[i]);
        const auto previous = reinterpret_cast<uintptr_t>(&reports[i - 1]);
        EXPECT_EQ(current - previous, sizeof(L1AicoreReport));
        EXPECT_NE(current / 64U, previous / 64U);
    }
}

TEST(L1AicoreReport, SeparatesStartupReportFromAtomicOnlyTeardownControl) {
    L1AicoreReport report{};
    report.aicore_done = 7;
    report.physical_core_id = 19;
    report.core_type = 1;

    EXPECT_EQ(report.aicore_done, 7U);
    EXPECT_EQ(report.physical_core_id, 19U);
    EXPECT_EQ(report.core_type, 1U);
    EXPECT_EQ(offsetof(L1AicoreReport, aicore_done), 0U);
    EXPECT_LT(offsetof(L1AicoreReport, core_type) + sizeof(report.core_type), 64U);
    EXPECT_EQ(offsetof(L1AicoreReport, teardown), 64U);
    EXPECT_EQ(sizeof(report.teardown), 64U);
    EXPECT_EQ(offsetof(L1AicoreTeardownControl, post_close_release), 0U);
}

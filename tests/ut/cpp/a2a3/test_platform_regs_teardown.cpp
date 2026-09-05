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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>

#include "aicpu/platform_regs.h"
#include "aicore_handshake_protocol.h"
#include "l1_aicore_report.h"

namespace {

struct alignas(64) SimRegisterBlock {
    std::array<std::byte, 0x500> bytes{};

    uint64_t address() { return reinterpret_cast<uint64_t>(bytes.data()); }
};

uint32_t load_reg(const SimRegisterBlock &registers, RegId id) {
    const auto *address = reinterpret_cast<const volatile uint32_t *>(
        reinterpret_cast<uintptr_t>(registers.bytes.data()) + reg_offset(id)
    );
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

TEST(AicoreTeardownProtocol, AckClosesFastPathBeforePublishingRelease) {
    SimRegisterBlock registers;
    L1AicoreTeardownControl teardown{};
    write_reg(registers.address(), RegId::COND, AICORE_EXITED_VALUE);
    write_reg(registers.address(), RegId::DATA_MAIN_BASE, 17);
    write_reg(registers.address(), RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);

    ASSERT_EQ(
        platform_finish_aicore_exit(
            registers.address(), std::numeric_limits<uint64_t>::max(), &teardown.post_close_release
        ),
        0
    );
    EXPECT_EQ(load_reg(registers, RegId::DATA_MAIN_BASE), AICPU_IDLE_TASK_ID);
    EXPECT_EQ(load_reg(registers, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_CLOSE);
    EXPECT_EQ(__atomic_load_n(&teardown.post_close_release, __ATOMIC_ACQUIRE), AICORE_POST_CLOSE_RELEASE);
}

TEST(AicoreTeardownProtocol, MissingAckDoesNotCloseOrPublishRelease) {
    SimRegisterBlock registers;
    L1AicoreTeardownControl teardown{};
    write_reg(registers.address(), RegId::COND, AICORE_IDLE_VALUE);
    write_reg(registers.address(), RegId::DATA_MAIN_BASE, 17);
    write_reg(registers.address(), RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);

    EXPECT_EQ(platform_finish_aicore_exit(registers.address(), 0, &teardown.post_close_release), -1);
    EXPECT_EQ(load_reg(registers, RegId::DATA_MAIN_BASE), 17U);
    EXPECT_EQ(load_reg(registers, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_OPEN);
    EXPECT_EQ(__atomic_load_n(&teardown.post_close_release, __ATOMIC_ACQUIRE), 0U);
}

TEST(AicoreTeardownProtocol, DeinitSignalsExitAndCompletesTheReturnHandoff) {
    SimRegisterBlock registers;
    L1AicoreTeardownControl teardown{};
    write_reg(registers.address(), RegId::COND, AICORE_IDLE_VALUE);
    write_reg(registers.address(), RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);

    std::thread aicore([&] {
        while (load_reg(registers, RegId::DATA_MAIN_BASE) != AICORE_EXIT_SIGNAL) {
            std::this_thread::yield();
        }
        write_reg(registers.address(), RegId::COND, AICORE_EXITED_VALUE);
    });

    const int32_t rc = platform_deinit_aicore_regs(registers.address(), &teardown.post_close_release);
    aicore.join();

    ASSERT_EQ(rc, 0);
    EXPECT_EQ(load_reg(registers, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_CLOSE);
    EXPECT_EQ(__atomic_load_n(&teardown.post_close_release, __ATOMIC_ACQUIRE), AICORE_POST_CLOSE_RELEASE);
}

}  // namespace

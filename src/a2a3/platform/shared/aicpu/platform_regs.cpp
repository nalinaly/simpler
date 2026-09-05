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
/**
 * @file platform_regs.cpp
 * @brief Platform-level register access implementation for AICPU
 *
 * Provides unified interface for:
 * 1. Platform register base address management (set/get_platform_regs)
 * 2. Register read/write operations (volatile MMIO, no barrier)
 * 3. Platform-agnostic AICore register initialization/deinitialization
 *
 * Ordering: read_reg / write_reg emit only the volatile MMIO load/store.
 * The MMIO region is Device-nGnRE (Early-write-ack, no Gathering, no
 * Reordering) — see docs/hardware/mmio-performance.md for the driver
 * source trace. nR orders accesses within the same region; cross
 * Device <-> Normal-cacheable ordering is the caller's responsibility
 * (wmb() before a publishing register write, rmb() after observing a
 * register hand-off bit).
 *
 * Platform Support:
 * - a2a3: MMIO volatile pointer access to real hardware registers
 * - a2a3sim: Volatile pointer access to host-allocated simulated registers
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "aicpu/device_time.h"
#include "common/platform_config.h"
#include "aicore_handshake_protocol.h"

static uint64_t g_platform_regs = 0;
static uint64_t g_platform_pmu_reg_addrs = 0;

void set_platform_regs(uint64_t regs) { g_platform_regs = regs; }

uint64_t get_platform_regs() { return g_platform_regs; }

void set_platform_pmu_reg_addrs(uint64_t pmu_regs) { g_platform_pmu_reg_addrs = pmu_regs; }

uint64_t get_platform_pmu_reg_addrs() { return g_platform_pmu_reg_addrs; }

volatile uint32_t *get_reg_ptr(uint64_t reg_base_addr, RegId reg) {
    return reinterpret_cast<volatile uint32_t *>(reg_base_addr + reg_offset(reg));
}

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    return static_cast<uint64_t>(reg_load_acquire(get_reg_ptr(reg_base_addr, reg)));
}

void platform_init_aicore_regs(uint64_t reg_addr) {
    // Both a2a3 and a2a3sim require fast path control to be enabled before use
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);

    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
}

void platform_signal_aicore_exit(uint64_t reg_addr) { write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL); }

void platform_close_aicore_fast_path(uint64_t reg_addr) {
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
}

uint64_t platform_aicore_exit_deadline() { return get_sys_cnt_aicpu() + inner_get_deinit_timeout_ticks(); }

int32_t platform_wait_aicore_exit_ack(uint64_t reg_addr, uint64_t deadline) {
    while (read_reg(reg_addr, RegId::COND) != AICORE_EXITED_VALUE) {
        if (get_sys_cnt_aicpu() > deadline) {
            return -1;
        }
    }
    return 0;
}

int32_t platform_finish_aicore_exit(uint64_t reg_addr, uint64_t deadline, volatile uint32_t *post_close_release) {
    // Wait for AICore to acknowledge exit, until the caller's deadline. On
    // timeout, skip register cleanup (AICore is unresponsive; host will
    // aclrtResetDevice to clear all hardware state).
    if (platform_wait_aicore_exit_ack(reg_addr, deadline) != 0) return -1;

    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
    // Close fast path control
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
    if (post_close_release != nullptr) {
        // Device-nGnRE writes are early-acknowledged.  Read back the same MMIO
        // window so CLOSE is complete before the normal-memory release becomes
        // visible to AICore.  This is the ordering required by the hardware
        // fast-path protocol: close 0x18 before the persistent wrapper exits.
        (void)read_reg(reg_addr, RegId::FAST_PATH_ENABLE);
        platform_publish_aicore_post_close_release(post_close_release);
    }
    return 0;
}

void platform_publish_aicore_post_close_release(volatile uint32_t *post_close_release) {
    if (post_close_release == nullptr) return;
    __atomic_store_n(post_close_release, AICORE_POST_CLOSE_RELEASE, __ATOMIC_RELEASE);
}

int32_t platform_deinit_aicore_regs(uint64_t reg_addr, volatile uint32_t *post_close_release) {
    platform_signal_aicore_exit(reg_addr);
    return platform_finish_aicore_exit(reg_addr, platform_aicore_exit_deadline(), post_close_release);
}

uint32_t platform_get_physical_cores_count() {
    return DAV_2201::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;
}

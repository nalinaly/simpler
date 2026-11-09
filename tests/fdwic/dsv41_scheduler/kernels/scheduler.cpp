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
#include <pto/pto-inst.hpp>
#include <pto/common/kernel_meta.hpp>

#define DSV_DEVICE __aicore__
#define DSV_GM __gm__
#include "../common/scheduler.h"

namespace {
struct DeviceOps {
    __aicore__ static uint64_t now() {
        asm volatile("" ::: "memory");
        return get_sys_cnt();
    }
    __aicore__ static uint64_t now_after(uint64_t value) {
        uint64_t clock = 0;
        asm volatile("MOV %0, %0\nMOV %1, SYS_CNT\n" : "+l"(value), "=&l"(clock)::"memory");
        return clock;
    }
    __aicore__ static uint64_t load(__gm__ uint64_t *p) { return atomicAdd(p, uint64_t(0)); }
    __aicore__ static uint64_t store(__gm__ uint64_t *p, uint64_t v) { return atomicExch(p, v); }
    __aicore__ static uint64_t add(__gm__ uint64_t *p, uint64_t v) { return atomicAdd(p, v); }
    __aicore__ static uint64_t cas(__gm__ uint64_t *p, uint64_t expected, uint64_t desired) {
        return atomicCAS(p, expected, desired);
    }
    __aicore__ static void save_sample(__gm__ dsv41::Sample *p, const dsv41::Sample &s) {
        auto *dst = reinterpret_cast<__gm__ uint64_t *>(p);
        auto *src = reinterpret_cast<const uint64_t *>(&s);
        for (uint32_t i = 0; i < sizeof(s) / 8; ++i)
            dst[i] = src[i];
        for (uint32_t i = 0; i < sizeof(s) / 8; i += 8)
            dcci(dst + i, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb(DSB_ALL);
    }
    __aicore__ static void save_worker(__gm__ dsv41::Worker *p, const dsv41::Worker &s) {
        auto *dst = reinterpret_cast<__gm__ uint64_t *>(p);
        auto *src = reinterpret_cast<const uint64_t *>(&s);
        for (uint32_t i = 0; i < sizeof(s) / 8; ++i)
            dst[i] = src[i];
        for (uint32_t i = 0; i < sizeof(s) / 8; i += 8)
            dcci(dst + i, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb(DSB_ALL);
    }
    __aicore__ static void flush_events(__gm__ dsv41::Event *events, uint64_t count) {
        auto *bytes = reinterpret_cast<__gm__ uint8_t *>(events);
        for (uint64_t i = 0; i < count * sizeof(dsv41::Event); i += 64)
            dcci(bytes + i, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb(DSB_ALL);
    }
    __aicore__ static void compute(__gm__ dsv41::Image *image, uint32_t worker) {
        using Global = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, 16, 1>>;
        Global a(image->tiles[worker][0]), b(image->tiles[worker][1]), c(image->tiles[worker][2]);
#if defined(DSV_AIC)
        using Mat =
            pto::Tile<pto::TileType::Mat, float, 16, 16, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
        using Left =
            pto::Tile<pto::TileType::Left, float, 16, 16, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
        using Right =
            pto::Tile<pto::TileType::Right, float, 16, 16, pto::BLayout::RowMajor, -1, -1, pto::SLayout::ColMajor>;
        using Acc = pto::TileAcc<float, 16, 16, 16, 16>;
        Mat ma(16, 16), mb(16, 16);
        Left left(16, 16);
        Right right(16, 16);
        Acc result;
        pto::TASSIGN(ma, 0x0);
        pto::TASSIGN(mb, 0x10000);
        pto::TASSIGN(left, 0);
        pto::TASSIGN(right, 0);
        pto::TASSIGN(result, 0);
        for (uint32_t i = 0; i < image->repeats; ++i) {
            pto::TLOAD(ma, a);
            pto::TLOAD(mb, b);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            pto::TMOV(left, ma);
            pto::TMOV(right, mb);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            pto::TMATMUL(result, left, right);
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            pto::TSTORE(c, result);
            set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
        }
#else
        using Tile = pto::Tile<pto::TileType::Vec, float, 16, 16, pto::BLayout::RowMajor, -1, -1>;
        Tile va(16, 16), vb(16, 16), vc(16, 16);
        pto::TASSIGN(va, 0);
        pto::TASSIGN(vb, 0x10000);
        pto::TASSIGN(vc, 0x20000);
        for (uint32_t i = 0; i < image->repeats; ++i) {
            pto::TLOAD(va, a);
            pto::TLOAD(vb, b);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            pto::TADD(vc, va, vb);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pto::TSTORE(c, vc);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        }
#endif
    }
};
}  // namespace

#if defined(DSV_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(dsv41_0_mix_aic, 1, 2);
extern "C" __global__ __aicore__ void dsv41_0_mix_aic(__gm__ dsv41::Image *image) {
    dsv41::dispatch<DeviceOps>(image, get_block_idx() * 3, get_coreid());
}
#else
PTO_SYNCALL_MIX_AIC_KERNEL_META(dsv41_0_mix_aiv, 1, 2);
extern "C" __global__ __aicore__ void dsv41_0_mix_aiv(__gm__ dsv41::Image *image) {
    dsv41::dispatch<DeviceOps>(image, get_block_idx() * 3 + 1 + get_subblockid(), get_coreid());
}
#endif

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
#include <cstdio>
#include <cstring>

#include "common/unified_log.h"
#include "common/kernel_args.h"
#include "hbg_aicpu_invocation.h"
#include "l1_aicpu_args.h"
#include "common/platform_config.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/aicpu_device_config.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_log.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#include "runtime.h"

// Run-wall capture: the host allocates a device buffer addressed by
// KernelArgs.device_wall_data_base holding one { start_cycle, end_cycle } pair
// per launched AICPU thread (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH pairs,
// raw sys-counter cycles), and resets it to { UINT64_MAX, 0 } before each run.
// Every surviving simpler_aicpu_exec thread writes its own start/end into its
// own slot (indexed by platform_aicpu_affinity_thread_idx()) — plain stores,
// no cross-thread atomics. The host reads the array and reduces once:
// wall = max(end) - min(start). No single-threaded pre-pass is needed to
// seed the start.

// Forward declaration of aicpu_execute (implemented in aicpu_executor.cpp).
// simpler_aicpu_register_callable is NOT declared/forwarded here: it is
// exported directly by the TMARB runtime (host_build_graph does not export it).
extern "C" int aicpu_execute(Runtime *arg);
extern "C" __attribute__((weak)) int
aicpu_execute_l1(Runtime *runtime, const void *unaligned_orch_args, int32_t invocation_callable_id);
extern "C" __attribute__((weak)) int
aicpu_execute_l1_hbg(Runtime *runtime, const simpler::hbg::HbgAicpuInvocationView *invocation);
extern "C" __attribute__((weak)) int simpler_aicpu_begin_l1_context(uint64_t context_generation);

static void PublishHbgPrelaunchCancel(const simpler::hbg::HbgAicpuInvocationView *invocation) {
    const auto *slot = invocation == nullptr ? nullptr : &invocation->slot;
    auto *control = simpler::hbg::hbg_l1_launch_control_or_fallback(
        slot, static_cast<uint64_t>(get_hbg_l1_prelaunch_control_addr())
    );
    if (control == nullptr) return;
    __atomic_store_n(&control->prelaunch_state, simpler::hbg::HBG_L1_PRELAUNCH_CANCEL, __ATOMIC_RELEASE);
    cache_flush_range(control, sizeof(*control));
}

/**
 * AICPU kernel main execution entry point.
 *
 * Called per-thread by the main aicpu_scheduler. Host registers this SO via
 * `rtsBinaryLoadFromFile` (JSON load, cpuKernelMode=0) and resolves this
 * symbol via `rtsFuncGetByName`; each per-task launch goes through
 * `rtsLaunchCpuKernel` on the cached `rtFuncHandle`. The bootstrap dispatcher
 * only writes this SO to the preinstall path — it does not dlsym this symbol
 * itself.
 *
 * @param arg Pointer to the front-less KernelArgs payload (runtime_args @ 0)
 * @return 0 on success, non-zero on error
 */
static int ExecuteAicpuKernel(
    const KernelArgs *k_args, const void *l1_orch_args_bytes, int32_t l1_callable_id,
    const simpler::hbg::HbgAicpuInvocationView *hbg_invocation
) {
    // Log severity was snapshot once by simpler_aicpu_init at worker init; the
    // resident SO keeps it across launches, so exec does not re-snapshot.
    Runtime *runtime = k_args->runtime_args;

    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid runtime_args: null pointer");
        PublishHbgPrelaunchCancel(hbg_invocation);
        return -1;
    }
    // Per-device invariants (log config, orch device id) were latched once by
    // simpler_aicpu_init at worker init. Profiling enable bits live on
    // KernelArgs::enable_profiling_flag now (no longer mirrored into
    // Handshake), so decode the umbrella bitmask once and hand it to the
    // existing platform-state setters.
    set_platform_regs(k_args->regs);
    set_platform_dump_base(k_args->dump_data_base);
    set_dump_args_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_DUMP_ARGS));
    set_platform_l2_swimlane_base(k_args->l2_swimlane_data_base);
    set_platform_l2_swimlane_aicore_rotation_table(k_args->l2_swimlane_aicore_rotation_table);
    set_l2_swimlane_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_L2_SWIMLANE));
    set_platform_pmu_base(k_args->pmu_data_base);
    set_pmu_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_PMU));
    set_platform_dep_gen_base(k_args->dep_gen_data_base);
    set_dep_gen_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_DEP_GEN));
    set_scope_stats_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_SCOPE_STATS));
    set_platform_scope_stats_base(k_args->scope_stats_data_base);

    // Filter-style affinity gate (a5). Host probed the topology, computed
    // ALLOWED_CPUS, and wrote it into runtime->aicpu_allowed_cpus[]. The
    // gate barriers exactly runtime->aicpu_launch_count threads (= the
    // count CANN was told to launch, which equals popcount(OCCUPY) — see
    // src/a5/platform/onboard/host/device_runner.cpp), keeps those whose
    // sched_getcpu() ∈ allowed_cpus, and exposes the deterministic
    // exec_idx via platform_aicpu_affinity_thread_idx() — the executor
    // reads it to assign sched/orch role.
    //
    // If the host probe didn't populate the gate inputs (allowed_count or
    // launch_count is 0) the gate would unconditionally drop every thread
    // and we'd silently return success without ever calling aicpu_execute.
    // That's a host-side bug; fail loud so it surfaces instead of
    // producing nothing at runtime.
    const int32_t allowed_cpu_count = runtime->get_aicpu_allowed_cpu_count();
    const int32_t aicpu_launch_count = runtime->get_aicpu_launch_count();
    if (!platform_aicpu_affinity_config_valid(allowed_cpu_count, aicpu_launch_count)) {
        LOG_ERROR(
            "Invalid AICPU affinity inputs: allowed_cpu_count=%d launch_count=%d max=%d", allowed_cpu_count,
            aicpu_launch_count, MAX_GATE_THREADS
        );
        PublishHbgPrelaunchCancel(hbg_invocation);
        return -1;
    }
    if (!platform_aicpu_affinity_gate_filter(
            runtime->get_aicpu_allowed_cpus(), allowed_cpu_count, aicpu_launch_count
        )) {
        return 0;
    }

    // Publish the phase-buffer base so the finer preamble/so_load/graph_build/
    // post_orch + orch/sched phases stamped inside aicpu_execute / the scheduler
    // resolve their per-thread slot via platform_aicpu_affinity_thread_idx()
    // (no C++ thread_local — see docs/dynamic-linking.md). Idempotent across the
    // concurrent exec threads (same base). Run-wall is stamped here.
    set_platform_phase_base(k_args->device_wall_data_base);
    AicpuPhaseScope run_wall(AicpuPhase::RunWall);

    int rc = 0;
    if (hbg_invocation != nullptr) {
        if (aicpu_execute_l1_hbg == nullptr) {
            LOG_ERROR("%s", "HBG L1 AICPU execution is unavailable in this runtime");
            PublishHbgPrelaunchCancel(hbg_invocation);
            return -1;
        }
        rc = aicpu_execute_l1_hbg(runtime, hbg_invocation);
    } else if (l1_orch_args_bytes != nullptr) {
        rc = aicpu_execute_l1(runtime, l1_orch_args_bytes, l1_callable_id);
    } else {
        rc = aicpu_execute(runtime);
    }
    if (rc != 0) {
        LOG_ERROR("AICPU executor failed with rc=%d", rc);
        // This is redundant after a fully initialized generation (its common
        // epilogue already closes every register window), but it also covers
        // failures that occur before the generation can establish its N-way
        // cleanup gate.
        PublishHbgPrelaunchCancel(hbg_invocation);
        return rc;
    }

    // Run-wall end is stamped by run_wall's destructor (covers the early return
    // above too); host reduces max(end) - min(start) → ns.
    return rc;
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *arg) {
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }
    return ExecuteAicpuKernel(reinterpret_cast<const KernelArgs *>(arg), nullptr, -1, nullptr);
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_exec(void *arg) {
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid L1 invocation arguments: null pointer");
        return -1;
    }
    // The runtime-owned task image is only a byte buffer.  Copy its small,
    // naturally-aligned prefix before typed access; the unique TRB
    // orchestrator worker snapshots the large orch payload into persistent
    // aligned storage.  This keeps the per-worker stack bounded and does not
    // assume the CANN argument pool preserves alignas(64).
    L1AicpuInvocationPrefix prefix{};
    if (!ReadL1AicpuInvocationPrefix(arg, &prefix) || !IsValidL1AicpuInvocationPrefix(prefix)) {
        LOG_ERROR(
            "Invalid L1 invocation ABI: version=%u size=%u reserved=%u", prefix.abi_version, prefix.struct_size,
            prefix.reserved
        );
        return -1;
    }
    if (aicpu_execute_l1 == nullptr) {
        LOG_ERROR("%s", "L1 AICPU execution is unavailable in this runtime");
        return -1;
    }
    return ExecuteAicpuKernel(&prefix.kernel_args, L1AicpuInvocationOrchArgsBytes(arg), prefix.callable_id, nullptr);
}

// Internal HBG trampoline. AICPU scheduler keeps inner runtime DSOs in one
// process-global namespace; exporting this helper lets a previously loaded TRB
// DSO preempt the HBG definition and route the HBG entry through TRB's null
// weak executor. CANN launches simpler_aicpu_l1_hbg_exec, not this bridge, so
// bind it to the DSO that contains it.
extern "C" __attribute__((visibility("hidden"))) int simpler_aicpu_execute_l1_hbg_platform(
    const KernelArgs *kernel_args, const simpler::hbg::HbgAicpuInvocationView *invocation
) {
    if (kernel_args == nullptr || invocation == nullptr) {
        LOG_ERROR("%s", "Invalid HBG L1 platform invocation");
        PublishHbgPrelaunchCancel(invocation);
        return -1;
    }
    return ExecuteAicpuKernel(kernel_args, nullptr, -1, invocation);
}

/**
 * AICPU per-device init entry point.
 *
 * Launched at worker init (before any register_callable / exec), this latches
 * the per-device invariants into the resident AICPU SO globals. It is launched
 * again only when first-use provisioning adds an async-DMA workspace. Because
 * the inner SO stays dlopen'd across launches, the latest values survive every
 * subsequent per-task launch.
 *
 * @param arg Pointer to an InitArgs payload
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_init(void *arg) {
    init_log_switch();
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid init kernel arguments: null pointer");
        return -1;
    }

    InitArgs *init_args = reinterpret_cast<InitArgs *>(arg);
    if (init_args->l1_context_generation != 0) {
        if (simpler_aicpu_begin_l1_context == nullptr) {
            if (init_args->hbg_l1_prelaunch_control_addr != 0) {
                LOG_ERROR("%s", "HBG L1 runtime is missing its context-generation hook");
                return -1;
            }
        } else if (simpler_aicpu_begin_l1_context(init_args->l1_context_generation) != 0) {
            LOG_ERROR(
                "Failed to begin borrowed-L1 context generation %llu",
                static_cast<unsigned long long>(init_args->l1_context_generation)
            );
            return -1;
        }
    }
    set_log_level(static_cast<int>(init_args->log_level));
    set_orch_device_id(static_cast<int>(init_args->device_id));
    set_scheduler_timeout_ms(static_cast<int>(init_args->scheduler_timeout_ms));
    for (int k = 0; k < DMA_WORKSPACE_KIND_COUNT; ++k) {
        set_dma_workspace_addr(k, init_args->dma_workspace_addr[k]);
    }
    set_hbg_l1_prelaunch_control_addr(init_args->hbg_l1_prelaunch_control_addr);

    return 0;
}

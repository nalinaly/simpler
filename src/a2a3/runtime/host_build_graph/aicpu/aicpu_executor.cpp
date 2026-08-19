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
#include <unistd.h>

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "aicpu/aicpu_device_config.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/device_time.h"
#include "aicpu/device_phase_aicpu.h"
#include "callable_protocol.h"
#include "common/kernel_args.h"
#include "hbg_aicpu_invocation.h"
#include "hbg_context_registry.h"
#include "hbg_execution_slot_registry.h"
#include "hbg_restore.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"
#include "spin_hint.h"

// Runtime headers (full struct definition for create/destroy + PTO2_SCOPE)
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// Performance profiling headers
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "common/unified_log.h"

// Register-based communication
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"
#include "utils/thread_completion_gate.h"

// Core type definitions
#include "common/core_type.h"

// CoreCallable for resolved dispatch address
#include "callable.h"

// Scheduler data structures (CoreExecState, CoreTracker, etc.)
#include "scheduler/scheduler_types.h"

// Scheduler context class
#include "scheduler/scheduler_context.h"

// From orchestration/common.cpp linked into this DSO — updates g_current_runtime
// here (cleared on teardown before runtime_destroy).
extern "C" void framework_bind_runtime(PTO2Runtime *rt);
extern "C" __attribute__((weak)) int simpler_aicpu_execute_l1_hbg_platform(
    const KernelArgs *kernel_args, const simpler::hbg::HbgAicpuInvocationView *invocation
);

static int32_t read_pto2_runtime_status(Runtime *runtime) {
    if (runtime == nullptr) {
        return 0;
    }

    void *sm = runtime->get_gm_sm_ptr();
    if (sm == nullptr) {
        return 0;
    }

    auto *header = static_cast<PTO2SharedMemoryHeader *>(sm);
    int32_t orch_error_code = header->orch_error_code.load(std::memory_order_acquire);
    int32_t sched_error_code = header->sched_error_code.load(std::memory_order_acquire);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

static PTO2Runtime *rt{nullptr};

struct AicpuExecutor {
    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};

    // Parallel-handshake coordination (see AicpuExecutor::init). hs_setup_done_
    // is published by the leader once the shared pre-handshake setup is visible;
    // hs_arrived_ is the barrier counting threads that finished their core slice.
    // hs_thread_seq_ hands out a distinct [0, nthreads) index when the platform
    // exposes no affinity idx (sim, where platform_aicpu_affinity_thread_idx()
    // is -1 during init) so the threads don't all collapse to leader 0.
    std::atomic<bool> hs_setup_done_{false};
    std::atomic<int32_t> hs_arrived_{0};
    std::atomic<int32_t> hs_thread_seq_{0};

    // Parallel-boot-classify coordination (see AicpuExecutor::run). classify_ready_
    // is published by the boot leader once its leader-only orchestration setup is
    // visible; classify_arrived_ is the barrier counting threads that finished
    // their slice of the initial classify. Both are one-shot per run and reset in
    // deinit().
    std::atomic<bool> classify_ready_{false};
    std::atomic<int32_t> classify_arrived_{0};

    int32_t aicpu_thread_num_{0};

    // ===== Task queue state (managed by scheduler ready queues) =====

    simpler::ThreadCompletionGate completion_gate_;
    std::atomic<int32_t> run_error_{0};
    std::atomic<bool> runtime_init_ready_{false};
    std::atomic<int32_t> hbg_restore_error_{0};
    std::atomic<uint32_t> hbg_fault_stage_{static_cast<uint32_t>(simpler::hbg::HbgL1FaultStage::None)};
    std::atomic<bool> hbg_fault_injected_{false};
    std::atomic<int32_t> hbg_unexpected_teardown_error_{0};

    // Per-Worker arena backing the PTO2Runtime + sm_handle + orch/sched/mailbox
    // sub-regions (created in runtime_create_from_sm, released in runtime_destroy).
    // Default-constructed: libc-backed backend, no ctx.
    DeviceArena runtime_arena_;

    // ===== Scheduler context (owns all dispatch/completion/drain state) =====
    SchedulerContext sched_ctx_;

    // ===== Methods =====
    int32_t init(Runtime *runtime, simpler::hbg::HbgL1FaultStage requested_fault);
    int32_t run(Runtime *runtime, const simpler::hbg::HbgAicpuInvocationView *hbg_invocation = nullptr);
    void arrive_and_finalize_run();
    void deinit(Runtime *runtime);

    void latch_run_error(int32_t error) {
        if (error == 0) return;
        int32_t expected = 0;
        (void)run_error_.compare_exchange_strong(expected, error, std::memory_order_acq_rel, std::memory_order_acquire);
    }
};

static AicpuExecutor g_aicpu_executor;

static simpler::hbg::HbgContextRegistry *current_hbg_context_registry() noexcept {
    const auto address = static_cast<uint64_t>(get_hbg_l1_context_registry_addr());
    if (address == 0) return nullptr;
    auto *registry = reinterpret_cast<simpler::hbg::HbgContextRegistry *>(address);
    cache_invalidate_range(registry, offsetof(simpler::hbg::HbgContextRegistry, execution_slot));
    return simpler::hbg::valid_hbg_context_registry_header(registry) ? registry : nullptr;
}

struct HbgRestoreFaultContext {
    simpler::hbg::HbgL1FaultStage stage{simpler::hbg::HbgL1FaultStage::None};
};

static int hbg_restore_copy(
    void *context, simpler::hbg::HbgLaunchRegionKind kind, void *destination, const void *source, size_t size
) noexcept {
    const auto *fault = static_cast<const HbgRestoreFaultContext *>(context);
    if (fault != nullptr && fault->stage == simpler::hbg::HbgL1FaultStage::RestoreCopy &&
        kind == simpler::hbg::HbgLaunchRegionKind::RuntimeArenaImage) {
        return simpler::hbg::hbg_l1_fault_error(fault->stage);
    }
    std::memcpy(destination, source, size);
    return 0;
}

static int hbg_restore_publish(
    void *context, simpler::hbg::HbgLaunchRegionKind kind, const void *destination, size_t size
) noexcept {
    const auto *fault = static_cast<const HbgRestoreFaultContext *>(context);
    if (fault != nullptr && fault->stage == simpler::hbg::HbgL1FaultStage::RestorePublish &&
        kind == simpler::hbg::HbgLaunchRegionKind::RuntimeArenaImage) {
        return simpler::hbg::hbg_l1_fault_error(fault->stage);
    }
    cache_flush_range(destination, size);
    return 0;
}

static bool publish_hbg_l1_prelaunch_cancel(simpler::hbg::HbgL1LaunchControl *control) noexcept {
    if (control == nullptr) {
        LOG_ERROR("HBG L1 prelaunch rejection has no valid cancellation control");
        return false;
    }
    __atomic_store_n(&control->prelaunch_state, simpler::hbg::HBG_L1_PRELAUNCH_CANCEL, __ATOMIC_RELEASE);
    cache_flush_range(control, sizeof(*control));
    return true;
}

static int reject_hbg_l1_via_control(simpler::hbg::HbgL1LaunchControl *control) noexcept {
    (void)publish_hbg_l1_prelaunch_cancel(control);
    return -1;
}

static int reject_hbg_l1_before_generation(const simpler::hbg::HbgExecutionSlotRegistration &slot) noexcept {
    return reject_hbg_l1_via_control(simpler::hbg::hbg_l1_launch_control(slot));
}

static bool publish_hbg_l1_without_slot_cancel() noexcept {
    // simpler_aicpu_init latched this address from the host-owned immutable
    // slot before any HBG registration or invocation task was enqueued. It is
    // therefore the only safe writer target when the registry being validated
    // is itself unavailable/corrupt; never recover a device pointer from the
    // unvalidated per-invocation HostArgs blob.
    auto *control = simpler::hbg::hbg_l1_launch_control_or_fallback(
        nullptr, static_cast<uint64_t>(get_hbg_l1_prelaunch_control_addr())
    );
    return publish_hbg_l1_prelaunch_cancel(control);
}

static int reject_hbg_l1_without_slot() noexcept {
    (void)publish_hbg_l1_without_slot_cancel();
    return -1;
}

// ===== AicpuExecutor Method Implementations =====

int32_t AicpuExecutor::init(Runtime *runtime, simpler::hbg::HbgL1FaultStage requested_fault) {
    if (runtime == nullptr) {
        LOG_ERROR("runtime is nullptr");
        return -1;
    }

    // All AICPU threads enter init. The per-core AICore handshake is the
    // dominant preamble cost (serial MMIO, ~217 µs of ~283 µs for 72 cores), so
    // it is parallelized: the leader (tidx 0) does the shared setup, every
    // thread handshakes a disjoint slice of cores, then the leader finishes init
    // after a barrier. Non-leaders spin on init_done_.
    int32_t nthreads = runtime->aicpu_thread_num;
    if (nthreads == 0) nthreads = 1;
    if (nthreads < 1 || nthreads > MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid aicpu_thread_num: %d", nthreads);
        return -1;
    }
    // Each thread needs a distinct index in [0, nthreads) to pick the leader and
    // partition the cores. Onboard the gate filter assigns it (exec_idx); sim's
    // gate does not, so platform_aicpu_affinity_thread_idx() is -1 here for every
    // thread — hand those a distinct index from a counter (mirrors run()'s
    // thread_idx_++ fallback) instead of collapsing them all to leader 0, which
    // would run pre_/post_handshake_init on every thread and race the shared
    // scheduler state. Exactly nthreads threads reach init (the gate drops the
    // rest), so the counter yields a gap-free [0, nthreads).
    int32_t tidx = platform_aicpu_affinity_thread_idx();
    if (tidx < 0) tidx = hs_thread_seq_.fetch_add(1, std::memory_order_acq_rel);
    platform_aicpu_affinity_set_thread_idx(tidx);
    // A thread whose index still falls outside [0, nthreads) owns no core slice:
    // handshake_partition would compute lo/hi past cores_total_num_ and index
    // all_handshakes[]/core_exec_states_ out of bounds. Reject it here (mirrors
    // the bounds guard already in run()). Fail only this thread and do NOT set
    // init_failed_ — that would make the valid peers abort before their
    // hs_arrived_ increment and hang the leader at the barrier below.
    if (tidx >= nthreads) {
        LOG_ERROR("AICPU affinity thread idx %d out of range [0,%d) in init", tidx, nthreads);
        return -1;
    }
    const bool is_leader = (tidx == 0);

    if (is_leader) {
        LOG_INFO("AicpuExecutor: Initializing");
        // The 0 → 1 fixup already applied above.
        aicpu_thread_num_ = nthreads;

        completion_gate_.reset();
        run_error_.store(0, std::memory_order_relaxed);
        hbg_fault_stage_.store(static_cast<uint32_t>(simpler::hbg::HbgL1FaultStage::None), std::memory_order_relaxed);
        hbg_fault_injected_.store(false, std::memory_order_relaxed);
        hbg_unexpected_teardown_error_.store(0, std::memory_order_relaxed);
        init_done_.store(false, std::memory_order_relaxed);
        init_failed_.store(false, std::memory_order_relaxed);
        hs_arrived_.store(0, std::memory_order_relaxed);
        if (sched_ctx_.pre_handshake_init(runtime, aicpu_thread_num_, get_platform_regs()) != 0) {
            init_failed_.store(true, std::memory_order_release);
        }
        hs_setup_done_.store(true, std::memory_order_release);
    } else {
        while (!hs_setup_done_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    // Every valid participant joins the same init barrier even when the leader's
    // shared setup failed. Skipping that arrival would let one AICPU task return
    // while peers remain blocked in this generation, which cannot be recovered
    // by resetting an L1 borrowed device context.
    if (!init_failed_.load(std::memory_order_acquire)) {
        sched_ctx_.handshake_partition(runtime, tidx, nthreads, requested_fault);
    }

    // Barrier: leader waits for every slice to finish, then completes init.
    hs_arrived_.fetch_add(1, std::memory_order_acq_rel);
    if (is_leader) {
        while (hs_arrived_.load(std::memory_order_acquire) < nthreads) {
            SPIN_WAIT_HINT();
        }
        if (!init_failed_.load(std::memory_order_acquire)) {
            const int32_t post_handshake_rc = sched_ctx_.post_handshake_init(runtime, requested_fault);
            if (post_handshake_rc != 0) {
                if ((requested_fault == simpler::hbg::HbgL1FaultStage::SchedulerAssign ||
                     requested_fault == simpler::hbg::HbgL1FaultStage::PhysicalCoreMapping ||
                     requested_fault == simpler::hbg::HbgL1FaultStage::PhysicalCoreId) &&
                    post_handshake_rc == simpler::hbg::hbg_l1_fault_error(requested_fault)) {
                    hbg_fault_stage_.store(static_cast<uint32_t>(requested_fault), std::memory_order_relaxed);
                    hbg_fault_injected_.store(true, std::memory_order_relaxed);
                    latch_run_error(post_handshake_rc);
                }
                init_failed_.store(true, std::memory_order_release);
            }
        }
        if (!init_failed_.load(std::memory_order_acquire) &&
            requested_fault == simpler::hbg::HbgL1FaultStage::SchedulerInit) {
            hbg_fault_stage_.store(static_cast<uint32_t>(requested_fault), std::memory_order_relaxed);
            hbg_fault_injected_.store(true, std::memory_order_relaxed);
            latch_run_error(simpler::hbg::hbg_l1_fault_error(requested_fault));
            init_failed_.store(true, std::memory_order_release);
        }
        init_done_.store(true, std::memory_order_release);
        if (!init_failed_.load(std::memory_order_acquire)) {
            LOG_INFO("AicpuExecutor: Init complete");
        }
    } else {
        while (!init_done_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }
    return init_failed_.load(std::memory_order_acquire) ? -1 : 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::run(Runtime *runtime, const simpler::hbg::HbgAicpuInvocationView *hbg_invocation) {
    int32_t affinity_exec_idx = platform_aicpu_affinity_thread_idx();
    int32_t thread_idx = (affinity_exec_idx >= 0) ? affinity_exec_idx : (thread_idx_++);
    if (thread_idx < 0 || thread_idx >= aicpu_thread_num_ || thread_idx >= MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Thread index %d out of bounds (active=%d max=%d exec_idx=%d)", thread_idx, aicpu_thread_num_,
            MAX_AICPU_THREADS, affinity_exec_idx
        );
        return -1;
    }
    auto requested_fault = simpler::hbg::HbgL1FaultStage::None;
    bool deferred_lifecycle_fault = false;
    int32_t run_rc = 0;
    if (const int32_t shared_error = run_error_.load(std::memory_order_acquire); shared_error != 0) {
        run_rc = shared_error;
        goto run_epilogue;
    }

    // Boot: the last AICPU thread (aicpu_thread_num_ - 1) performs the one-time
    // host-orch attach. host_build_graph's orchestrator already ran on the host,
    // which also relocated every cross-task pointer to its final device address
    // before H2D — so the SM/arena this thread sees are already fully
    // device-addressed. This thread attaches the prebuilt arena, points the SM
    // handle's ring-header pointers at the device SM WITHOUT resetting the
    // host-populated data, hands the host-computed task count to the scheduler,
    // and releases the other threads. It then falls through and schedules its own
    // cores like every other thread — host_build_graph has no device-side
    // orchestrator, so there is no orch/sched split.
    if (thread_idx == aicpu_thread_num_ - 1) {
        const bool is_hbg_l1 = hbg_invocation != nullptr;
        void *prebuilt_arena = is_hbg_l1 ? reinterpret_cast<void *>(hbg_invocation->slot.binding.runtime_arena_base) :
                                           runtime->get_prebuilt_arena_base();
        size_t off_runtime =
            is_hbg_l1 ? hbg_invocation->slot.binding.runtime_offset : runtime->get_prebuilt_runtime_offset();

        // A boot failure falls through to the common teardown at the end of
        // run() — it must NOT return early. This thread owns a core slice
        // (handshake_partition assigns [lo, total) to the last thread), so an
        // early return would skip shutdown(thread_idx) — leaving its AICore
        // cores spinning on an unclosed register window — and the completion
        // gate never opens, so the host hangs into the op-execute
        // timeout (507018) instead of seeing the failure. On failure: record it
        // in run_rc, leave rt null so the dispatch block below skips, and still
        // publish runtime_init_ready_ (single point at the block's end) so the
        // peer threads stop spinning.
        bool boot_ok = (prebuilt_arena != nullptr);
        if (!boot_ok) {
            LOG_ERROR("Thread %d: host-orch: prebuilt_arena_base is null", thread_idx);
            rt = nullptr;
            run_rc = -1;
        }

        if (boot_ok && is_hbg_l1) {
            cache_invalidate_range(hbg_invocation->blob, static_cast<size_t>(hbg_invocation->blob_size));
            requested_fault = simpler::hbg::hbg_aicpu_fault_stage(hbg_invocation);
            hbg_fault_stage_.store(static_cast<uint32_t>(requested_fault), std::memory_order_release);
            simpler::hbg::HbgRestoreCommit commit{};
            HbgRestoreFaultContext restore_context{requested_fault};
            const simpler::hbg::HbgRestoreOps restore_ops{&restore_context, hbg_restore_copy, hbg_restore_publish};
            const auto restore = simpler::hbg::restore_hbg_launch_blob(
                hbg_invocation->blob, static_cast<size_t>(hbg_invocation->blob_size), hbg_invocation->slot,
                hbg_invocation->header.identity, restore_ops, &commit
            );
            if (restore.status != simpler::hbg::HbgRestoreStatus::Ok ||
                commit.plan_generation != hbg_invocation->header.plan_generation ||
                commit.plan_hash != hbg_invocation->header.plan_hash) {
                LOG_ERROR(
                    "Thread %d: HBG task package restore failed status=%u region=%u callback=%d", thread_idx,
                    static_cast<unsigned>(restore.status), restore.region_index, restore.callback_error
                );
                const int32_t injected_error = simpler::hbg::hbg_l1_fault_error(requested_fault);
                if (injected_error != 0 && restore.callback_error == injected_error) {
                    hbg_fault_injected_.store(true, std::memory_order_release);
                    run_rc = injected_error;
                } else {
                    run_rc = -1;
                }
                rt = nullptr;
                boot_ok = false;
            }
        }

        if (boot_ok && requested_fault == simpler::hbg::HbgL1FaultStage::AfterSchedulerInit) {
            run_rc = simpler::hbg::hbg_l1_fault_error(requested_fault);
            hbg_fault_injected_.store(true, std::memory_order_release);
            boot_ok = false;
        }

        if (boot_ok) {
            runtime_arena_.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);
            rt = reinterpret_cast<PTO2Runtime *>(static_cast<char *>(prebuilt_arena) + off_runtime);
            if (!runtime_has_valid_prebuilt_invocation_state(rt)) {
                LOG_ERROR("Thread %d: host-orch: invalid task-owned invocation state", thread_idx);
                rt = nullptr;
                run_rc = -1;
                boot_ok = false;
            }
            if (boot_ok && is_hbg_l1 &&
                !simpler::hbg::hbg_prebuilt_invocation_matches(
                    &rt->prebuilt_invocation, hbg_invocation->header.identity.function_binding_hash,
                    hbg_invocation->header.identity.host_total_tasks
                )) {
                LOG_ERROR("Thread %d: restored HBG invocation identity does not match the graph image", thread_idx);
                rt = nullptr;
                run_rc = -1;
                boot_ok = false;
            }
        }

        if (boot_ok) {
            runtime_wire_arena_pointers(runtime_arena_, rt->prebuilt_layout, rt);

            void *sm_ptr = is_hbg_l1 ? reinterpret_cast<void *>(hbg_invocation->slot.binding.shared_memory_base) :
                                       runtime->get_gm_sm_ptr();
            uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size_per_ring(rt->prebuilt_layout.task_window_sizes);
            memset(rt->sm_handle, 0, sizeof(*rt->sm_handle));
            if (!rt->sm_handle->attach_populated(sm_ptr, sm_size, rt->prebuilt_layout.task_window_sizes)) {
                LOG_ERROR("Thread %d: host-orch: sm_handle->attach_populated failed", thread_idx);
                rt = nullptr;
                run_rc = -1;
                boot_ok = false;
            }
        }

        if (boot_ok) {
            memset(rt->aicore_mailbox, 0, sizeof(*rt->aicore_mailbox));
            runtime_finalize_after_wire(rt, sched_ctx_.aic_count(), sched_ctx_.aiv_count());
            runtime->set_slot_states_ptr(nullptr);

            sched_ctx_.bind_runtime(rt);
            // Latch the host-built task count (on_orchestration_done sets total_tasks_)
            // BEFORE the runtime_init_ready_ release below — that store is the barrier
            // that unblocks the scheduler threads. Otherwise they would acquire
            // runtime_init_ready_ with total_tasks_=0 and race to an early exit before
            // the host task count is visible (host-orch has no concurrent orchestrator
            // to keep them alive).
            // NOTE: do NOT call rt_orchestration_done(rt) here. The HOST already
            // called it in run_host_orchestration; the orchestrator's own
            // task-allocator pointers are intentionally NOT relocated (only the
            // SM cross-task pointers and the host-built fanout adjacency —
            // dep_pool / ready queues / fanout_head — were), so they still hold
            // host addresses and mark_done()'s active_count() read would
            // dereference host memory and fault the AICPU. on_orchestration_done
            // only needs total_tasks and the scalar
            // orchestrator.inline_completed_tasks, both already valid.
            const int32_t host_total_tasks = rt->prebuilt_invocation.host_total_tasks;
            sched_ctx_.on_orchestration_done(runtime, rt, thread_idx, host_total_tasks);
            LOG_INFO("Thread %d: host-orch boot complete (%d tasks)", thread_idx, host_total_tasks);
        }

        if (boot_ok && requested_fault == simpler::hbg::HbgL1FaultStage::BeforeClassify) {
            run_rc = simpler::hbg::hbg_l1_fault_error(requested_fault);
            hbg_fault_injected_.store(true, std::memory_order_release);
        }

        // Publish "leader setup done" (SM attached, task count latched, queues
        // allocated). Every thread then classifies its slice below before any of
        // them may dispatch — the leader holds runtime_init_ready_ until then.
        if (is_hbg_l1) hbg_restore_error_.store(run_rc, std::memory_order_release);
        classify_ready_.store(true, std::memory_order_release);
    }

    // Parallel initial classify. Every AICPU thread waits for the leader's
    // orchestration setup, seeds its disjoint slice of the whole graph's ready
    // set + wake lists, then barriers. Only once all slices are done does the
    // leader publish runtime_init_ready_, so no thread dispatches against a
    // half-seeded graph.
    while (!classify_ready_.load(std::memory_order_acquire)) {
        SPIN_WAIT_HINT();
    }
    if (hbg_invocation != nullptr && thread_idx != aicpu_thread_num_ - 1) {
        requested_fault = static_cast<simpler::hbg::HbgL1FaultStage>(hbg_fault_stage_.load(std::memory_order_acquire));
    }
    if (hbg_invocation != nullptr && thread_idx != aicpu_thread_num_ - 1) {
        cache_invalidate_range(
            reinterpret_cast<const void *>(hbg_invocation->slot.binding.shared_memory_base),
            static_cast<size_t>(hbg_invocation->slot.binding.shared_memory_capacity)
        );
        cache_invalidate_range(
            reinterpret_cast<const void *>(hbg_invocation->slot.binding.runtime_arena_base),
            static_cast<size_t>(hbg_invocation->slot.binding.runtime_arena_capacity)
        );
    }
    if (hbg_invocation != nullptr) {
        const int32_t restore_error = hbg_restore_error_.load(std::memory_order_acquire);
        if (restore_error != 0) run_rc = restore_error;
    }
    if (run_rc == 0 && !sched_ctx_.is_completed() && rt != nullptr) {
        sched_ctx_.classify_partition(thread_idx, aicpu_thread_num_);
    }
    classify_arrived_.fetch_add(1, std::memory_order_acq_rel);
    if (thread_idx == aicpu_thread_num_ - 1) {
        while (classify_arrived_.load(std::memory_order_acquire) < aicpu_thread_num_) {
            SPIN_WAIT_HINT();
        }
        runtime_init_ready_.store(true, std::memory_order_release);
    } else {
        while (!runtime_init_ready_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    if (run_rc == 0 && requested_fault == simpler::hbg::HbgL1FaultStage::BeforeDispatch) {
        run_rc = simpler::hbg::hbg_l1_fault_error(requested_fault);
        hbg_fault_injected_.store(true, std::memory_order_release);
    }

    // Every AICPU thread schedules its assigned cores.
    deferred_lifecycle_fault = requested_fault == simpler::hbg::HbgL1FaultStage::Shutdown ||
                               requested_fault == simpler::hbg::HbgL1FaultStage::RuntimeDestroy;
    if (run_rc == 0 && !deferred_lifecycle_fault && !sched_ctx_.is_completed()) {
        if (rt == nullptr) {
            LOG_ERROR("Thread %d: rt is null after orchestrator error, skipping dispatch", thread_idx);
        } else {
            sched_ctx_.bind_runtime(rt);
            // 3S+1P: the last thread is the core-less resolution (P) thread; the
            // rest are core-owning schedulers (S).
            int32_t completed = (thread_idx == sched_ctx_.p_thread_idx()) ?
                                    sched_ctx_.run_resolution_thread(runtime, thread_idx) :
                                    sched_ctx_.resolve_and_dispatch(runtime, thread_idx, requested_fault);
            if (completed < 0) {
                if (requested_fault == simpler::hbg::HbgL1FaultStage::SchedulerDispatch &&
                    completed == simpler::hbg::hbg_l1_fault_error(requested_fault)) {
                    hbg_fault_injected_.store(true, std::memory_order_release);
                }
                LOG_ERROR("Thread %d: Scheduler failed with rc=%d", thread_idx, completed);
                run_rc = completed;
            } else {
                LOG_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);
            }
        }
    }

run_epilogue:
    // Always shutdown AICore — even if sched_ctx_.completed_ was already true.
    // platform_deinit_aicore_regs is idempotent.
    int32_t shutdown_rc = sched_ctx_.shutdown(thread_idx);
    if (shutdown_rc != 0) {
        int32_t expected = 0;
        (void)hbg_unexpected_teardown_error_.compare_exchange_strong(
            expected, shutdown_rc, std::memory_order_acq_rel, std::memory_order_acquire
        );
    }
    if (shutdown_rc != 0 && run_rc == 0) {
        run_rc = shutdown_rc;
    }
    if (shutdown_rc == 0 && run_rc == 0 && requested_fault == simpler::hbg::HbgL1FaultStage::Shutdown) {
        run_rc = simpler::hbg::hbg_l1_fault_error(requested_fault);
        hbg_fault_injected_.store(true, std::memory_order_release);
    }
    latch_run_error(run_rc);

    LOG_INFO("Thread %d: Completed", thread_idx);

    arrive_and_finalize_run();

    const int32_t shared_error = run_error_.load(std::memory_order_acquire);
    return shared_error != 0 ? shared_error : run_rc;
}

void AicpuExecutor::arrive_and_finalize_run() {
    completion_gate_.arrive_and_finalize_if_last(aicpu_thread_num_, [&] {
        aicpu_publish_task_timing_tail_usage(aicpu_thread_num_);
        // Destroy the host_build_graph runtime. sm_handle / rt are recreated
        // every run, so always tear them down here.
        if (rt != nullptr) {
            // Clear g_current_runtime in this DSO before destroying rt.
            framework_bind_runtime(nullptr);
            // Graph execution blocks are host-owned GM retained by the
            // DeviceRunner. This run only drops its submission references;
            // the blocks remain reusable until Worker finalization.
            runtime_destroy(rt, runtime_arena_);
            rt = nullptr;
        }
        const auto fault_stage =
            static_cast<simpler::hbg::HbgL1FaultStage>(hbg_fault_stage_.load(std::memory_order_acquire));
        if (fault_stage == simpler::hbg::HbgL1FaultStage::RuntimeDestroy) {
            hbg_fault_injected_.store(true, std::memory_order_release);
            latch_run_error(simpler::hbg::hbg_l1_fault_error(fault_stage));
        }
    });
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset all SchedulerContext-owned state in one place.
    sched_ctx_.deinit();

    completion_gate_.reset();
    run_error_.store(0, std::memory_order_release);
    runtime_init_ready_.store(false, std::memory_order_release);
    hbg_restore_error_.store(0, std::memory_order_release);
    hbg_fault_stage_.store(static_cast<uint32_t>(simpler::hbg::HbgL1FaultStage::None), std::memory_order_release);
    hbg_fault_injected_.store(false, std::memory_order_release);
    hbg_unexpected_teardown_error_.store(0, std::memory_order_release);

    aicpu_thread_num_ = 0;

    // Clear the file-scope runtime pointer (freed by the last scheduler thread before deinit).
    rt = nullptr;

    LOG_INFO("DeInit: Runtime execution state reset");

    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    hs_setup_done_.store(false, std::memory_order_release);
    hs_arrived_.store(0, std::memory_order_release);
    hs_thread_seq_.store(0, std::memory_order_release);
    classify_ready_.store(false, std::memory_order_release);
    classify_arrived_.store(0, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);

    LOG_INFO("DeInit: AicpuExecutor reset complete");
}

// ===== Public Entry Point =====

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_register_execution_slot(void *arg) {
    if (arg == nullptr) {
        LOG_ERROR("%s", "simpler_aicpu_l1_hbg_register_execution_slot: null argument");
        return -1;
    }

    // CANN owns a byte snapshot of HostArgs but does not promise the C++
    // alignment of that task-argument address. Copy before typed access.
    simpler::hbg::HbgExecutionSlotRegistration registration{};
    std::memcpy(&registration, arg, sizeof(registration));
    auto *context_registry = current_hbg_context_registry();
    if (context_registry == nullptr || registration.binding.slot_generation != context_registry->context_generation) {
        LOG_ERROR("%s", "simpler_aicpu_l1_hbg_register_execution_slot: context registry mismatch");
        return -1;
    }
    cache_invalidate_range(&context_registry->execution_slot, sizeof(context_registry->execution_slot));
    const auto status = simpler::hbg::publish_hbg_execution_slot_registration(
        &context_registry->execution_slot, &registration, get_orch_device_id()
    );
    if (status != simpler::hbg::HbgExecutionSlotRegistryStatus::Published &&
        status != simpler::hbg::HbgExecutionSlotRegistryStatus::AlreadyRegistered) {
        LOG_ERROR("simpler_aicpu_l1_hbg_register_execution_slot: rejected status=%u", static_cast<unsigned>(status));
        return -1;
    }
    cache_flush_range(&context_registry->execution_slot, sizeof(context_registry->execution_slot));
    return 0;
}

// Compatibility ABI entry. Callable identity and the function table live in
// the authenticated launch blob, so this entry owns no resident state.
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_register_callable(void *arg) {
    return arg == nullptr ? -1 : 0;
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_exec(void *arg) {
    auto *context_registry = current_hbg_context_registry();
    if (context_registry == nullptr) {
        LOG_ERROR("%s", "simpler_aicpu_l1_hbg_exec: context registry unavailable");
        return reject_hbg_l1_without_slot();
    }
    cache_invalidate_range(&context_registry->execution_slot, sizeof(context_registry->execution_slot));
    simpler::hbg::HbgExecutionSlotRegistration slot{};
    const auto slot_status = simpler::hbg::acquire_hbg_execution_slot_registration(
        &context_registry->execution_slot, get_orch_device_id(), &slot
    );
    if (slot_status != simpler::hbg::HbgExecutionSlotRegistryStatus::Acquired ||
        slot.binding.slot_generation != context_registry->context_generation) {
        LOG_ERROR(
            "simpler_aicpu_l1_hbg_exec: execution slot unavailable status=%u", static_cast<unsigned>(slot_status)
        );
        return reject_hbg_l1_without_slot();
    }
    if (arg == nullptr) {
        LOG_ERROR("%s", "simpler_aicpu_l1_hbg_exec: null argument");
        return reject_hbg_l1_before_generation(slot);
    }
    cache_invalidate_range(arg, sizeof(simpler::hbg::HbgLaunchBlobHeader) + sizeof(simpler::hbg::HbgLaunchRegion));
    if (slot.device_kernel_args_size != sizeof(KernelArgs) || slot.outer_runtime_size != sizeof(Runtime)) {
        LOG_ERROR("%s", "simpler_aicpu_l1_hbg_exec: registered runtime ABI size mismatch");
        return reject_hbg_l1_before_generation(slot);
    }

    simpler::hbg::HbgLaunchBlobHeader header{};
    std::memcpy(&header, arg, sizeof(header));
    if (!simpler::hbg::hbg_valid_invocation_identity(header.identity)) {
        LOG_ERROR("%s", "simpler_aicpu_l1_hbg_exec: invalid fixed invocation identity");
        return reject_hbg_l1_before_generation(slot);
    }
    simpler::hbg::HbgAicpuInvocationView invocation{};
    const auto invocation_status = simpler::hbg::make_hbg_aicpu_invocation_view(arg, slot, &invocation);
    if (invocation_status != simpler::hbg::HbgAicpuInvocationStatus::Ok) {
        LOG_ERROR(
            "simpler_aicpu_l1_hbg_exec: fixed invocation rejected status=%u", static_cast<unsigned>(invocation_status)
        );
        return reject_hbg_l1_before_generation(slot);
    }
    auto prelaunch_fault = simpler::hbg::HbgL1FaultStage::None;
    if ((invocation.header.flags & simpler::hbg::HBG_LAUNCH_TEST_FAULT_INJECTION) != 0) {
        cache_invalidate_range(invocation.blob, static_cast<size_t>(invocation.blob_size));
        const auto fault_status = simpler::hbg::authenticate_hbg_aicpu_fault_stage(&invocation, &prelaunch_fault);
        if (fault_status != simpler::hbg::HbgLaunchBlobStatus::Ok) {
            LOG_ERROR(
                "HBG L1 prelaunch fault package authentication failed status=%u", static_cast<unsigned>(fault_status)
            );
            return reject_hbg_l1_before_generation(slot);
        }
    }

    auto *kernel_args = reinterpret_cast<const KernelArgs *>(slot.device_kernel_args_base);
    cache_invalidate_range(kernel_args, sizeof(*kernel_args));
    const bool kernel_args_match = kernel_args->runtime_args != nullptr &&
                                   reinterpret_cast<uint64_t>(kernel_args->runtime_args) == slot.outer_runtime_base;
    const bool inject_kernel_args_fault = prelaunch_fault == simpler::hbg::HbgL1FaultStage::KernelArgsRuntime;
    if (!kernel_args_match || inject_kernel_args_fault) {
        LOG_ERROR("%s", "simpler_aicpu_l1_hbg_exec: KernelArgs runtime does not match the registered slot");
        const bool cancelled = publish_hbg_l1_prelaunch_cancel(simpler::hbg::hbg_l1_launch_control(slot));
        return inject_kernel_args_fault && kernel_args_match && cancelled ? 0 : -1;
    }
    if (prelaunch_fault == simpler::hbg::HbgL1FaultStage::SlotFallbackControl) {
        return publish_hbg_l1_without_slot_cancel() ? 0 : -1;
    }
    if (prelaunch_fault == simpler::hbg::HbgL1FaultStage::PlatformBridge) {
        return publish_hbg_l1_prelaunch_cancel(simpler::hbg::hbg_l1_launch_control(slot)) ? 0 : -1;
    }
    cache_invalidate_range(kernel_args->runtime_args, sizeof(Runtime));
    if (simpler_aicpu_execute_l1_hbg_platform == nullptr) {
        LOG_ERROR("%s", "simpler_aicpu_l1_hbg_exec: platform bridge is unavailable");
        return reject_hbg_l1_before_generation(slot);
    }
    return simpler_aicpu_execute_l1_hbg_platform(kernel_args, &invocation);
}

extern "C" int32_t aicpu_prewarm_callable(Runtime *runtime) {
    // host_build_graph host-orch: the orchestration .so is dlopen'd on the HOST
    // during prepare_callable_impl and the whole task graph is built host-side,
    // so there is no device-side orchestrator .so to pre-load — prewarm is a
    // no-op. The symbol is retained because the platform onboard kernel
    // (src/a2a3/platform/onboard/aicpu/kernel.cpp) links it strongly via
    // simpler_aicpu_prewarm_callable; removing it would break the onboard link.
    (void)runtime;
    return 0;
}

static int32_t
execute_runtime_generation(Runtime *runtime, const simpler::hbg::HbgAicpuInvocationView *hbg_invocation) {
    auto init_fault = simpler::hbg::HbgL1FaultStage::None;
    if (hbg_invocation != nullptr &&
        (hbg_invocation->header.flags & simpler::hbg::HBG_LAUNCH_TEST_FAULT_INJECTION) != 0) {
        cache_invalidate_range(hbg_invocation->blob, static_cast<size_t>(hbg_invocation->blob_size));
        const auto status = simpler::hbg::authenticate_hbg_aicpu_fault_stage(hbg_invocation, &init_fault);
        if (status != simpler::hbg::HbgLaunchBlobStatus::Ok) {
            LOG_ERROR("HBG L1 test fault package authentication failed status=%u", static_cast<unsigned>(status));
            return -1;
        }
    }
    const int32_t init_rc = g_aicpu_executor.init(runtime, init_fault);
    int32_t rc = 0;
    if (init_rc != 0) {
        const int32_t failed_thread_idx = platform_aicpu_affinity_thread_idx();
        if (g_aicpu_executor.aicpu_thread_num_ <= 0 || failed_thread_idx < 0 ||
            failed_thread_idx >= g_aicpu_executor.aicpu_thread_num_) {
            return -1;
        }
        const int32_t shutdown_rc = g_aicpu_executor.sched_ctx_.shutdown(failed_thread_idx);
        if (shutdown_rc != 0) {
            int32_t expected = 0;
            (void)g_aicpu_executor.hbg_unexpected_teardown_error_.compare_exchange_strong(
                expected, shutdown_rc, std::memory_order_acq_rel, std::memory_order_acquire
            );
        }
        g_aicpu_executor.latch_run_error(-1);
        // Every participant closes its assigned AICore windows before joining
        // the generation-wide finalize/depart protocol. shutdown() is
        // idempotent, including failures that already ran emergency_shutdown().
        g_aicpu_executor.arrive_and_finalize_run();
        rc = -1;
    } else {
        rc = g_aicpu_executor.run(runtime, hbg_invocation);
    }

    g_aicpu_executor.completion_gate_.wait_for_finalization();
    const int32_t shared_error = g_aicpu_executor.run_error_.load(std::memory_order_acquire);
    if (shared_error != 0) rc = shared_error;
    const auto requested_fault =
        static_cast<simpler::hbg::HbgL1FaultStage>(g_aicpu_executor.hbg_fault_stage_.load(std::memory_order_acquire));
    const int32_t requested_fault_error = simpler::hbg::hbg_l1_fault_error(requested_fault);
    const bool controlled_fault = g_aicpu_executor.hbg_fault_injected_.load(std::memory_order_acquire) &&
                                  requested_fault_error != 0 && shared_error == requested_fault_error;
    const int32_t unexpected_teardown_error =
        g_aicpu_executor.hbg_unexpected_teardown_error_.load(std::memory_order_acquire);

    const bool runtime_state_readable =
        init_rc == 0 &&
        (hbg_invocation == nullptr || g_aicpu_executor.hbg_restore_error_.load(std::memory_order_acquire) == 0);
    const int32_t runtime_rc = runtime_state_readable ? read_pto2_runtime_status(runtime) : 0;

    if (g_aicpu_executor.completion_gate_.depart_and_claim_cleanup_if_last(g_aicpu_executor.aicpu_thread_num_)) {
        g_aicpu_executor.deinit(runtime);
    }
    if (runtime_rc != 0 && (!controlled_fault || runtime_rc != requested_fault_error)) return runtime_rc;
    if (unexpected_teardown_error != 0) return unexpected_teardown_error;
    if (controlled_fault) return 0;
    return rc;
}

/**
 * aicpu_execute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task runtime execution:
 * 1. Initialize executor: all threads enter init(), which handshakes the cores
 *    in parallel and barriers internally until init is complete (or a thread
 *    failed); its return value is authoritative on every thread.
 * 2. Execute tasks on managed cores
 * 3. Cleanup when last thread finishes
 *
 * @param runtime Pointer to Runtime structure
 * @return 0 on success, non-zero on error
 */
extern "C" int32_t aicpu_execute(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    LOG_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

    const int32_t rc = execute_runtime_generation(runtime, nullptr);
    if (rc != 0) {
        LOG_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
        return rc;
    }

    LOG_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}

// Internal half of the platform trampoline. Keep it local to this runtime DSO
// for the same reason as simpler_aicpu_execute_l1_hbg_platform: sequential TRB
// and HBG contexts share one AICPU scheduler dynamic-link namespace.
extern "C" __attribute__((visibility("hidden"))) int32_t
aicpu_execute_l1_hbg(Runtime *runtime, const simpler::hbg::HbgAicpuInvocationView *invocation) {
    if (runtime == nullptr || invocation == nullptr) {
        LOG_ERROR("%s", "aicpu_execute_l1_hbg: invalid argument");
        return -1;
    }

    return execute_runtime_generation(runtime, invocation);
}

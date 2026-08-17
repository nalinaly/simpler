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
 * @file load_aicpu_op.h
 * @brief Host-side AICPU operation loader.
 *
 * L2/L3 owned-device architecture:
 *
 *   1. BootstrapDispatcher (per-DeviceRunner, idempotent across instances in
 *      the same process via a content-fingerprint cache): bundles dispatcher
 *      SO bytes + runtime SO bytes into a single
 *      `rtAicpuKernelLaunchExWithArgs` (kernel_type =
 *      `KERNEL_TYPE_AICPU_KFC`) targeting libaicpu_extend_kernels. Our
 *      dispatcher then writes the runtime SO to
 *      `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<device_id>.so`
 *      using sched-thread (HwHiAiUser) write permission. The dispatcher SO
 *      itself is never persisted to disk.
 *
 *   2. Init (per-DeviceRunner): JSON-registers the runtime SO via
 *      `rtsBinaryLoadFromFile` (cpuKernelMode=0, kernelSo points at the
 *      preinstall basename), then resolves runtime SO entry points such as
 *      `simpler_aicpu_exec`, `simpler_aicpu_init`, and
 *      `simpler_aicpu_register_callable` to `rtFuncHandle`s via
 *      `rtsFuncGetByName`. JSON is per-process
 *      (`/tmp/simpler_inner_<fp>_<pid>.json`) so concurrent multi-chip /
 *      multi-worker tests don't race on a shared file.
 *
 *   3. LaunchBuiltInOp (per-task): `rtsLaunchCpuKernel` on the cached
 *      `rtFuncHandle`. No per-launch string marshalling, no global op
 *      registry lookups.
 *
 * See common/aicpu_loader/device/aicpu_dispatcher.h for the bootstrap protocol
 * details (extended DeviceArgs with inner_so_bin/inner_so_len,
 * fingerprint-named preinstall files).
 *
 * L1 borrowed-device architecture:
 *
 *   1. InitFromData loads the AICPU runtime SO directly from its host byte
 *      image with cpuKernelMode=2 and registers every entry with
 *      aclrtRegisterCpuFunc. No dispatcher task, device-side file, JSON, or
 *      stream synchronization participates in this path.
 *
 *   2. LaunchWithHostArgs enqueues a runtime-owned argument snapshot on the
 *      caller-provided stream through aclrtLaunchKernelWithHostArgs.
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/kernel_args.h"
#include "runtime/runtime/rts/rts_kernel.h"
#include "runtime/rt.h"
#include "task_interface/host_args_launch.h"

namespace host {

/**
 * @brief AICPU operation configuration for JSON descriptor generation.
 */
struct AicpuOpConfig {
    std::string functionName;
    std::string kernelSo;
    std::string opKernelLib;
    std::string computeCost = "100";
    std::string engine = "DNN_VM_AICPU";
    std::string flagAsync = "False";
    std::string flagPartial = "False";
    std::string userDefined = "False";
    std::string opType;
};

/**
 * @brief Host-side AICPU operation loader.
 *
 * One instance per DeviceRunner; manages bootstrap (dispatcher upload) +
 * JSON registration of the runtime SO + per-task launches via the runtime
 * SO's direct rtFuncHandles.
 */
class LoadAicpuOp {
public:
    LoadAicpuOp() = default;
    ~LoadAicpuOp();

    LoadAicpuOp(const LoadAicpuOp &) = delete;
    LoadAicpuOp &operator=(const LoadAicpuOp &) = delete;
    LoadAicpuOp(LoadAicpuOp &&) = delete;
    LoadAicpuOp &operator=(LoadAicpuOp &&) = delete;

    /**
     * @brief One-shot bootstrap: upload runtime SO to preinstall via dispatcher.
     *
     * @param dispatcher_so_data  Dispatcher SO bytes (caller-owned, must outlive call)
     * @param dispatcher_so_len   Dispatcher SO size
     * @param inner_so_data       Runtime SO bytes (caller-owned, must outlive call)
     * @param inner_so_len        Runtime SO size
     * @param stream              Stream on which to enqueue the bootstrap
     * @param device_id           ACL device ordinal; suffixed onto the preinstall
     *                            basename (simpler_inner_<fp>_<device_id>.so) so
     *                            paired dies sharing the filesystem do not write
     *                            and execute the same on-disk SO.
     * @return 0 on success, error code on failure
     */
    int BootstrapDispatcher(
        const void *dispatcher_so_data, size_t dispatcher_so_len, const void *inner_so_data, size_t inner_so_len,
        rtStream_t stream, int device_id
    );

    /**
     * @brief JSON-register the runtime SO and resolve its entry handles.
     *
     * @param extra_symbols  Runtime-specific AICPU entry symbols beyond the base
     *                       set ({RunName, InitName}, exported by every runtime).
     *                       Each runtime's host part reports what it additionally
     *                       exports — e.g. TMARB adds RegisterCallableName, while
     *                       host_build_graph reports none. This keeps the common
     *                       loader free of any runtime-specific symbol knowledge.
     */
    int Init(const std::vector<std::string> &extra_symbols);

    /**
     * @brief Load the runtime SO from host bytes and register its AICPU entries.
     *
     * @param inner_so_data  Runtime SO bytes (caller-owned, must outlive call)
     * @param inner_so_len   Runtime SO size
     * @param extra_symbols  Runtime-specific entries beyond {RunName, InitName}
     * @return 0 on success, error code on failure
     */
    int InitFromData(const void *inner_so_data, size_t inner_so_len, const std::vector<std::string> &extra_symbols);

    /**
     * @brief Release binary handle, function handles, and temporary JSON.
     *
     * A data-loaded borrowed-device binary remains owned when unload fails so
     * explicit L1 close can retry. A file-loaded owned-device binary clears
     * its host handle even when RTS reports an unload failure because the L2
     * teardown immediately destroys that RTS context.
     */
    int Finalize();

    /**
     * @brief Launch a runtime SO entry point via rtsLaunchCpuKernel.
     *
     * @param stream       RTS stream
     * @param args         Launch-arg payload (KernelArgs for exec, InitArgs for
     *                     init, RegisterCallableArgs for register_callable)
     * @param args_size    Size of the payload in bytes
     * @param aicpu_num    Number of AICPU threads
     * @param func_name    Lookup key in func_handles_ (KernelNames::*)
     * @return 0 on success, error code on failure
     */
    int LaunchBuiltInOp(rtStream_t stream, void *args, size_t args_size, int aicpu_num, const std::string &func_name);

    /**
     * @brief Launch a registered AICPU entry with a runtime-owned host-args snapshot.
     *
     * @param stream       Caller-owned stream
     * @param host_args    Host argument image copied by the runtime during enqueue
     * @param args_size    Size of the host argument image
     * @param aicpu_num    Number of AICPU blocks
     * @param func_name    Lookup key in func_handles_ (KernelNames::*)
     * @return 0 on success, error code on failure
     */
    int LaunchWithHostArgs(
        rtStream_t stream, const void *host_args, size_t args_size, int aicpu_num, const char *func_name
    );

    /**
     * @brief Launch a mutable variable-length host-args image with inline placeholders.
     *
     * This is the HBG-facing bridge. Unlike the fixed TRB overload, callers
     * must provide a fresh writable blob because CANN may patch pointer fields
     * while snapshotting the task. The method validates all lossy size carriers
     * and placeholder writes before invoking the runtime and performs no host
     * allocation or synchronization.
     *
     * @param stream             Caller-owned stream
     * @param host_args          Fresh writable host argument image
     * @param args_size          Exact byte length of the argument image
     * @param placeholders       Placeholder descriptors, or null when count is zero
     * @param placeholder_count  Number of descriptors
     * @param aicpu_num          Number of AICPU blocks
     * @param func_name          Lookup key in func_handles_ (KernelNames::*)
     * @return 0 on successful enqueue, error code otherwise
     */
    int LaunchWithMutableHostArgs(
        rtStream_t stream, void *host_args, size_t args_size, simpler::host_args::HostArgsPlaceholder *placeholders,
        size_t placeholder_count, int aicpu_num, const char *func_name
    );

private:
    enum class BinaryLoadMode : uint8_t {
        None,
        RtsFile,
        AclData,
    };

    void *binary_handle_ = nullptr;
    BinaryLoadMode binary_load_mode_ = BinaryLoadMode::None;
    std::unordered_map<std::string, rtFuncHandle> func_handles_;
    std::string json_file_path_;
    uint64_t inner_fp_ = 0;
    int device_id_ = 0;
    std::string inner_so_basename_;
    // Full set of AICPU entry symbols to JSON-register and resolve: the base
    // {RunName, InitName} plus the runtime-reported extras passed to Init().
    std::vector<std::string> kernel_symbols_;

    bool GenerateAicpuOpJson(const std::string &json_path, const std::string &kernel_so);
    void SetKernelSymbols(const std::vector<std::string> &extra_symbols);
    int AicpuKernelLaunch(rtFuncHandle func_handle, rtStream_t stream, void *args, size_t args_size, int aicpu_num);
};

// Runtime SO's actual exported symbol name. Looked up via the runtime SO's
// own JSON registration (no dispatcher hop at runtime).
namespace KernelNames {
constexpr const char *RunName = "simpler_aicpu_exec";   // multi-threaded exec
constexpr const char *InitName = "simpler_aicpu_init";  // per-device one-shot invariants
constexpr const char *RegisterCallableName = "simpler_aicpu_register_callable";
constexpr const char *L1RegisterCallableName = "simpler_aicpu_l1_register_callable";
constexpr const char *L1HbgRegisterExecutionSlotName = "simpler_aicpu_l1_hbg_register_execution_slot";
constexpr const char *L1RunName = "simpler_aicpu_l1_exec";
}  // namespace KernelNames

}  // namespace host

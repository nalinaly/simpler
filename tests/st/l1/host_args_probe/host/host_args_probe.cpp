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

#include <acl/acl.h>
#include <runtime/rt.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "host_args_probe_abi.h"

namespace {

using simpler::test::host_args_probe::host_args_probe_checksum;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_MAGIC;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_OK;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_PAYLOAD_COUNT;
using simpler::test::host_args_probe::HOST_ARGS_PROBE_RESULT_MAGIC;
using simpler::test::host_args_probe::HostArgsProbeHeader;
using simpler::test::host_args_probe::HostArgsProbeResult;

constexpr uint32_t kBootstrapDeviceArgsBytes = 256;
constexpr size_t kBootstrapDispatcherBinaryOffset = 96;
constexpr size_t kBootstrapDispatcherLengthOffset = 104;
constexpr size_t kBootstrapDeviceIdOffset = 112;
constexpr size_t kBootstrapInnerBinaryOffset = 120;
constexpr size_t kBootstrapInnerLengthOffset = 128;
constexpr uint8_t kReleasedScratchPoison = 0xa5;
constexpr uint8_t kReleasedScratchReuse = 0x5a;

bool CheckAcl(aclError rc, const char *expression, const char *stage) {
    if (rc == ACL_SUCCESS) return true;
    std::fprintf(stderr, "FAIL %s: %s rc=%d\n", stage, expression, static_cast<int>(rc));
    return false;
}

bool CheckRt(rtError_t rc, const char *expression, const char *stage) {
    if (rc == RT_ERROR_NONE) return true;
    std::fprintf(stderr, "FAIL %s: %s rc=%d\n", stage, expression, static_cast<int>(rc));
    return false;
}

#define PROBE_ACL(expr, stage) CheckAcl((expr), #expr, (stage))
#define PROBE_RT(expr, stage) CheckRt((expr), #expr, (stage))

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer() { Reset(); }

    bool Allocate(size_t size) {
        if (pointer_ != nullptr || size == 0) return false;
        if (!PROBE_ACL(aclrtMalloc(&pointer_, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) return false;
        size_ = size;
        return true;
    }

    void Reset() noexcept {
        if (pointer_ != nullptr) {
            (void)aclrtFree(pointer_);
            pointer_ = nullptr;
            size_ = 0;
        }
    }

    bool ClearAsync(aclrtStream stream) {
        return pointer_ != nullptr && PROBE_ACL(aclrtMemsetAsync(pointer_, size_, 0, size_, stream), "clear result");
    }

    bool Clear() { return pointer_ != nullptr && PROBE_ACL(aclrtMemset(pointer_, size_, 0, size_), "clear result"); }

    bool CopyFromHost(const void *source, size_t size) {
        return pointer_ != nullptr && size <= size_ &&
               PROBE_ACL(aclrtMemcpy(pointer_, size_, source, size, ACL_MEMCPY_HOST_TO_DEVICE), "copy to device");
    }

    bool CopyToHost(void *destination, size_t size) const {
        return pointer_ != nullptr && size <= size_ &&
               PROBE_ACL(aclrtMemcpy(destination, size, pointer_, size, ACL_MEMCPY_DEVICE_TO_HOST), "copy from device");
    }

    void *get() const noexcept { return pointer_; }
    size_t size() const noexcept { return size_; }

private:
    void *pointer_{nullptr};
    size_t size_{0};
};

class AclSession {
public:
    AclSession() = default;
    AclSession(const AclSession &) = delete;
    AclSession &operator=(const AclSession &) = delete;

    ~AclSession() {
        if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
        if (context_ != nullptr) (void)aclrtDestroyContext(context_);
        // This is an isolated probe process.  aclFinalize releases process
        // state, while intentionally omitting aclrtResetDevice.
        if (initialized_) (void)aclFinalize();
    }

    bool Initialize(int device_id) {
        if (!PROBE_ACL(aclInit(nullptr), "aclInit")) return false;
        initialized_ = true;
        if (!PROBE_ACL(aclrtSetDevice(device_id), "aclrtSetDevice")) return false;
        if (!PROBE_ACL(aclrtCreateContext(&context_, device_id), "aclrtCreateContext")) return false;
        if (!PROBE_ACL(aclrtCreateStream(&stream_), "aclrtCreateStream")) return false;
        return true;
    }

    aclrtStream stream() const noexcept { return stream_; }

private:
    bool initialized_{false};
    aclrtContext context_{nullptr};
    aclrtStream stream_{nullptr};
};

class AicpuBinary {
public:
    AicpuBinary() = default;
    AicpuBinary(const AicpuBinary &) = delete;
    AicpuBinary &operator=(const AicpuBinary &) = delete;
    ~AicpuBinary() {
        if (handle_ != nullptr) (void)aclrtBinaryUnLoad(handle_);
    }

    aclrtBinHandle *out() noexcept { return &handle_; }
    aclrtBinHandle get() const noexcept { return handle_; }

private:
    aclrtBinHandle handle_{nullptr};
};

class ModelOwner {
public:
    ModelOwner() = default;
    ModelOwner(const ModelOwner &) = delete;
    ModelOwner &operator=(const ModelOwner &) = delete;
    ~ModelOwner() { Reset(); }

    void Adopt(aclmdlRI model) noexcept { model_ = model; }
    aclmdlRI get() const noexcept { return model_; }
    bool Reset() noexcept {
        if (model_ == nullptr) return true;
        const aclError rc = aclmdlRIDestroy(model_);
        if (rc != ACL_SUCCESS) {
            std::fprintf(stderr, "FAIL aclmdlRIDestroy rc=%d\n", static_cast<int>(rc));
            return false;
        }
        model_ = nullptr;
        return true;
    }

private:
    aclmdlRI model_{nullptr};
};

struct Options {
    int device_id{-1};
    std::vector<size_t> eager_sizes{64U << 10U, 1U << 20U, 16U << 20U, 64U << 20U};
    std::array<size_t, 2> graph_sizes{1U << 20U, 16U << 20U};
    size_t replay_count{100};
    size_t pressure_count{512};
    size_t pressure_size{64U << 10U};
};

struct Invocation {
    uint64_t id{0};
    uint64_t canonical_hash{0};
    std::vector<uint8_t> canonical;
    std::array<aclrtPlaceHolderInfo, HOST_ARGS_PROBE_PAYLOAD_COUNT> placeholders{};
};

struct CapturedInvocation {
    Invocation invocation;
    DeviceBuffer result;
    ModelOwner model;
};

bool ReadFile(const char *path, std::vector<char> *output) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    const std::streampos end = file.tellg();
    if (end < 0) return false;
    output->assign(static_cast<size_t>(end), 0);
    file.seekg(0);
    if (end > 0 && !file.read(output->data(), end)) {
        std::fprintf(stderr, "cannot read %s\n", path);
        return false;
    }
    return true;
}

void WriteU64(char *bytes, size_t offset, uint64_t value) { std::memcpy(bytes + offset, &value, sizeof(value)); }

bool ReadElfBuildId(const char *data, size_t size, uint64_t *output) {
    if (size < 64 ||
        std::memcmp(
            data,
            "\x7f"
            "ELF",
            4
        ) != 0 ||
        data[4] != 2)
        return false;
    uint64_t section_offset = 0;
    uint16_t section_entry_size = 0;
    uint16_t section_count = 0;
    uint16_t string_section_index = 0;
    std::memcpy(&section_offset, data + 40, sizeof(section_offset));
    std::memcpy(&section_entry_size, data + 58, sizeof(section_entry_size));
    std::memcpy(&section_count, data + 60, sizeof(section_count));
    std::memcpy(&string_section_index, data + 62, sizeof(string_section_index));
    if (section_entry_size != 64 || section_offset > size ||
        static_cast<uint64_t>(section_entry_size) * section_count > size - section_offset ||
        string_section_index >= section_count) {
        return false;
    }

    const char *string_section =
        data + section_offset + static_cast<uint64_t>(section_entry_size) * string_section_index;
    uint64_t string_offset = 0;
    uint64_t string_size = 0;
    std::memcpy(&string_offset, string_section + 24, sizeof(string_offset));
    std::memcpy(&string_size, string_section + 32, sizeof(string_size));
    if (string_offset > size || string_size > size - string_offset) return false;
    const char *strings = data + string_offset;

    for (uint16_t index = 0; index < section_count; ++index) {
        const char *section = data + section_offset + static_cast<uint64_t>(section_entry_size) * index;
        uint32_t name_offset = 0;
        uint32_t type = 0;
        uint64_t note_offset = 0;
        uint64_t note_size = 0;
        std::memcpy(&name_offset, section, sizeof(name_offset));
        std::memcpy(&type, section + 4, sizeof(type));
        std::memcpy(&note_offset, section + 24, sizeof(note_offset));
        std::memcpy(&note_size, section + 32, sizeof(note_size));
        if (type != 7 || name_offset >= string_size) continue;
        const size_t max_name = string_size - name_offset;
        const size_t name_size = strnlen(strings + name_offset, max_name);
        if (name_size == max_name || std::strcmp(strings + name_offset, ".note.gnu.build-id") != 0) continue;
        if (note_size < 20 || note_offset > size || note_size > size - note_offset) return false;
        const char *note = data + note_offset;
        uint32_t name_bytes = 0;
        uint32_t description_bytes = 0;
        uint32_t note_type = 0;
        std::memcpy(&name_bytes, note, sizeof(name_bytes));
        std::memcpy(&description_bytes, note + 4, sizeof(description_bytes));
        std::memcpy(&note_type, note + 8, sizeof(note_type));
        const size_t aligned_name_bytes = (static_cast<size_t>(name_bytes) + 3U) & ~size_t{3U};
        if (note_type != 3 || description_bytes < sizeof(uint64_t) || aligned_name_bytes > note_size - 12 ||
            note_size - 12 - aligned_name_bytes < sizeof(uint64_t)) {
            return false;
        }
        std::memcpy(output, note + 12 + aligned_name_bytes, sizeof(*output));
        return true;
    }
    return false;
}

uint64_t FingerprintBytes(const void *data, size_t size) {
    uint64_t build_id = 0;
    if (ReadElfBuildId(static_cast<const char *>(data), size, &build_id)) return build_id;
    return host_args_probe_checksum(static_cast<const uint8_t *>(data), size);
}

std::string MakeAicpuDescriptor(uint64_t fingerprint, const std::string &so_basename) {
    char init_op[128]{};
    char run_op[128]{};
    std::snprintf(init_op, sizeof(init_op), "simpler_host_args_probe_init_%016" PRIx64, fingerprint);
    std::snprintf(run_op, sizeof(run_op), "simpler_host_args_probe_run_%016" PRIx64, fingerprint);
    const auto entry = [&](const char *op_name, const char *function_name) {
        std::string value = "  \"";
        value += op_name;
        value += "\": {\n    \"opInfo\": {\n      \"functionName\": \"";
        value += function_name;
        value += "\",\n      \"kernelSo\": \"";
        value += so_basename;
        value += "\",\n      \"opKernelLib\": \"AICPUKernel\",\n";
        value += "      \"computeCost\": \"100\",\n      \"engine\": \"DNN_VM_AICPU\",\n";
        value += "      \"flagAsync\": \"False\",\n      \"flagPartial\": \"False\",\n";
        value += "      \"userDefined\": \"False\"\n    }\n  }";
        return value;
    };
    return "{\n" + entry(init_op, "simpler_host_args_probe_init") + ",\n" +
           entry(run_op, "simpler_host_args_probe_run") + "\n}\n";
}

int BootstrapAicpu(
    int device_id, aclrtStream stream, const std::vector<char> &dispatcher, const std::vector<char> &inner
) {
    DeviceBuffer device_dispatcher;
    DeviceBuffer device_inner;
    DeviceBuffer device_args;
    if (!device_dispatcher.Allocate(dispatcher.size()) || !device_inner.Allocate(inner.size()) ||
        !device_args.Allocate(kBootstrapDeviceArgsBytes)) {
        return 1;
    }
    if (!device_dispatcher.CopyFromHost(dispatcher.data(), dispatcher.size()) ||
        !device_inner.CopyFromHost(inner.data(), inner.size())) {
        return 1;
    }

    std::array<char, kBootstrapDeviceArgsBytes> host_device_args{};
    WriteU64(
        host_device_args.data(), kBootstrapDispatcherBinaryOffset, reinterpret_cast<uint64_t>(device_dispatcher.get())
    );
    WriteU64(host_device_args.data(), kBootstrapDispatcherLengthOffset, dispatcher.size());
    WriteU64(host_device_args.data(), kBootstrapDeviceIdOffset, static_cast<uint64_t>(device_id));
    WriteU64(host_device_args.data(), kBootstrapInnerBinaryOffset, reinterpret_cast<uint64_t>(device_inner.get()));
    WriteU64(host_device_args.data(), kBootstrapInnerLengthOffset, inner.size());
    if (!device_args.CopyFromHost(host_device_args.data(), host_device_args.size())) return 1;

    struct BootstrapArgs {
        struct {
            uint64_t unused[5]{};
            uint64_t device_args_ptr{0};
            uint64_t padding[20]{};
        } kernel_args;
        char kernel_name[32]{};
        char so_name[32]{};
        char op_name[32]{};
    } arguments{};
    arguments.kernel_args.device_args_ptr = reinterpret_cast<uint64_t>(device_args.get());
    std::strncpy(arguments.kernel_name, "DynTileFwkKernelServerInit", sizeof(arguments.kernel_name) - 1);
    std::strncpy(arguments.so_name, "libaicpu_extend_kernels.so", sizeof(arguments.so_name) - 1);

    rtAicpuArgsEx_t runtime_args{};
    runtime_args.args = &arguments;
    runtime_args.argsSize = sizeof(arguments);
    runtime_args.kernelNameAddrOffset = offsetof(BootstrapArgs, kernel_name);
    runtime_args.soNameAddrOffset = offsetof(BootstrapArgs, so_name);
    if (!PROBE_RT(
            rtAicpuKernelLaunchExWithArgs(
                rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1, &runtime_args, nullptr, stream, 0
            ),
            "bootstrap AICPU dispatcher"
        )) {
        return 1;
    }
    return PROBE_ACL(aclrtSynchronizeStream(stream), "synchronize AICPU bootstrap") ? 0 : 1;
}

int LoadAicpuRunHandle(
    const std::vector<char> &inner, int device_id, AicpuBinary *binary, aclrtFuncHandle *run_handle
) {
    const uint64_t fingerprint = FingerprintBytes(inner.data(), inner.size());
    char so_basename[96]{};
    std::snprintf(
        so_basename, sizeof(so_basename), "simpler_host_args_probe_%016" PRIx64 "_%d.so", fingerprint, device_id
    );
    char descriptor_path[] = "/tmp/gpt_host_args_probe_XXXXXX.json";
    const int descriptor_fd = mkstemps(descriptor_path, 5);
    if (descriptor_fd < 0) {
        std::fprintf(stderr, "mkstemps failed: %s\n", std::strerror(errno));
        return 1;
    }
    const std::string descriptor = MakeAicpuDescriptor(fingerprint, so_basename);
    const ssize_t written = write(descriptor_fd, descriptor.data(), descriptor.size());
    close(descriptor_fd);
    if (written != static_cast<ssize_t>(descriptor.size())) {
        std::remove(descriptor_path);
        return 1;
    }

    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    aclrtBinaryLoadOptions options{};
    options.options = &option;
    options.numOpt = 1;
    aclError rc = aclrtBinaryLoadFromFile(descriptor_path, &options, binary->out());
    std::remove(descriptor_path);
    if (!CheckAcl(rc, "aclrtBinaryLoadFromFile", "load probe AICPU binary")) return 1;

    char run_op[128]{};
    std::snprintf(run_op, sizeof(run_op), "simpler_host_args_probe_run_%016" PRIx64, fingerprint);
    rc = aclrtBinaryGetFunction(binary->get(), run_op, run_handle);
    if (rc != ACL_SUCCESS) {
        rc = aclrtRegisterCpuFunc(binary->get(), "simpler_host_args_probe_run", run_op, run_handle);
    }
    if (!CheckAcl(rc, "aclrtBinaryGetFunction/aclrtRegisterCpuFunc", "resolve probe run function")) return 1;
    std::printf("AICPU probe loaded: op=%s handle=%p\n", run_op, static_cast<void *>(*run_handle));
    return 0;
}

size_t AlignUp(size_t value, size_t alignment) { return (value + alignment - 1U) & ~(alignment - 1U); }

uint8_t PatternByte(uint64_t invocation_id, size_t region, size_t index) {
    const uint64_t mixed = invocation_id * 17U + region * 53U + index * 131U + (index >> 7U) * 29U;
    return static_cast<uint8_t>((mixed ^ (mixed >> 11U) ^ (mixed >> 23U)) & 0xffU);
}

bool BuildInvocation(size_t total_size, uint64_t invocation_id, uint64_t result_addr, Invocation *output) {
    if (output == nullptr || result_addr == 0 || total_size < 256 ||
        total_size > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    output->id = invocation_id;
    output->canonical.assign(total_size, 0);

    HostArgsProbeHeader header{};
    header.magic = HOST_ARGS_PROBE_MAGIC;
    header.total_size = static_cast<uint32_t>(total_size);
    header.invocation_id = invocation_id;
    header.result_addr = result_addr;

    const size_t payload_begin = sizeof(HostArgsProbeHeader);
    const size_t payload_bytes = total_size - payload_begin;
    const size_t offsets[HOST_ARGS_PROBE_PAYLOAD_COUNT] = {
        payload_begin,
        AlignUp(payload_begin + payload_bytes / 3U, alignof(uint64_t)),
        AlignUp(payload_begin + (payload_bytes * 2U) / 3U, alignof(uint64_t)),
    };
    if (offsets[1] <= offsets[0] || offsets[2] <= offsets[1] || offsets[2] >= total_size) return false;

    for (size_t index = 0; index < HOST_ARGS_PROBE_PAYLOAD_COUNT; ++index) {
        const size_t end = index + 1U == HOST_ARGS_PROBE_PAYLOAD_COUNT ? total_size : offsets[index + 1U];
        const size_t size = end - offsets[index];
        header.payload_offset[index] = static_cast<uint32_t>(offsets[index]);
        header.payload_size[index] = static_cast<uint32_t>(size);
        uint8_t *payload = output->canonical.data() + offsets[index];
        for (size_t byte = 0; byte < size; ++byte)
            payload[byte] = PatternByte(invocation_id, index, byte);
        header.expected_checksum[index] = host_args_probe_checksum(payload, size);
        header.expected_first[index] = payload[0];
        header.expected_middle[index] = payload[size / 2U];
        header.expected_tail[index] = payload[size - 1U];
        output->placeholders[index].addrOffset =
            static_cast<uint32_t>(offsetof(HostArgsProbeHeader, payload_addr) + index * sizeof(uint64_t));
        output->placeholders[index].dataOffset = header.payload_offset[index];
    }
    std::memcpy(output->canonical.data(), &header, sizeof(header));
    output->canonical_hash = host_args_probe_checksum(output->canonical.data(), output->canonical.size());
    return true;
}

void ReadHostPointerSlots(
    const std::vector<uint8_t> &scratch, std::array<uint64_t, HOST_ARGS_PROBE_PAYLOAD_COUNT> *slots
) {
    for (size_t index = 0; index < HOST_ARGS_PROBE_PAYLOAD_COUNT; ++index) {
        std::memcpy(
            &(*slots)[index], scratch.data() + offsetof(HostArgsProbeHeader, payload_addr) + index * sizeof(uint64_t),
            sizeof(uint64_t)
        );
    }
}

void PoisonReleaseAndReuse(std::vector<uint8_t> *scratch) {
    const size_t size = scratch->size();
    std::fill(scratch->begin(), scratch->end(), kReleasedScratchPoison);
    std::vector<uint8_t>().swap(*scratch);
    // Encourage immediate reuse of the released host range.  Correctness must
    // not depend on the allocator returning the same virtual address.
    std::vector<uint8_t> reuse(size, kReleasedScratchReuse);
    volatile uint8_t tail = reuse.empty() ? 0 : reuse.back();
    (void)tail;
}

aclError EnqueueInvocation(
    aclrtFuncHandle run_handle, aclrtStream stream, Invocation *invocation,
    std::array<uint64_t, HOST_ARGS_PROBE_PAYLOAD_COUNT> *host_pointer_slots
) {
    std::vector<uint8_t> scratch = invocation->canonical;
    aclError rc = aclrtLaunchKernelWithHostArgs(
        run_handle, 1, stream, nullptr, scratch.data(), scratch.size(), invocation->placeholders.data(),
        invocation->placeholders.size()
    );
    ReadHostPointerSlots(scratch, host_pointer_slots);
    PoisonReleaseAndReuse(&scratch);
    if (host_args_probe_checksum(invocation->canonical.data(), invocation->canonical.size()) !=
        invocation->canonical_hash) {
        std::fprintf(stderr, "FAIL canonical image changed for invocation=%" PRIu64 "\n", invocation->id);
        return ACL_ERROR_INVALID_PARAM;
    }
    return rc;
}

bool ValidateResult(const Invocation &invocation, const DeviceBuffer &device_result, const char *stage) {
    HostArgsProbeResult result{};
    if (!device_result.CopyToHost(&result, sizeof(result))) return false;
    if (result.magic != HOST_ARGS_PROBE_RESULT_MAGIC || result.invocation_id != invocation.id ||
        result.total_size != invocation.canonical.size() || result.status != HOST_ARGS_PROBE_OK) {
        std::fprintf(
            stderr,
            "FAIL %s result: magic=%016" PRIx64 " id=%" PRIu64 "/%" PRIu64 " size=%u/%zu status=0x%08x args=%016" PRIx64
            " mod64=%u\n",
            stage, result.magic, result.invocation_id, invocation.id, result.total_size, invocation.canonical.size(),
            result.status, result.args_base, result.args_base_mod_64
        );
        for (size_t index = 0; index < HOST_ARGS_PROBE_PAYLOAD_COUNT; ++index) {
            std::fprintf(
                stderr,
                "  region%zu addr=%016" PRIx64 "/%016" PRIx64 " checksum=%016" PRIx64 "/%016" PRIx64
                " samples=%02x,%02x,%02x\n",
                index, result.observed_addr[index], result.expected_addr[index], result.observed_checksum[index],
                result.expected_checksum[index], result.observed_first[index], result.observed_middle[index],
                result.observed_tail[index]
            );
        }
        return false;
    }
    std::printf(
        "PASS %-24s id=%" PRIu64 " size=%zu args=%016" PRIx64 " mod64=%u\n", stage, invocation.id,
        invocation.canonical.size(), result.args_base, result.args_base_mod_64
    );
    return true;
}

void PrintMemory(const char *stage) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const aclError rc = aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes);
    if (rc != ACL_SUCCESS) {
        std::printf("MEM %-24s unavailable rc=%d\n", stage, static_cast<int>(rc));
        return;
    }
    std::printf("MEM %-24s free=%zu total=%zu used=%zu\n", stage, free_bytes, total_bytes, total_bytes - free_bytes);
}

bool RunEagerSize(aclrtFuncHandle run_handle, aclrtStream stream, size_t args_size, uint64_t invocation_id) {
    DeviceBuffer result;
    if (!result.Allocate(sizeof(HostArgsProbeResult)) || !result.ClearAsync(stream)) return false;
    Invocation invocation;
    if (!BuildInvocation(args_size, invocation_id, reinterpret_cast<uint64_t>(result.get()), &invocation)) {
        std::fprintf(stderr, "FAIL build eager invocation size=%zu\n", args_size);
        return false;
    }
    std::array<uint64_t, HOST_ARGS_PROBE_PAYLOAD_COUNT> host_slots{};
    const auto begin = std::chrono::steady_clock::now();
    const aclError launch_rc = EnqueueInvocation(run_handle, stream, &invocation, &host_slots);
    const auto launch_end = std::chrono::steady_clock::now();
    if (!CheckAcl(launch_rc, "aclrtLaunchKernelWithHostArgs", "eager probe launch")) return false;
    if (!PROBE_ACL(aclrtSynchronizeStream(stream), "synchronize eager probe")) return false;
    const auto end = std::chrono::steady_clock::now();
    if (!ValidateResult(invocation, result, "eager snapshot")) return false;
    std::printf(
        "TIME eager size=%zu launch_us=%lld complete_us=%lld host_patch=[%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64
        "]\n",
        args_size,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(launch_end - begin).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()),
        host_slots[0], host_slots[1], host_slots[2]
    );
    return true;
}

bool CaptureInvocation(
    aclrtFuncHandle run_handle, aclrtStream stream, size_t args_size, uint64_t invocation_id,
    CapturedInvocation *captured
) {
    if (!captured->result.Allocate(sizeof(HostArgsProbeResult)) || !captured->result.Clear() ||
        !BuildInvocation(
            args_size, invocation_id, reinterpret_cast<uint64_t>(captured->result.get()), &captured->invocation
        )) {
        return false;
    }
    std::vector<uint8_t> scratch = captured->invocation.canonical;
    std::array<uint64_t, HOST_ARGS_PROBE_PAYLOAD_COUNT> host_slots{};
    const auto begin = std::chrono::steady_clock::now();
    if (!PROBE_ACL(aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "begin probe capture")) {
        return false;
    }
    aclError launch_rc = aclrtLaunchKernelWithHostArgs(
        run_handle, 1, stream, nullptr, scratch.data(), scratch.size(), captured->invocation.placeholders.data(),
        captured->invocation.placeholders.size()
    );
    // The ownership assertion starts exactly when the launch API returns:
    // destroy and reuse its writable source while capture is still active.
    ReadHostPointerSlots(scratch, &host_slots);
    PoisonReleaseAndReuse(&scratch);
    aclmdlRI model = nullptr;
    const aclError end_rc = aclmdlRICaptureEnd(stream, &model);
    const auto end = std::chrono::steady_clock::now();
    if (!CheckAcl(launch_rc, "aclrtLaunchKernelWithHostArgs", "captured probe launch") ||
        !CheckAcl(end_rc, "aclmdlRICaptureEnd", "end probe capture")) {
        if (model != nullptr) (void)aclmdlRIDestroy(model);
        return false;
    }
    if (host_args_probe_checksum(captured->invocation.canonical.data(), captured->invocation.canonical.size()) !=
        captured->invocation.canonical_hash) {
        std::fprintf(stderr, "FAIL canonical image changed during capture id=%" PRIu64 "\n", invocation_id);
        (void)aclmdlRIDestroy(model);
        return false;
    }
    captured->model.Adopt(model);
    std::printf(
        "CAPTURE id=%" PRIu64 " size=%zu us=%lld host_patch=[%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64 "]\n",
        invocation_id, args_size,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()),
        host_slots[0], host_slots[1], host_slots[2]
    );
    return true;
}

bool ReplayInvocation(CapturedInvocation *captured, aclrtStream stream, size_t replay_index, const char *name) {
    if (!captured->result.ClearAsync(stream)) return false;
    const auto begin = std::chrono::steady_clock::now();
    if (!PROBE_ACL(aclmdlRIExecuteAsync(captured->model.get(), stream), "execute captured probe")) return false;
    if (!PROBE_ACL(aclrtSynchronizeStream(stream), "synchronize captured probe")) return false;
    const auto end = std::chrono::steady_clock::now();
    char stage[64]{};
    std::snprintf(stage, sizeof(stage), "%s replay %zu", name, replay_index);
    if (!ValidateResult(captured->invocation, captured->result, stage)) return false;
    std::printf(
        "TIME %s replay=%zu us=%lld\n", name, replay_index,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())
    );
    return true;
}

bool RunAllocatorPressure(
    aclrtFuncHandle run_handle, aclrtStream stream, size_t count, size_t args_size, uint64_t first_id
) {
    if (count == 0) return true;
    DeviceBuffer result;
    if (!result.Allocate(sizeof(HostArgsProbeResult)) || !result.ClearAsync(stream)) return false;
    Invocation last;
    for (size_t index = 0; index < count; ++index) {
        Invocation invocation;
        if (!BuildInvocation(args_size, first_id + index, reinterpret_cast<uint64_t>(result.get()), &invocation)) {
            return false;
        }
        std::array<uint64_t, HOST_ARGS_PROBE_PAYLOAD_COUNT> host_slots{};
        const aclError rc = EnqueueInvocation(run_handle, stream, &invocation, &host_slots);
        if (rc != ACL_SUCCESS) {
            std::fprintf(
                stderr, "FAIL allocator pressure after %zu successful launches: rc=%d\n", index, static_cast<int>(rc)
            );
            (void)aclrtSynchronizeStream(stream);
            return false;
        }
        if (index + 1U == count) last = std::move(invocation);
    }
    if (!PROBE_ACL(aclrtSynchronizeStream(stream), "synchronize allocator pressure")) return false;
    if (!ValidateResult(last, result, "allocator pressure tail")) return false;
    std::printf("PASS allocator pressure count=%zu size=%zu\n", count, args_size);
    return true;
}

bool ParseUnsigned(const std::string &text, size_t *value) {
    if (text.empty() || text[0] == '-') return false;
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    *value = static_cast<size_t>(parsed);
    return true;
}

bool ParseSizeList(const std::string &text, std::vector<size_t> *values) {
    values->clear();
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t comma = text.find(',', begin);
        const size_t end = comma == std::string::npos ? text.size() : comma;
        size_t value = 0;
        if (!ParseUnsigned(text.substr(begin, end - begin), &value)) return false;
        values->push_back(value);
        begin = end + 1U;
    }
    return !values->empty();
}

void PrintUsage(const char *program) {
    std::fprintf(
        stderr,
        "usage: %s --device=<id> [--eager-sizes=bytes,...] [--graph-sizes=a,b]\n"
        "          [--replays=count] [--pressure-count=count] [--pressure-size=bytes]\n"
        "env: SIMPLER_DISPATCHER_SO, HOST_ARGS_PROBE_AICPU_SO\n",
        program
    );
}

bool ParseOptions(int argc, char **argv, Options *options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") return false;
        const size_t equals = argument.find('=');
        if (equals == std::string::npos) return false;
        const std::string name = argument.substr(0, equals);
        const std::string value = argument.substr(equals + 1U);
        size_t parsed = 0;
        if (name == "--device") {
            if (!ParseUnsigned(value, &parsed) || parsed > static_cast<size_t>(std::numeric_limits<int>::max())) {
                return false;
            }
            options->device_id = static_cast<int>(parsed);
        } else if (name == "--eager-sizes") {
            if (!ParseSizeList(value, &options->eager_sizes)) return false;
        } else if (name == "--graph-sizes") {
            std::vector<size_t> sizes;
            if (!ParseSizeList(value, &sizes) || sizes.size() != options->graph_sizes.size()) return false;
            std::copy(sizes.begin(), sizes.end(), options->graph_sizes.begin());
        } else if (name == "--replays") {
            if (!ParseUnsigned(value, &options->replay_count)) return false;
        } else if (name == "--pressure-count") {
            if (!ParseUnsigned(value, &options->pressure_count)) return false;
        } else if (name == "--pressure-size") {
            if (!ParseUnsigned(value, &options->pressure_size)) return false;
        } else {
            return false;
        }
    }
    if (options->device_id < 0 || options->replay_count == 0) return false;
    const auto valid_size = [](size_t value) {
        return value >= 256 && value <= std::numeric_limits<uint32_t>::max();
    };
    if (!std::all_of(options->eager_sizes.begin(), options->eager_sizes.end(), valid_size) ||
        !std::all_of(options->graph_sizes.begin(), options->graph_sizes.end(), valid_size) ||
        !valid_size(options->pressure_size)) {
        return false;
    }
    return true;
}

int RunProbe(const Options &options, const char *dispatcher_path, const char *aicpu_path) {
    std::vector<char> dispatcher;
    std::vector<char> inner;
    if (!ReadFile(dispatcher_path, &dispatcher) || !ReadFile(aicpu_path, &inner)) return 2;

    AclSession session;
    if (!session.Initialize(options.device_id)) return 3;
    std::printf("DEVICE explicit_id=%d (no reset on teardown)\n", options.device_id);
    PrintMemory("after context create");
    if (BootstrapAicpu(options.device_id, session.stream(), dispatcher, inner) != 0) return 4;

    AicpuBinary binary;
    aclrtFuncHandle run_handle = nullptr;
    if (LoadAicpuRunHandle(inner, options.device_id, &binary, &run_handle) != 0) return 5;

    uint64_t invocation_id = 1;
    for (const size_t size : options.eager_sizes) {
        if (!RunEagerSize(run_handle, session.stream(), size, invocation_id++)) return 6;
    }

    PrintMemory("before capture");
    CapturedInvocation graph_a;
    CapturedInvocation graph_b;
    if (!CaptureInvocation(run_handle, session.stream(), options.graph_sizes[0], invocation_id++, &graph_a) ||
        !CaptureInvocation(run_handle, session.stream(), options.graph_sizes[1], invocation_id++, &graph_b)) {
        return 7;
    }
    PrintMemory("after capture");

    if (!RunAllocatorPressure(
            run_handle, session.stream(), options.pressure_count, options.pressure_size, invocation_id
        )) {
        return 8;
    }
    invocation_id += options.pressure_count;
    PrintMemory("after args pressure");

    for (size_t replay = 0; replay < options.replay_count; ++replay) {
        if (!ReplayInvocation(&graph_a, session.stream(), replay, "graph-A") ||
            !ReplayInvocation(&graph_b, session.stream(), replay, "graph-B")) {
            return 9;
        }
    }
    PrintMemory("after alternating replay");

    if (!graph_b.model.Reset() || !graph_a.model.Reset()) return 10;
    if (!PROBE_ACL(aclrtSynchronizeStream(session.stream()), "synchronize graph destruction")) return 10;
    PrintMemory("after graph destroy");
    graph_b.result.Reset();
    graph_a.result.Reset();
    PrintMemory("after result release");

    std::printf(
        "HOST_ARGS_PROBE PASS device=%d eager_sizes=%zu graph_sizes=%zu,%zu replays_each=%zu pressure=%zu@%zu\n",
        options.device_id, options.eager_sizes.size(), options.graph_sizes[0], options.graph_sizes[1],
        options.replay_count, options.pressure_count, options.pressure_size
    );
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    Options options;
    if (!ParseOptions(argc, argv, &options)) {
        PrintUsage(argv[0]);
        return 2;
    }
    const char *dispatcher_path = std::getenv("SIMPLER_DISPATCHER_SO");
    const char *aicpu_path = std::getenv("HOST_ARGS_PROBE_AICPU_SO");
    if (dispatcher_path == nullptr || aicpu_path == nullptr) {
        PrintUsage(argv[0]);
        return 2;
    }
    try {
        return RunProbe(options, dispatcher_path, aicpu_path);
    } catch (const std::bad_alloc &) {
        std::fprintf(stderr, "FAIL host allocation while constructing a requested args image\n");
        return 11;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL unexpected exception: %s\n", error.what());
        return 12;
    }
}

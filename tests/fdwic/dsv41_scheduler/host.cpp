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
#include "common/scheduler.h"
#include "common/validate.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

#ifdef DSV_ONBOARD
#include <acl/acl.h>
#include <runtime/rt.h>
#endif

namespace {
struct CpuOps {
    static uint64_t now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    static uint64_t load(uint64_t *p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
    static uint64_t now_after(uint64_t) { return now(); }
    static uint64_t store(uint64_t *p, uint64_t v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
    static uint64_t add(uint64_t *p, uint64_t v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
    static uint64_t cas(uint64_t *p, uint64_t expected, uint64_t desired) {
        __atomic_compare_exchange_n(p, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        return expected;
    }
    static void save_sample(dsv41::Sample *p, const dsv41::Sample &v) { *p = v; }
    static void save_worker(dsv41::Worker *p, const dsv41::Worker &v) { *p = v; }
    static void flush_events(dsv41::Event *, uint64_t) {}
    static void compute(dsv41::Image *image, uint32_t worker) {
        for (uint32_t repeat = 0; repeat < image->repeats; ++repeat) {
            auto &tile = image->tiles[worker];
            for (uint32_t i = 0; i < 16; ++i)
                for (uint32_t j = 0; j < 16; ++j) {
                    float value = 0;
                    if (worker % 3 == 0) {
                        for (uint32_t k = 0; k < 16; ++k)
                            value += tile[0][i * 16 + k] * tile[1][k * 16 + j];
                    } else value = tile[0][i * 16 + j] + tile[1][i * 16 + j];
                    tile[2][i * 16 + j] = value;
                }
        }
    }
};

void run_cpu(dsv41::Image *image) {
    std::vector<std::thread> threads;
    for (uint32_t id = 0; id < image->clusters * 3; ++id)
        threads.emplace_back([=] {
            dsv41::dispatch<CpuOps>(image, id, id);
        });
    for (auto &thread : threads)
        thread.join();
}

#ifdef DSV_ONBOARD
void check(int rc, const char *where) {
    if (rc != 0) throw std::runtime_error(std::string(where) + ": " + std::to_string(rc));
}
class Device {
public:
    Device(uint32_t device, const std::string &binary) :
        device_(device) {
        std::ifstream input(binary, std::ios::binary);
        dsv41::require(input.good(), "cannot read A5 kernel ELF");
        data_ = std::vector<char>(std::istreambuf_iterator<char>(input), {});
        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(device), "aclrtSetDevice");
        const char *soc = aclrtGetSocName();
        dsv41::require(soc && std::string(soc).find("Ascend950") == 0, "this executable requires A5/Ascend950");
        check(aclrtCreateStream(&stream_), "create stream");
        rtDevBinary_t dev_binary{RT_DEV_BINARY_MAGIC_ELF, 0, data_.data(), data_.size()};
        check(rtRegisterAllKernel(&dev_binary, &kernel_), "register kernel");
        check(aclrtMalloc(&memory_, sizeof(dsv41::Image), ACL_MEM_MALLOC_HUGE_FIRST), "allocate image");
    }
    ~Device() {
        // A failed synchronize does not prove that the device stopped using GM.
        // The runner exits without reclaiming uncertain allocations in that case.
        if (in_flight_) return;
        if (memory_) aclrtFree(memory_);
        if (kernel_) rtDevBinaryUnRegister(kernel_);
        if (stream_) aclrtDestroyStream(stream_);
        aclrtResetDevice(device_);
        aclFinalize();
    }
    void run(dsv41::Image *image) {
        check(aclrtMemcpy(memory_, sizeof(*image), image, sizeof(*image), ACL_MEMCPY_HOST_TO_DEVICE), "upload image");
        void *parameters[] = {memory_};
        rtArgsEx_t args{};
        args.args = parameters;
        args.argsSize = sizeof(parameters);
        rtTaskCfgInfo_t config{};
        in_flight_ = true;
        check(rtKernelLaunchWithHandleV2(kernel_, 0, image->clusters, &args, nullptr, stream_, &config), "launch");
        check(aclrtSynchronizeStreamWithTimeout(stream_, 15000), "synchronize");
        in_flight_ = false;
        check(aclrtMemcpy(image, sizeof(*image), memory_, sizeof(*image), ACL_MEMCPY_DEVICE_TO_HOST), "download image");
    }

private:
    uint32_t device_;
    aclrtStream stream_{nullptr};
    void *kernel_{nullptr};
    void *memory_{nullptr};
    bool in_flight_{false};
    std::vector<char> data_;
};
#endif

struct Options {
    std::string flavor{"both"}, selected{"all"}, output{"outputs"}, binary, workloads;
    uint32_t clusters{8}, rounds{1}, repeats{1}, device{0}, profiling{3}, warmup{0};
    uint64_t delay{20000};
    bool empty{false}, self_test{false}, list{false};
};
uint32_t positive(const std::string &s, uint32_t max) {
    size_t end = 0;
    auto value = std::stoul(s, &end);
    if (end != s.size() || value == 0 || value > max) throw std::runtime_error("invalid positive argument: " + s);
    return value;
}
Options parse(int argc, char **argv) {
    Options o;
    o.workloads = (std::filesystem::absolute(argv[0]).parent_path().parent_path().parent_path() / "workloads").string();
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--empty") o.empty = true;
        else if (key == "--self-test") o.self_test = true;
        else if (key == "--list") o.list = true;
        else {
            if (++i == argc) throw std::runtime_error("missing value: " + key);
            std::string value = argv[i];
            if (key == "--case") o.selected = value;
            else if (key == "--flavor") o.flavor = value;
            else if (key == "--output") o.output = value;
            else if (key == "--binary") o.binary = value;
            else if (key == "--workloads") o.workloads = value;
            else if (key == "--clusters") o.clusters = positive(value, 32);
            else if (key == "--profiling") {
                if (value != "0" && value != "1" && value != "3")
                    throw std::runtime_error("profiling must be 0, 1 or 3");
                o.profiling = std::stoul(value);
            } else if (key == "--warmup") o.warmup = positive(value, 100);
            else if (key == "--rounds") o.rounds = positive(value, 1000);
            else if (key == "--repeats") o.repeats = positive(value, 10000);
            else if (key == "--device") o.device = positive(std::to_string(std::stoul(value) + 1), 64) - 1;
            else if (key == "--defer-us") o.delay = positive(value, 1000000) * 1000ULL;
            else throw std::runtime_error("unknown option: " + key);
        }
    }
    dsv41::require(o.flavor == "both" || o.flavor == "pypto_lib" || o.flavor == "pypto_lib_dsl", "unknown flavor");
    return o;
}

void self_test(const Options &options) {
    auto graph = dsv41::make_workload(options.workloads + "/pypto_lib_dsl/moe", 2, 1, 20000, false);
    run_cpu(graph.image.get());
    dsv41::validate(graph);
    auto &sample = graph.image->samples[0];
    ++sample.value;
    bool rejected = false;
    try {
        dsv41::validate(graph);
    } catch (const std::exception &) {
        rejected = true;
    }
    dsv41::require(rejected, "validator accepted corrupt payload");
    --sample.value;
    for (uint32_t id = 0; id < graph.image->task_count; ++id) {
        if (!graph.image->tasks[id].defer_ns) continue;
        graph.image->controls[id].completed_ns = graph.image->controls[id].due - 1;
        rejected = false;
        try {
            dsv41::validate(graph);
        } catch (const std::exception &) {
            rejected = true;
        }
        dsv41::require(rejected, "validator accepted premature async completion");
        break;
    }
    std::cout << "PASS: corrupt payload and premature async completion rejected\n";
}
}  // namespace

int main(int argc, char **argv) {
    try {
        Options o = parse(argc, argv);
        if (o.list) {
            for (const auto &name : dsv41::kCases)
                std::cout << name << '\n';
            return 0;
        }
        if (o.self_test) {
            self_test(o);
            return 0;
        }
        std::filesystem::create_directories(o.output);
#ifdef DSV_ONBOARD
        Device device(o.device, o.binary);
        const std::string backend = "a5";
#else
        const std::string backend = "cpu";
#endif
        for (const std::string flavor : {"pypto_lib", "pypto_lib_dsl"}) {
            if (o.flavor != "both" && o.flavor != flavor) continue;
            for (const auto &name : dsv41::kCases) {
                if (o.selected != "all" && name != o.selected) continue;
                for (uint32_t round = 0; round < o.rounds + o.warmup; ++round) {
                    auto graph = dsv41::make_workload(
                        o.workloads + "/" + flavor + "/" + name, o.clusters, o.repeats, o.delay, o.empty
                    );
                    graph.image->profiling = o.profiling;
#ifdef DSV_ONBOARD
                    device.run(graph.image.get());
#else
                    run_cpu(graph.image.get());
#endif
                    dsv41::validate(graph);
                    if (round < o.warmup) continue;
                    auto prefix = o.output + "/" + flavor + "-" + name + "-" + std::to_string(round - o.warmup);
                    dsv41::write_results(graph, prefix, backend, flavor, name, round - o.warmup);
                    std::cout << "PASS " << backend << ' ' << flavor << ' ' << name << " round=" << round
                              << " tasks=" << graph.image->task_count << " executions=" << graph.image->execution_count
                              << '\n';
                }
            }
        }
        dsv41::require(
            o.selected == "all" ||
                std::find(dsv41::kCases.begin(), dsv41::kCases.end(), o.selected) != dsv41::kCases.end(),
            "unknown case"
        );
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

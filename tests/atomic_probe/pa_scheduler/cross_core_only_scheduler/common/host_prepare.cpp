#include "only_scheduler_host.h"
#define PA_DEVICE inline
#define PA_DEVICE_NOINLINE static __attribute__((noinline))
#define PA_GM
#include "pa_scheduler_core.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <memory>

namespace {
#if PA_BUILD_PERF_CLOCK
thread_local uint32_t g_perf_clock_read_count = 0;
#endif
struct PrepareOps {
    // CPU 后端只验证调度协议与 raw schema，没有建立与 A5 CCEC
    // 同构的“atomic 返回值依赖 + SYS_CNT”硬件边界；因此必须标记为
    // source_issue，不能让 x86 built-in 的函数返回冒充 A5 return_ready。
    static constexpr bool kAtomicReturnReadyObserved = false;

    // 用 fetch_add(0) 模拟 A5 atomicAdd(addr, 0) 原子读，而不是退化为普通
    // CPU load。Acquire/AcqRel 只建立本 CPU 协议回归需要的发布/观察关系，
    // 不模拟 A5 cache 或设备内存模型细节。
    static inline int32_t Load(volatile int32_t *address) {
        return __atomic_fetch_add(address, static_cast<int32_t>(0), __ATOMIC_ACQUIRE);
    }

    static inline int64_t Load(volatile int64_t *address) {
        // Host 构建也保留原子 add-zero，16 个 pthread 共享同一协议。
        return __atomic_fetch_add(address, static_cast<int64_t>(0), __ATOMIC_ACQUIRE);
    }

    static inline uint64_t Load(volatile uint64_t *address) {
        return __atomic_fetch_add(address, static_cast<uint64_t>(0), __ATOMIC_ACQUIRE);
    }

    static inline int32_t Exchange(volatile int32_t *address, int32_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static inline int64_t Exchange(volatile int64_t *address, int64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static inline uint64_t Exchange(volatile uint64_t *address, uint64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static inline int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        // 与 production atomic wrapper 保持一致：返回线性化点观察到的
        // 旧值，而不是 bool。失败时目标字保持原样，调用方据此保留现场。
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(address, &observed, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        return observed;
    }

    static inline int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static inline int64_t FetchMax(volatile int64_t *address, int64_t value, uint64_t &retries) {
        // CPU 没有直接对应本测试签名的 fetch-max，用 CAS loop 实现同一返回值
        // 语义；retries 仅用于诊断软件竞争，不能与 A5 硬件 AtomicMax 对比。
        int64_t current = __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        while (value > current) {
            if (__atomic_compare_exchange_n(address, &current, value, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                break;
            }
            ++retries;
        }
        return current;
    }

    // 前后的 Exchange 已使用 AcqRel，Publish 使用 Release，因此这里不重复
    // 插入 fence，保持与设备适配层相同的调用边界。
    static inline void StoreBarrier() {}

    // 将 steady_clock 统一换算成纳秒，数值上适配公共模型的 1 GHz tick 标度；
    // 这不表示 CPU 物理时钟为 1 GHz，也不保证实际分辨率达到 1 ns。
    static inline uint64_t Now() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count()
        );
    }

#if PA_BUILD_PERF_CLOCK
    static inline void ResetPerfClockReadCount() { g_perf_clock_read_count = 0; }

    static inline uint32_t PerfClockReadCount() { return g_perf_clock_read_count; }

    static inline uint64_t PerfClockNow() {
        ++g_perf_clock_read_count;
        return Now();
    }
#endif

    template <typename T>
    static inline uint64_t NowAfterAtomicResult(T value) {
        // 空 asm 让编译器保留返回值到计时点的数据依赖，不额外插入 CPU fence。
        asm volatile("" : "+r"(value));
        return Now();
    }

    static inline bool ExecuteBoundKernel(
        pa_scheduler::SchedulerState *, pa_scheduler::WorkerState &, pa_scheduler::cross_core::ExecutionToken &,
        pa_scheduler::TaskKind, uint32_t
    ) {
        return true;
    }
    static inline void
    ExecuteKernel(pa_scheduler::SchedulerState *, pa_scheduler::WorkerState &, pa_scheduler::TaskKind, uint32_t) {}
    static inline bool PmuWindowStart(pa_scheduler::SchedulerState *, uint32_t) { return false; }

    static inline void PmuWindowStop(pa_scheduler::SchedulerState *, uint32_t, bool) {}

    static inline void SpinHint() {}

    static inline void PreloadDataCache(void *) {
        // CPU 只回归协议；不模拟 A5 DCache preload hint。
    }

    static inline void StorePayloadWord(volatile uint64_t *address, uint64_t value) { *address = value; }

    static inline void StoreTokenPayloadWord(volatile uint64_t *address, uint64_t value) { *address = value; }

    static inline uint64_t LoadPayloadWord(const volatile uint64_t *address) { return *address; }

    static inline void PreloadBuildDestination(void *, uint64_t) {}

    static inline void PreloadPayloadSource(const void *, uint64_t) {}

    static inline void PreloadTokenDestination(void *, uint64_t) {}

    static inline void BeforeBuiltPublish(uint32_t) {}

    static inline void BeforePayloadAcquire(uint32_t) {}

    static inline void InvalidateRegion(const void *, uint64_t) {
        // CPU 没有 A5 DCache line 失效指令；这里仅提供保守的本线程顺序边界，
        // 接口占位但不模拟设备 cache line 行为。共享状态本身仍使用 atomic。
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static inline void FlushRegion(void *, uint64_t) { std::atomic_thread_fence(std::memory_order_seq_cst); }

    static inline void Publish(uint64_t *address, uint64_t value) {
        // Release store 对应设备端 bypass-DCache 结果发布的可见性边界。
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }
};

}  // namespace
namespace pa_scheduler::host {
bool PrepareExecutionImage(SchedulerState *state, const Options &options) {
    // The ordinary callback constructs descriptors and fanin on CPU. Its
    // executor only advances construction lifetimes; it performs no arithmetic.
    const RunConfig saved_config = state->config;
    void *scratch = nullptr;
    if (options.trace_enabled) {
        scratch = std::aligned_alloc(64, kTraceBytes);
        if (!scratch) return false;
        InitializeTraceHeader(static_cast<TraceHeader *>(scratch));
    }
    ConfigureTrace(state, options, scratch);
    std::vector<std::thread> builders;
    for (uint32_t w = 0; w < kWorkers; ++w) {
        builders.emplace_back([=]() {
            RunScheduler<PrepareOps>(state, w, w < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv);
        });
    }
    for (auto &thread : builders)
        thread.join();
    std::free(scratch);
    state->config = saved_config;
    if (state->fatal.value != 0 || state->exec_fatal.state != 0) {
        std::fprintf(
            stderr, "Host execution-image preparation failed: fatal=%d exec=%lld\n", state->fatal.value,
            static_cast<long long>(state->exec_fatal.state)
        );
        return false;
    }
    const uint32_t count = state->build_dispatch.task_count;
    if (count != state->config.batches * 5) return false;
    for (uint32_t t = 0; t < count; ++t) {
        auto &cell = state->exec_cells[t];
        if (t % 5 == 0) {
            if (state->tasks[t].flag != 1 || cell.control.state != 0) return false;
            continue;
        }
        const auto old = cross_core::DecodeExecState(cell.control.state);
        if (!old.valid || old.phase != cross_core::ExecPhase::Done || old.task_id != t) return false;
        const auto header = cross_core::DecodeExecPayloadHeader(cell.payload);
        // H2D must not carry a host descriptor/function pointer. Ordinary uses
        // inline descriptors and numeric function IDs; reject a changed ABI.
        if (header.tensor_reference_mask != 0 || header.function_address != 0 || header.function_id != (t % 5) - 1)
            return false;
        cell.control.state = static_cast<int64_t>(cross_core::EncodeExecState(
            cross_core::ExecPhase::Built, cross_core::kExecMaxOwner, cross_core::kExecUnboundOwner, old.engine_class,
            old.payload_lines, t
        ));
        state->tasks[t].flag = 0;
        state->tasks[t].vend = 0;
    }
    state->started_count.value = 0;
    state->replay_done.value = 0;
    state->frontier.value = -1;
    state->build_dispatch.next_task.value = 0;
    state->exec_dispatch.aic_next.value = 0;
    state->exec_dispatch.aiv_next.value = 0;
    std::memset(&state->exec_drain, 0, sizeof(state->exec_drain));
    std::memset(state->results, 0, sizeof(state->results));
    for (auto &tokens : state->exec_tokens)
        for (auto &token : tokens)
            cross_core::ResetExecutionToken(token);
    std::printf(
        "[PREPARED] host-only tasks=%u alloc=%u executable=%u; device Build disabled\n", count, state->config.batches,
        state->build_dispatch.executable_task_count
    );
    return true;
}
}  // namespace pa_scheduler::host

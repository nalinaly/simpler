# A5 HBG AICore 原生 Graph 的物化策略

**状态**：提案，未实现，**未实测**
**基线**：`feat/fully_distributed_within_core` @ `47c55ec57`（a5 AICore 原生 Graph 路径）
**对照**：`main` @ `bad268384`（a2a3 / a5 AICPU legacy Graph 路径）

**复核（2026-11-02）**：已对照当前 `f0722c567` 核实。
P1/P2 的启动串行段成立；P3 的 legacy 对照、方案 A 的适用范围、
缓存可见性及失败语义论证已修正。方案仍未实现、未做本次性能测量。
逐项证据见[复核记录](../../../docs/zh-cn/investigations/2026-11-a5-graph-materialization-review.md)。

**已选定的后续路线**：先做保留启动屏障的多核并行构造，
再有条件评估验证/生成分离及按实例发布。正式协议、实施顺序与验收见
[物化优化方案](../../../docs/zh-cn/a5-hbg-aicore-graph-materialization-plan.md)。
原方案 A/B 保留为后续候选，不再作为第一步直接替换运行时。

本文最初提出把 a5 AICore 原生 Graph 的**全量前置物化 + 全核屏障**改为**增量物化**，
参考同一仓库内 AICPU 路径已服役的切片式实现；下文保留其问题分析和修订后的适用边界。

本文的论据是代码阅读和机制对比，**不含性能测量**。第 5 节给出取数方案；
在拿到数字之前，本文不主张任何具体的收益幅度。

## 1. 两条路径的物化策略

a5 同时存在两条 Graph 执行路径，由 bind 期的能力检查切换
（`src/a5/runtime/host_build_graph/host/runtime_maker.cpp`，`graph_binding->supported()`
为假时 `select_legacy_scheduler(..., SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH)`）。
两条路径对"把 Definition 展开成可执行的子任务存储"采取了相反的策略。

| 维度 | AICPU（a2a3 / a5 legacy） | AICore 原生（本提案） |
| ---- | ------------------------- | --------------------- |
| 物化粒度 | 4 个子任务一片（`GRAPH_MATERIALIZE_SLICE_TASKS`） | 整个 instance 一次到底 |
| 触发时机 | 调度主循环内，与普通任务 dispatch 交织 | bootstrap 阶段，执行开始前 |
| 未完成时 | 返回 `PENDING`，重回 `graph_prepare_queue` | 有物化生命周期状态，但不发布可执行的局部前缀 |
| 首个子任务可执行时刻 | 该片物化并发布后 | **全 run 所有 instance 物化完之后** |
| 并行度 | 可在不同调度线程服务各实例；每实例单 owner | 不超过 `min(instance 数, worker 数)` |
| 物化完成屏障 | 无全图物化完成屏障；仍有启动初始化屏障 | run 级物化完成屏障 + 每核 DCache 刷新 |
| 收尾聚合 | wake list 释放依赖，另有每 child 的 `remaining_tasks` / `retired_tasks` 原子操作 | wake list 释放依赖，另有每 child 的 `remaining_children` 原子递减 |

### 1.1 AICPU 路径：切片物化，边物化边发布

`runtime/scheduler/scheduler.h` 的 `prepare_graph_task` → `graph_incremental_publish`：
物化一片立刻发布该片 —— root 进 ready 队列，非 root 注册到第一个未满足的 producer 的
wake list，early-dispatch 候选进 ED 队列；没物化完返回 `PENDING`，由
`scheduler_dispatch.cpp` 重新 push 回 `graph_prepare_queue`，下一轮继续。

`graph_incremental_publish` 的函数头注释把设计意图写死了：

> `register_graph_wake` and `graph_first_unmet_producer` are safe against a
> producer completing concurrently, **which is what lets a task dispatch before
> the whole graph is materialized**.

支撑这个并发的有三个机制。AICore 增量方案必须建立等价的不丢唤醒、
不重复发布和单 owner 保证；按整实例发布不等于必须照搬切片游标：

- **`route_cursor` 的 CAS 游标**（`graph_route_ready_roots`）让"物化进度推进"和
  "外部依赖门打开"两个事件可以任意先后、各自幂等地触发 root 入队，每个 root 恰好入队一次。
- **`graph_first_unmet_producer`** 对并发完成的 producer 安全：注册 wake 时 producer
  可能已经完成，重查一次即可，不需要锁住整图。
- **prepare-queue 与 `materialize_busy` CAS**：排队服务之外，
  `graph_execution_materialize_slice` 还显式用 CAS 排除并发物化，切片区间不重叠。

### 1.2 AICore 原生路径：全量前置 + 全核屏障

`aicore/aicore_executor.cpp` 的 `bootstrap_graph_instances`：

1. 所有 worker 抢 `build->next_instance` 的 `fetch_add`，领到 index 就 CAS
   `materialize_owner` 独占；
2. `scheduler_graph_materialize`（`runtime/scheduler/scheduler_graph_materialize.h`）
   把该 instance 的全部子任务一次物化完；
3. 按 64B 步进 `dcci` 写回 + `dsb`，`remaining_instances` 递减，最后一个贡献者
   `st_dev` 发 `materialized` 信号；
4. **所有核等待物化成功发布**：先用 bypass load 探测 `materialized`，
   看到成功信号后才用原子读确认 `remaining_instances == 0`；随后
   `dcci(ENTIRE_DATA_CACHE)`，然后才进 `bootstrap_ready_graph`。
   不是所有等待者始终对完成计数做 RMW 热轮询。

没有切片，没有 `PENDING`，没有与执行的重叠。

## 2. 问题

**P1 — 单实例的前置物化仍然串行。** 实例级并行度上限为
`min(instance 数, worker 数)`。常见的
"一次调用一个 Graph 实例"场景退化成：1 个核做全部物化，其余核在 `local_backoff`
里空转。物化本身不便宜 —— `task_count = 1024`（`SUB_TASK_MAX_NUM`）时，仅 task storage
写回就是 `1024 × 320 / 64 ≈ 5120` 次 `dcci`，加上逐 tensor 96B 重绑和逐 scalar 继承查表，
全部串行在一个核上。

**P2 — 屏障挡在所有执行之前。** 即使有多个 instance，instance 0 中已满足依赖的 children
可能在本实例物化完后就具备执行条件，却要等到最后一个 instance 也物化完。整个 run
的物化时间全额计入首任务延迟。

**P3 — `remaining_children` 是潜在的收尾争用点，尚未测得瓶颈。**
N 个 child 完成会对同一个计数器做 N 次 GM atomicAdd。
该控制结构按 128B 对齐，不据此推断硬件原子的实际争用粒度。
原文“AICPU 没有这个计数器”不成立：
`graph_execution_complete_sub_task` 对 `remaining_tasks` 执行 `fetch_sub`，
其调用方还执行 `retired_tasks.fetch_add`。wake list 管理依赖释放，不代替完成聚合。
两条路径的原子指令、内存层级和参与线程不同，不能只数操作次数比较性能。

commit `9381db6b7` 处理过物化阶段的「**启动长尾**」，
但等待发布信号的长尾不能直接归因于实例内的串行物化。
既有[性能报告](../../../docs/zh-cn/a5-hbg-aicore-graph-p4-performance.md)
保留了失败样本、退避候选和隔离信号候选；最终既有稳定态统计，也单列每版三次首调用。
三次首调用不足以估计冷启动尾分布。它没有评估本提案的增量方案，
也没有把写入、写回、原子仲裁与等待成本完全分离。

## 3. 修订后的路线与候选

### 3.1 优先实施：并行构造，保留启动屏障

大单实例按 descriptor、Tensor pool、Scalar pool 的实际写回范围划分工作，
由多个核领取；每个物理写回单元只允许一个 writer，实例由最后一个成功贡献者唯一发布。
共享参数槽位归并，Scalar 尾部与相邻区域不能按 child 编号随意切分。
小图合并工作项，避免领取和完成计数开销压过构造收益。

第一阶段仍等全 run 成功物化才放行 kernel，保留物化失败时零执行和输出不变。
完成计数器保持现状，消费端全缓存观察也先保留，单独归因并行构造的收益。
工作项/区间元数据、RMW 和 Host 规划成本全部进入预算和完整时延对照。

### 3.2 方案 A：验证前置后的按实例延后物化（第二阶段候选）

先完成全 run 的可失败验证并固定本轮只读元数据，
再研究把剩余的有界生成工作挪进 **ENTER 任务的 body**。
若延后阶段还会发现原本应在启动时报告的错误，该方案仍不满足进入条件。

ENTER 已经是每实例一个的 inline 任务（`scheduler_ready.h` 的
`scheduler_complete_inline_task`），其 ready owner 可作为物化 owner。
这只解决所有权的一部分；还必须定义发布顺序、消费端可见性、
错误传播、超时和现有 Graph 协议记录，不能称为“不需要新增协议”。

预期作用及边界：

- 可以研究删除 run 级物化等待；ready/wake 注册屏障及 AICPU kernel 放行仍然存在；
- 多实例的物化与执行自然重叠；
- 单实例仍由一个核完整物化，不能解决 P1 的单实例并行度或分片首任务延迟；
- DCache 是核内状态，不能把消费核刷新直接替换成物化核刷新，必须另证可见性。

不必照搬四任务切片，但 AICore 同样在协作式循环里处理调度和执行。
长 ENTER 会推迟该核的其他调度、完成处理和错误观察，需测量服务时间上限。

更重要的是，当前故障测试要求某个实例物化失败时，整个 run 都不能进入 kernel。
延后到 ENTER 后，独立的健康实例可能先执行。这是可观察行为的变化，
落地前必须保留启动前验证，或者明确修改失败契约及其测试，不能只搬调用位置。

### 3.3 方案 B：消费端按需失效

可作为独立候选，但不能只在 `scheduler_prepare_dispatch_slot` 开头失效 child storage。
MIX 在进入该函数前已经解析 payload；scalar pool 和 Tensor 描述符池在 storage 之外。
Tensor 参数传给执行核的是描述符地址，实际执行核也是消费者。
必须覆盖普通/MIX 的最早解析点、分离的参数池、执行核读取，
并保留 producer 写回、发布顺序及其他启动数据的可见性保证。

精确维护可见性后再比较逐行维护与每 run 全缓存维护的成本。
现有代码不能证明全量前置的历史选择仅仅出于缓存失效开销。

### 3.4 P3 的配套

只有测得计数器争用是瓶颈后，再评估按 cluster 分片和两级归约。
任务分派不预先固定 cluster；新协议必须保证未 claim 的 child 不会被提前判为完成，
并覆盖 predicate 跳过、MIX 聚合和错误退出。不能预设分片一定更快。

## 4. 时序论证与未核验项

方案 A 至少需要证明：读取顺序、发布/缓存可见性、唯一 owner、失败与退出语义。
**child storage 的读者晚于 ENTER 完成是必要条件，不是充分条件。**

已核验：

- bootstrap 路由（`scheduler_bootstrap_route_task`）只读 host 建好的 node map
  fanin 和 metadata，不碰 child storage；
- `scheduler_classify_task_shape`（含 O(n²) 重复边检测）**只在 host 侧**调用
  （`host/scheduler_graph_binding.h`、`host/runtime_maker.cpp`），设备侧不走；
- 普通路径在 `scheduler_prepare_dispatch_slot` 中解析 payload；
  MIX 的 `scheduler_dispatch_mix_ready` 先调用 `scheduler_parse_task_payload_arguments`，
  然后才进入 prepare。两条路径都在 ready claim 之后；根 child 经 `ENTER_GATE`
  依赖 ENTER，非根通过内部 producer 传递依赖。
- 完成计数器两条路径都有；native 的贡献发生在 child 的 wake list 释放后。
- 现有物化故障测试明确断言 `executions` 为空、`resolved_task_count == 0`，
  并检查健康和失败实例的输出均保持原值。

**复核范围与未完成项**：已逐项检查上述三个 accessor、payload parser、
普通/MIX dispatch 和原始 storage 地址的当前源码调用点。
它们并未给出一个早于 ENTER 的正常 child 消费路径，
但这不等于证明每个执行核的缓存都新鲜，也不证明延后失败与旧契约等价。
尚未实现候选，未进行缓存非一致性模型验证或新方案上板验证。

## 5. 验证方案

两条路径在同一块 a5 上共存，可以比较完整后端差异，
但这不是仅改变物化策略的受控 A/B，不能直接量出增量物化的收益。
`get_last_scheduler_runtime_mode()` 是观测接口，不是强制选择接口。
不能靠改变 workload 使其 fallback，再把差异归因于物化。

1. **先量现状**：冻结 workload、kernel 二进制与配置，记录每实例物化/发布、
   每 worker 等待及首任务开始，区分首调用与稳定态。完整 native/legacy 对照的路径可由
   `get_last_scheduler_runtime_mode()`（本分支新增的 C API）确证，
   `get_scheduler_mode_publish_count()` 区分"本次 bind 重新选择"与"锁存着上一轮的值"。
2. **扫 instance 数**：单实例与多实例分别取点，记录实际参与构造的 worker 数，
   检验单实例并行与多实例负载分配。
3. **分别扫 `task_count` 和参数字节数**：测量构造、写回与领取/计数开销的变化，
   不预设实测时延一定线性增长。
4. 按 `.claude/rules/running-onboard.md`：先过 `onboard-arch-precheck`，
   经 `task-submit` 取设备锁，`ASCEND_PROCESS_LOG_PATH` 指向本次 run 的输出目录。
5. 候选实现后，在同一 native 基线上只改变物化策略，做交替 A/B 和同二进制重复对照，
   检查性能及数值。记录 `task-submit --list` 的并发占用，以识别而非保证排除环境漂移。
6. 保留物化失败前零 kernel 进入的验收；若拟改变，先明确行为契约。
   覆盖单/多实例、普通/MIX、跨实例依赖、重复 replay、动态 scalar/Tensor 重绑和失败回收。

按正式方案预先冻结的门槛决定是否纳入实现：目标场景首任务延迟可重复改善，
完整 RunWall、小图和无 Graph 对照没有稳定回退。已测得无收益或回退的候选
按 `docs/investigations/` 模板归档；数据不足以区分波动时记录为“不确定”，
不把尚未实现、尚未测量的方案称为“measured & dropped”。

## 6. 参考

- `src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp` — `bootstrap_graph_instances`
- `src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph_materialize.h`
- `src/a5/runtime/host_build_graph/runtime/scheduler/scheduler.h` — AICPU legacy 的
  `prepare_graph_task` / `graph_incremental_publish`
- `src/a2a3/runtime/host_build_graph/runtime/scheduler/scheduler.h` — 同上，a2a3 侧
- `src/common/host_build_graph/docs/GRAPH_EXECUTION.md` — Graph 特性的架构中立设计
- commit `9381db6b7` — 启动长尾的既有证据与缓解手段
- `.claude/rules/codestyle.md` §5 — 禁止在 AICPU 自旋中调用
  `std::this_thread::yield()` / `sched_yield()`；这不是禁止所有有界本地退避的规则
- `src/common/host_build_graph/graph_execution.h` — legacy 完成与退休计数
- `tests/st/a5/host_build_graph/graph_execution/test_graph_device_failure.py` — 启动物化失败契约

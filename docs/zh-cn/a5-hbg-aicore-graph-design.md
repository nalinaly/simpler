# A5 HBG AICore resident Graph 支持设计

日期：2026-10-30。状态：**实施中；多 block、Mix、sync-start 和单 block predicate 已通过 A5sim/A5 功能验证，完整故障矩阵与原生 L4/性能验收尚未完成**。进度和测试证据见[实现过程](a5-hbg-aicore-graph-implementation.md)。

开发基线：`02335c0f46d6f12a0fd0adaeccf4869b1164ff64`，来自 `feat/fully_distributed_within_core`；独立分支 `feat/a5-hbg-aicore-graph`，工作目录 `simpler-aicore-graph`。不包含其他工作目录的未跟踪文件，不以旧精简 PR 的版本代替当前基线。

本文描述目标内部设计，不新增公共 Graph API。各对象的已实现/已验证程度以过程记录为准；未完成的阶段不能从设计措辞推断为已经可用，字段布局由跨编译目标的断言约束。

## 1. 目标、现状与不能改变的边界

### 1.1 目标不是重新实现普通 DAG 调度

这里的 Graph 是 `rt_submit_graph` 提交的复合任务：Host 录制并复用一个 Definition；每次调用产生一个外层 `TaskKind::GRAPH`，设备实例化并执行其内部子 DAG。它不是 kernel 融合，也不是泛指普通 HBG DAG。

目标是在保留 Graph API、Definition 复用和父子依赖语义的前提下，让 **AICore Scalar 完成子任务实例化、ready 推进、领取、执行和完成聚合**。Host 仍负责录制、TensorMap 和内存规划；AICPU 保留拓扑发现、启动/退出及错误收敛，不承担正常 Graph 子任务调度。

开发基线的 [create_scheduler_state](../../src/a5/runtime/host_build_graph/host/runtime_maker.cpp) 发现任意 GRAPH 即整轮选择 `SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH`。实施中只按已验收子集逐步开放 native；既有 Graph 数值测试的 legacy 通过记录不代表 resident 支持。[现有公共合同](../../src/common/host_build_graph/docs/GRAPH_EXECUTION.md)继续有效。

### 1.2 经源码确认的起点

| 项目 | 基线现状 | 本工作需要解决 |
| ---- | -------- | -------------- |
| Definition | 真机随 AICPU launch arguments 携带；各线程只读自己的视图 | 给 Scalar 提供 run-owned、可共享的 GM 载体 |
| 子任务存储 | 在每个外层 Graph 的预分配 heap 尾部，不占顶层 task window | 复用存储规模/参数布局合同，建立 resident 访问视图 |
| 子任务身份 | 公共 TaskId 有 GLOBAL、SUB_TASK、PARAM，SUB_TASK 带父身份 | 不把 SUB_TASK 的 local_id 当作全局唯一索引 |
| resident 引用 | `task_ref` 是顶层表下标；ready 队列用一个 64 位 CAS 字承载两个 int32 端点 | 给内部节点分配可放入原队列的稠密引用 |
| 普通执行能力 | C/V 分波、GM Mix、sync_start、单 block 单 lane predicate、无 kernel 服务 | Graph 子任务复用这些能力，不另写第二套执行器 |
| 有效优化 | 票号领取、claim_seal、同轮 ACK 正向预检查后强确认、Ready 前缀/负探测复用 | 保留原所有权及公平竞争条件，不为 Graph 撤销 |
| 当前协议 | `hbg-native-protocol-v7`，Mix close seal stride 为 128B | 新增父子身份/控制事件需明确版本演进，不能冒用 v7 |

### 1.3 首版范围和不变量

- 支持单层 Graph、同一定义多实例、不同定义、多 Graph 与普通任务混排；保留原 tensor/view、scalar 转发和 Config 身份合同。
- 逐步覆盖纯 AIC/AIV、普通多 block、Mix masks 3/5/6/7、合法 sync_start、Dummy 与现有单 block 单 lane predicate。多 block/Mix predicate 不因放开 Graph 顺带开放。
- 沿用每 run 最多 16 个 Definition、每个定义最多 1024 个子任务等当前边界；不新增嵌套 Graph、跨 run Definition 缓存或 deferred completion 支持。
- 每次 Graph 仍只占一个公共顶层 task slot、只贡献一次公共顶层完成；不把内部节点重新调用 Host submit 塞进顶层 TensorMap。
- 保留所有原本合法核/cluster 的领取资格。不得固定小 N 任务的执行核、缩减 Mix 候选集合、常驻占用一个 AIV 代调度，或改变三 lane 资源合同。
- 不改计算 kernel、输入、block 数、数值容差或 task 模拟负载；不在 Scalar 动态分配 heap，不在 Scalar 重新推导 tensor 依赖。
- 无 Graph 的路径不分配 Graph 专属存储，不进入新增物化阶段；普通任务热路径必须有独立回归和性能对照。
- 不新增环境开关/编译宏。按实际图的已验证能力选择整轮 resident 或 legacy；不得执行到一半才回退，也不以“数值相同”掩盖回退。

## 2. 推荐结构：共享定义、稠密引用、分层完成

### 2.1 路线选择

| 路线 | 评价 |
| ---- | ---- |
| 仅删除 GRAPH 回退判断 | 不成立：寻址、参数绑定、内部依赖、父子完成均未接入 |
| Host 将所有子任务重新按普通任务提交 | 可作为实验对照，不作为正式实现；改变了 Graph 的提交与复用成本 |
| AICPU 持续展开/调度 Graph，AICore 只接收 leaf | 不满足本次目标；不能声称已完成 resident Graph |
| 共享 Definition + AICore 实例化 + 原 resident 执行体系 | 推荐；复用 Host 录制、存储规划和既有 ready/完成/协同协议 |

**首个可验收版本建议在 Graph run 的启动阶段完成全部实例物化，再开放 kernel 执行。** 这是明确的首版折中：不保留旧 AICPU 路径“边物化边执行”的重叠，不声称性能等价。物化属于设备工作，必须计入完整 Device 时间。

增量物化作为后续独立阶段，只有确认存在显著物化瓶颈并完成进度证明才接入。不能把首版启动屏障扩展成每个 Graph 执行前的全核屏障，更不能等前一个 Graph 全局结束才运行独立的另一个 Graph。

### 2.2 Definition 载体与复用

在 Host recording/commit 完成后，对 Definition 做能力和大小检查，**先确定整轮 backend，再发布其载体**：

- resident Graph run：把每个不同 Definition 恰好一份打包进现有 scheduler state 的 run-owned GM 分配，复用 `acquire_scheduler_state_storage` 的 retained host/device 存储对与发布流程；实例只持有 offset/length。
- legacy run：继续使用现有 AICPU launch-argument section，不改变 A2/A3 或 legacy 执行语义。
- 优先复用 Definition 的现有 POD 编码、framing、CSR 和参数来源信息；只增加 Scalar 可编译的有界读取视图。共享对象存 offset/index，不传 Host 指针或某个 AICPU 线程的参数包指针。
- framing 固定为 16B 对齐、32B header，在 Host/AICPU/CCEC 上共同断言；不得用目标相关的 `max_align_t` 推导跨端 wire 大小。此约束来自原生真机首测，保持既有 Host/AICPU 编码不变。
- 不为每个实例单独 malloc/upload Definition，不恢复逐实例完整复制、全图 hash 或反复校验。Host 做静态结构/范围校验；设备保留 framing、实际实例地址及关键控制状态检查。
- carrier 必须活到本轮设备访问完全静止；pipeline slot 复用、失败退出和旧 epoch 的读者一起纳入生命周期。新增 GM 拷贝开销不能藏到 Host/Device 指标之外。

旧 [single-upload 调查](../investigations/2026-08-hbg-graph-definition-single-upload.md)是历史证据，不是当前 launch-package 实现；复用它的“按定义共享、按 run 管理”原则，不重新引入已撤回的完整内容 hash。

### 2.3 公共身份不变，内部引用重新编号

推荐给 Graph-enabled run 建立紧凑的内部编号。普通顶层任务保留一个内部节点；每个 Graph 逻辑上分成 `ENTER`、原有子任务、`EXIT`。ENTER/EXIT 是**调度内部的无 kernel 服务节点**，不增加公共 task slot 或 kernel 次数。

```text
原始公共图： P -> G { a -> b } -> Q
内部执行图： P -> ENTER(G) -> a -> b -> EXIT(G) -> Q
                                      所有子任务终态汇总到 EXIT
```

编号按顶层提交顺序展开；一个 Graph 内为 ENTER、按录制序排列的子任务、EXIT。这样保留现有“正常 fanin 的 producer ref 小于 consumer ref”校验。数值大小只用于存储/拓扑合法性，不变成 kernel 执行顺序；ready 的独立任务仍按原策略竞争。

| 引用 | 含义与转换 |
| ---- | ---------- |
| 公共 GLOBAL TaskId | 不改原 ID；普通任务映射到自己，Graph 作为外部 producer 时映射到 EXIT |
| 公共 SUB_TASK TaskId | 仍由父 GLOBAL ID 与子任务 local_id 唯一确定；映射到该实例的连续子任务范围 |
| 内部 `task_ref` | 只索引 resident 元数据/控制/地址表，范围先验证能装进原 int32 队列端点 |
| ENTER 的 fanin | 由 Graph 外部 fanin 转换，引用前驱普通任务或前驱 Graph 的 EXIT |
| 子任务的 fanin | 非 root 由保存的 CSR 转为本实例引用；root 增加 ENTER gate，不等待父 Graph 的最终 DONE |
| EXIT 的就绪 | 由最后一个子任务终态触发，不作为零 fanin 的启动 root，不构造超长“全部子任务”fanin 数组 |

新增只读映射至少包含 node kind、public identity、实例编号、descriptor/payload 地址或 offset、fanin 来源。无 Graph run 继续用现有公式寻址；Graph run 中连同普通顶层节点一起经映射解析，不能在每个热路径散落重复分支。

Graph 子任务原 payload 的 `fanin_count` 为零，依赖保存在 CSR。统一读取器必须返回转换后的依赖数和引用，不能因 payload 没有内联 fanin 就把所有子任务当成 root。root 的 ENTER 边可由视图生成，不必复制或修改整个 Definition。

**内部 CSR 与顶层 fanin 的容量不同**：顶层内联 fanin 上限为 128，Graph 的单个子任务可依赖此前全部子任务，合法上限为 1023。访问器、Host 能力检查和 trace 导出都必须按节点类型处理；“EXIT 不存超长 fanin”并不能替代对内部宽 fanin 的支持。

必须同步修改 `scheduler_classify_task_shape` 中“descriptor.task_id 等于 task_ref”的假设：比较公共身份与映射声明，而不是把 descriptor 内的公共 TaskId 改成稠密序号。`TaskId::raw` 不直接塞进两个 int32 的队列端点；不放宽原有越界/重复/非法边校验。

### 2.4 实例状态与存储预算

每个实例有独立的边界参数、子任务存储、完成状态和输出 heap；共享 Definition，不共享任何可变完成/参数区。建议复用现有 heap-tail 规划和 tensor/scalar pool，不再复制一份完整任务表，也不维护每核全图状态。

拟议最小实例控制状态：物化 owner/终态、ENTER/EXIT 引用、子任务起始引用和数量、剩余子任务数、epoch。它不是另一套 TensorMap。先复用可兼容字段；不能直接把含 `std::atomic<uint8_t>` 和 AICPU 指针的 `GraphExecution` 原样交给 Scalar。

设普通顶层节点数为 O，Graph 实例数为 I，第 i 个实例含 N_i 个子任务，则内部节点预算 `R = O + Σ(N_i + 2)`。检查 int32 引用、字段窄化、CSR/trace 前缀、所有乘积和总分配溢出；不把 R 与公共顶层 task_count 混用。

| 存储 | 预算原则 |
| ---- | -------- |
| Definition GM | 不同定义的大小之和，不能乘实例数或 worker 数 |
| 子任务存储/参数池 | 使用每实例原有执行区，按实际 N_i 和参数量规划 |
| resident task controls | 以 `sizeof` 计算；基线为每节点 512B，若先统一分配约为 `512 × R` |
| Mix close seals | 基线每个 Mix task × candidate cluster × 128B，必须计入 Graph 内 Mix 数量 |
| 映射/实例控制 | O(R + I)，不做 O(worker_count × R) 的副本 |
| 依赖引用 | 若生成逐实例的稠密 fanin 数组，另计 `4 × 转换后边数`；优先共享 Definition CSR 并在读取时加实例基址，不能把边数组藏进 O(R) 预算 |
| trace | 依采集级别预算；L0 无逐操作诊断写入，保留容量/丢弃检测 |

例如 I=100、N_i=1024 时，仅统一 task controls 已约 50.1 MiB，不包含原有执行存储和 Mix seals。这个算例是容量估算，不是实测。禁止先按最大图给每个核分配完整状态。

## 3. 执行、完成、发布与故障协议

### 3.1 启动与物化

1. Host 完成录制、边界匹配、能力检查和地址/空间规划；初始化本轮控制区，发布 Definition、映射及实例描述。
2. AICPU 沿原路径完成硬件拓扑/启动，不执行 Graph prepare/dispatch 主循环。
3. 已启动的 Scalar worker 动态领取实例物化工作。首版一个实例一个临时 owner；不是固定 builder 核，进入执行阶段后所有合法核仍参与正常竞争。
4. owner 按 Definition 重绑定本次 tensor/scalar、内部 heap 偏移和 predicate 地址，初始化子任务 descriptor/payload。此为既定拓扑实例化，不是依赖推导；scratch 按单项/小块使用，不把整个 Graph 放入栈。
5. 所有实例均完成发布，且所有 worker 均确认读取视图就绪后，进入正常 resident bootstrap/ready 路径。无 Graph run 完全不经过此阶段。

物化 owner 只能等待有限的发布事务，不能等待 kernel 或外部 fanin；不持锁进入 kernel。任一失败立即走现有 first-error/退出收敛，等待启动完成的 worker 必须持续检查错误和生命周期状态，不能无限等失败 owner。

若后续把同一实例拆成并行 chunk，必须先证明 descriptor/参数区的写回粒度不交叠、边界元数据只读、发布者唯一；首版不默认这些性质已经成立。

### 3.2 ENTER、子任务和 EXIT

- ENTER 只有在外部依赖满足后才由唯一 ready owner 执行。它不占 kernel slot，不申请协同 order_key；完成该 gate 后按原 wake-list 释放 Graph roots。
- ENTER/EXIT 都是有界控制服务；不能因两个普通 payload 槽占满而永久停止服务，已承诺的完成/释放事务必须在进入后续阻塞 kernel 前收尾。
- 非 root 的内部依赖来自录制 CSR；沿原完成驱动的注册/关闭/重新检查协议推进，不让每个核轮询整份 fanin。
- 所有真实子任务复用当前普通/协同准入和 kernel entry gate。Graph 边界不能成为 ordinary direct refill、Mix 或 sync_start 的越序旁路。
- 子任务的逻辑终态只发生一次：普通/Mix 的原 N 个 block 完成，或者按已有 predicate/Dummy 语义正常退休。先发布输出/终态并完成内部依赖处理，再贡献一次实例 remaining 递减。
- 只有递减观察到旧值 1 的执行者可以触发 EXIT。重复完成必须在身份/所有权层拦截，不能靠计数“恰好到零”兜底；计数小于零明确失败。
- EXIT 只贡献父 Graph 一次公共完成、释放外部后继，不调用计算 kernel。Graph 的输出只有到这里才可被外部消费者使用，不能以某个 root/leaf 的返回代替全体子任务终态。
- 原先在 Host 已完成的 Alloc/服务保持既有待遇；图内 Dummy、predicate skip 不伪造 kernel 记录，但必须正确推进内部逻辑完成。

内部节点退休数、公共顶层完成数、物理 kernel 执行数分开统计。旧 `pending_completed/resolved_task_count` 的阈值不能沿用顶层数量去判断扩展图结束；计划用明确的内部 pending 总量判断调度收敛，公共完成只由普通顶层节点与 EXIT 贡献。已经 Host 完成的节点从内部待处理量和实例 remaining 中扣除，不重复记账。若合法实例没有待处理子任务，仍须由 ENTER 在外部依赖满足后唯一触发 EXIT，不能在启动时提前释放外部后继；不因此放宽原来拒绝的空 Definition。

错误路径的退休记账不代表成功 DONE。退出仍须满足：无新工作可发布、所有本核/组义务排空、trace/error 发布收尾和原生命周期握手完成；不能仅凭公共完成数到齐释放存储。

### 3.3 共享内存发布矩阵

| 数据 | writer / reader | 必须保持的合同 |
| ---- | --------------- | -------------- |
| Definition、映射、实例地址 | Host → Scalar | launch 前完整上传，本轮只读，run-owned，不读 AICPU 参数包 |
| 子任务 descriptor/payload/参数池 | 唯一物化 owner → 执行/调度 worker | 完整写回和 barrier 在物化完成信号之前；消费者 acquire 后失效相应缓存再读 |
| 物化 owner、remaining、唯一 EXIT 发布权 | 竞争者/完成者 → 同组读者 | 使用现有原子所有权原语；布局隔离，不被 cached 写回覆盖 |
| 物化完成信号 | 单 owner → 其他 worker | 可评估复用 bypass signal，但须满足整条缓存行不混 atomic/cached 写回的前提 |
| ready/wake、claim/ack/seal | 原调度器参与者 | 复用基线协议，包括 claim_seal、ACK 原子强确认与 128B close seal |

以上是需要实现和真机验证的发布链，不因使用 C++ acquire/release 就自动成立。首版优先复用已验证原语，不以“都是 GM”取消 DCCI/barrier，不把 `ld_dev` 预筛当成原子所有权证明。

每个并发 writer 的物化范围按实际 DCache 写回粒度隔离；尤其检查相邻实例 heap 尾部、scalar pool 边界和控制字，避免整行写回覆盖其他 writer 的字节。

### 3.4 进度、兼容边界和后续增量物化

Graph 本身不占一个执行核等待其子任务，也不整体持有协同 admission。普通子任务可分波，sync_start 仍是“一个子任务的 blocks”共同驻留，不能把多个 roots 或整个 Graph 合成一个 cohort。

首版在启动阶段完成物化，进入 kernel 后没有“尚未构造的必要同伴”，使现有共同准入的进度前提不被悄悄破坏。增量版本必须独立证明：必要的构造/发布/释放不会被后序阻塞 kernel 卡住；不能仅把控制 pass 移到主循环靠前的位置。[已有 Graph/drain 反例](../investigations/2026-09-hbg-graph-release-vs-drain-reorder.md)必须变成回归门禁。

同一个 Graph 的物化/记录错误、边界不匹配、越界、非法拓扑仍然是错误，不转成 legacy 成功。合法但尚未开放的组合在 launch 前整轮 legacy，清楚报告具体 Graph/子任务原因；strict resident 测试必须拒绝该回退。单 block predicate、Mix 等资格须扫描到 Definition 内部，不能只检查外层 GRAPH。

内部数据布局、读取器与导出器在同一变更中更新。A2/A3 与 legacy 的公共 Definition 语义不变；公共 API、Config/boundary 规则或通用 wire ABI 若确需改变，必须单独说明，不能借本次内部索引适配顺带改约定。

## 4. 分阶段实现和验收

### 4.1 开发顺序

| 阶段 | 交付 | 通过门槛 |
| ---- | ---- | -------- |
| P0：固定合同 | 复用现有 Graph fixture，增加严格 resident 红灯；CPU 小模型覆盖 ENTER/EXIT 映射 | 旧代码数值可过但来源必失败；模型证明父子完成及无额外 Graph 间屏障 |
| P1：数据和访问器 | GM Definition 载体、稠密映射、checked layout、公共身份分离 | C++/Python 布局、映射、非法输入、容量测试；不先移除生产回退 |
| P2：最小 native Graph | Graph run 启动物化、纯 C/V 子任务、外部 gate 和 EXIT | A5sim/A5 原样数值、多个实例不同地址/scalar、混排依赖、fresh resident 来源 |
| P3：能力对齐 | 原 N 分波、GM Mix、Dummy、单 block predicate、sync_start | masks/容量两侧/慢核/跨 Graph 重叠/失败矩阵及旧 resident 全部相关回归 |
| P4：完整观测与端到端 | 父子身份、原生协议因果证明、泳道导出和内存/时间报告 | 严格 L4，无缺失/重复/截断；无插桩结果与 legacy 同业务对照 |
| P5：有证据再优化 | 物化分块、缓存观察或增量执行等独立方案 | 不改变负载和公平性，完整 Host+Device 数据支持，回退或无收益不并入功能主线 |

P2 只能开放已验收的子集，其余继续整轮 legacy；P3/P4 完成前不宣称完整 Graph 支持。每阶段小步提交，是否 push 由用户另行授权。用户已要求实施本设计，实际证据在实现过程记录。

### 4.2 必须覆盖的正反例

| 类别 | 最小测试与断言 |
| ---- | -------------- |
| 定义复用 | 同一定义 2/多次调用，改变合法边界地址与 scalar；一个 Definition、多个独立 execution，无串实例读写 |
| 身份与映射 | 两个 Graph 都有 local_id=0；全局/子任务/ENTER/EXIT 不混淆，dense ref 越界和公共身份错绑失败 |
| 外部依赖 | P→G、G→Q、G1→G2；roots 不能早读，Q 必须等全部子任务；保留真实 IN/OUT/INOUT/WAR/WAW 边 |
| 内部拓扑 | 链、菱形、多 root、多 terminal、1024 子任务；内部单任务 129/1023 fanin，顶层仍拒绝超过 128；EXIT 不构造全部子任务 fanin；恰好一次完成 |
| 内存边界 | 参数 view/alias/内部输出/边界 scalar/predicate 重绑定；无 tensor 边界、相邻实例、畸形 offset/大小、分配溢出 |
| 原 N 与协同 | N=1、容量以下/等于/以上、512-block 分波；Mix 3/5/6/7，mask 6 无 AIC kernel，inactive lane 义务不丢 |
| sync_start | 同一实例多个协同 roots、不同 Graph 交叉参与者、早返回、满容量和超容量；不二次领取，不反序等待 |
| 局部推进 | G1 的慢远端仍在运行时，空闲成员可执行独立 G2；真实 G1→G2 时不得提前 |
| 服务/predicate | Dummy、Host inline 完成、predicate true/false/失败；skip 不伪造 kernel，unsupported 不误放行 |
| 发布与复用 | 物化 owner 暂停、信号先后交错、旧 epoch/旧组 token、重复 run/不同大小、pipeline slot 回收 |
| 故障 | 构造失败、子任务失败、超时、错误发生在最后贡献前后；首现场含父/子身份，不发布假成功、不提前 free |
| 无 Graph 回归 | 固定 CSA、PA、ordinary/Mix/sync_start/错误路径；没有新增 Graph 存储和每轮扫描开销 |

先复用 `tests/st/a5/host_build_graph/graph_execution/`、`graph_predicated_dispatch/`、`graph_sync_start_early_dispatch/` 以及 C++ `test_graph_activation.cpp` 的业务/反例。旧 AICPU C++ 测试不能冒充新 resident helper 验证；为新视图和状态机补专门测试，但不复制计算 kernel。

pytest 与 standalone 都需逐轮检查 `scheduler_mode_publish_count` 增加、mode 为 resident。仅修改 pytest marker 或只看数值不算接通。现有 `graph_execution` 的 legacy marker 只在对应子集通过后调整；保留明确的 unsupported fallback 测试。

### 4.3 泳道与协议证据

沿用 `simpler_setup/tools/hbg/` 和标准 converter。新 trace 明确区分公共父 TaskId、子任务 local_id、dense ref、block、逻辑 slot、物理 worker、epoch/generation；父子 flow 不用不同核的原始 cycle 大小推断。

- 增加物化起止、ENTER、子任务终态、EXIT 的有界事件，分别对照 Host 冻结 Definition/实例清单；不能从最终 kernel 数量反推构造协议正确。
- Graph 父区间展示生命周期；真实子任务仍落在对应 AIC/AIV 的 Scalar/kernel 轨道。ENTER/EXIT/Dummy 不产生伪 kernel span。
- Scalar 等待按未满足外部依赖、内部依赖、协同准入和 Mix 资源区分；连续 poll 合并为区间并保留次数，不逐轮膨胀原始记录。
- 当前 v7 的 schema 语义不改写。新增父子映射和控制事件时制定下一版协议，旧 v1-v7 捕获仍按原规则读取；没有 Graph 证据的旧 capture 不标为 Graph verified。
- 功能阶段使用现有有界 trace 和严格完整性检查；L0 不增加逐操作探针。保存唯一输出目录，泳道图/raw 默认不进入 Git。

### 4.4 性能与资源验收

比较同一编排/Definition/输入/kernel/原 N 的旧 AICPU Graph 与新 resident Graph。可另设普通展开形式解释开销，但不能用它替代同负载 Graph A/B，更不能把 PyPTO 重生成 kernel 的差异算作 scheduler 收益。

至少分别报告 Host bind（录制/pack/新增 GM 发布）、完整 Device RunWall（含物化/启动/执行/退出）、Host 调用总窗口、物化独占时间、kernel span、实际 GM 占用。冷启动与预热后分开，记录 A/B 交替批次、样本数、中位数/P90/P99和版本/二进制摘要。

无 Graph 的固定 CSA 在相同计时范围下独立对照，不把旧 2.084ms 或后续优化数字当成本分支已测结果。当前未设 Graph 的绝对性能目标，也未证明首版启动前物化比增量物化更快；先获得完整功能基线，再由数据决定优化。

## 5. 代码落点、评审项和接续工作

### 5.1 修改面与复用依据

| 位置 | 预计工作 |
| ---- | -------- |
| [Host runtime maker](../../src/a5/runtime/host_build_graph/host/runtime_maker.cpp) | preflight、resident/legacy carrier 选择、Graph layout/refs/计数/发布 |
| [Graph 公共 Definition](../../src/common/host_build_graph/graph_definition.h)、[执行合同](../../src/common/host_build_graph/graph_execution.h)、[公共实例化](../../src/common/host_build_graph/device/graph_execution.cpp) | 复用静态编码、字段/参数绑定语义；Scalar 的 [GM 读取器](../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph_definition.h)使用同一份 POD，不整体搬入 CPU scheduler |
| [task_id.h](../../src/common/host_build_graph/task_id.h)、[scheduler_graph.h](../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h) | 公共身份与内部索引分离、地址/fanin 访问器、严格验证 |
| [scheduler_types.h](../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h)、[scheduler_layout.h](../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_layout.h) | checked sizing、实例状态、引用表、启动阶段描述 |
| [AICore executor](../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp) | 有界物化/启动发布及原主循环接线，不另建集中式调度线程 |
| [ready](../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h)、[completion](../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h) | ENTER/EXIT 服务、子任务 wake、父子计数；保持 direct/own/Mix entry gate |
| [memory](../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_memory.h)、[GM Mix](../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_gm_mix.h) | 复用原语与协同协议，仅在新共享状态处补发布合同 |
| [协议导出](../../src/a5/runtime/host_build_graph/host/scheduler_protocol_export.h)、[读取/审计](../../simpler_setup/tools/hbg/native_protocol.py) | 新身份和计数口径、schema 演进、因果与完整性检查 |
| [Graph 原测试](../../tests/st/a5/host_build_graph/graph_execution/test_graph_execution.py)、[激活反例](../../tests/ut/cpp/a5/runtime/host_build_graph/test_graph_activation.cpp) | 复用业务输入/内核，补来源、并发与非法状态测试 |

### 5.2 设计评审需要明确的取舍

1. **首版全部物化后执行**：功能与进度基线简单，但增加启动串行窗口；建议接受为第一阶段，不将增量执行能力隐去或宣称已经保留。
2. **ENTER/EXIT 内部服务节点**：建议复用 ready/wake 执行模型、避免根节点等待父 DONE 的环；成本为每实例两个内部节点。若减少节点，必须提供同等清晰的 gate/终态唯一性证明，不能仅删计数。
3. **一个实例一个临时物化 owner**：避免首版共享参数池的多 writer 问题；大单 Graph 的构造并行度有限。是否拆 chunk 由后续实测和发布证明决定，不牺牲计算任务公平性。
4. **小量 carrier/metadata 扩展**：优先只动 A5 resident 私有布局；若需要公共 HostApi 或 Definition ABI 变更，实施前单独列出影响，不把它当成已获确认的接口。

这些是设计选择而非已完成的技术结论。实施从 P0 的可失败门禁和映射模型开始，不从删除 fallback 开始；完成度以实现过程中的实际测试证据为准。

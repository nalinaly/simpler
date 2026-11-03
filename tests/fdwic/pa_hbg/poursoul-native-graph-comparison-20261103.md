# poursoul a5-hbg-native-graph：三模型 HBG 对照与归因

日期：2026-11-03。比较 `poursoul/simpler:a5-hbg-native-graph` 与当前 `nalinaly/simpler:feat/fully_distributed_within_core`。所有组均使用 HBG；每组 10 次正式调用、0 次额外预热、保留首轮和全部样本。

## 1. 结果与适用范围

对方默认实现使 Qwen Device 中位数降低 **18.04%**，但 CSA、SWA 分别增加 **15.03% / 11.28%**。双方 AICPU 的 Device 差异均小于 0.5%。全部数值检查通过，实际上传的计算 kernel 指纹一致。

完整 Device RunWall 中位数，单位 ms；变化率为 `(对方 H8 / 我们 AICore - 1) × 100%`。

| 用例 | 我们 AICore | 对方 AICore H8 | 变化 | 我们 AICPU | 对方 AICPU |
| ---- | ----------: | -------------: | ---: | ---------: | ---------: |
| Qwen3-14B | 49.540841 | 40.602912 | -18.04% | 29.329414 | 29.300176 |
| V4 Pro CSA | 2.522630 | 2.901679 | +15.03% | 2.069485 | 2.072893 |
| DS4.1 Flash SWA | 0.894401 | 0.995286 | +11.28% | 0.726281 | 0.729459 |

Host / bind 中位数，单位 ms。Host 包含构图、搬运及同步，不是纯调度时间；共享主机上的 Host 波动较大，不能据这张表单独认定稳定的 Host 优化。

| 用例 | 我们 AICore Host / bind | 对方 H8 Host / bind | 我们 AICPU Host / bind | 对方 AICPU Host / bind |
| ---- | ----------------------: | ------------------: | ---------------------: | ---------------------: |
| Qwen3-14B | 55.450565 / 1.569550 | 44.743935 / 1.074100 | 30.462010 / 0.225790 | 30.806410 / 0.354170 |
| V4 Pro CSA | 21.594435 / 16.913425 | 17.434185 / 12.899855 | 26.645890 / 21.455970 | 26.571910 / 20.943940 |
| DS4.1 Flash SWA | 14.428390 / 10.936385 | 13.735380 / 10.471260 | 13.507205 / 9.557730 | 7.174955 / 5.361250 |

对方 Qwen H8 仍比其 HBG/AICPU 慢约 38.6%，因此不能把相对我们的改善表述为已超过 AICPU。SWA 的 H8 在后续独立消融批次测得 0.957815 ms，比主对照低 3.76%；11.28% 是这次主对照的结果，不是稳定常数。当前每组只有 10 个样本，未做多批交错置信区间。

## 2. 固定条件与验证

- 我们的分支 HEAD：`3f821683ede07db14f09d8df442dfa37df9f54d5`；实测冻结 runtime 为 `0f4b1e73d2b24240e49ea6a51869f6b33528ed2f`，二者仅差上一份性能 Markdown，运行时代码相同。
- 对方固定提交：`fcdc77f299698fdc4636bdaa235e72d0ba5057a3`；共同祖先为 `8f42cef9b3f0ebbda800e239aab0949e95552f25`。这是整个实现分支的对比，不是一个提交的单变量对比。
- 同一设备 5，A5 Ascend950DT_9581，32 AIC / 64 AIV；每批持有 `task-submit` 锁，组间串行。共享主机及其他 NPU 的负载不受该锁排他保护。
- CANN 9.3.0、CCEC 15.0.5、GCC 12.3.1、Python 3.12.15、Torch 2.12.0+cpu；PTO-ISA `217bb98952a8f9004b95fe938da97d8d5f68f702`。
- 各分支独立 worktree / venv / binding；构建最多 2 worker，测量绑定 CPU 0、1，Torch intra-op 2 / inter-op 1。
- 正式计时关闭泳道和 PMU，用既有 STRACE 解析 `chip.run.runner_run.device_wall`、`chip.run`、`chip.run.bind`，核对每组恰好 10 个完整窗口且无丢弃。
- 每次调用都验证新的 scheduler mode 发布：AICore 实际均 mode=1；AICPU 对照通过隔离副本中 `create_scheduler_state()` 的测试选路补丁进入原有 HBG legacy 路径，mode=4。该标记是强制实验选路，不表示负载不支持原生调度。
- 在每个分支内部，AICore/AICPU 两包只改变 Host 选路，AICore executor、AICPU executor 和 dispatcher 的二进制相同。跨分支运行库按各自实现构建。

### 工作负载相同，Host 编排按分支重编

复用上一轮已经数值验证的冻结计算镜像，针对对方 headers 重新编译相同 Host 编排源码，再将原 CoreCallable 字节装入新的 ChipCallable。这样兼容内部 Graph wire/类型实现差异，同时保持业务机器码不变。不能直接把旧编排 .so 当作新分支的接口实现。

三种模型的所有 5 个主对照臂，实际上传的计算 kernel 长度/FNV1a64 逐项一致；构造时也核对全部 CoreCallable SHA256。整体 ChipCallable SHA256 可以因 Host 编排 .so 不同而不同。

| 用例 | 配置 | 计算 kernel | 数值检查 |
| ---- | ---- | ----------: | -------- |
| Qwen3-14B | 40 层、B16、seq_len=3500、max_seq=5500、seed=1234；单层 Graph 定义复用 40 次 | 37 | 每臂 10 次后比较完整 out 和 40 层 K/V；rtol=0.05、atol=0.1 |
| V4 Pro CSA | B4/S2、seed=0、原冻结 position_ids 与 58 个数据文件 | 54 | 每次恢复原 INOUT 初值，并检查全部 7 项输出 |
| DS4.1 Flash SWA | 72 tokens、12 requests、seed=17、mixed；每请求 query_len=6、start_pos=8192 | 28 | 每次恢复原 INOUT 初值，并检查全部 5 项输出 |

Qwen 使用原合成、跨层复制权重的 fixture，KV 写入幂等；不是训练 checkpoint 的连续生成测试。CSA/SWA 复用原输入和 golden，不重新生成参考值。详细输入、比较器和来源提交沿用[此前三模型报告](hbg-three-models-performance-20261103.md#3-三种负载及数值检查)。本轮只测上述组件范围，不代表完整 DS4.1 decode_layer 已跑通。

## 3. 实测归因：ready 队列与任务形态

### 对方当前启用 ticket/H8，链式队列尚未接入

对方 [Host 配置](https://github.com/poursoul/simpler/blob/fcdc77f299698fdc4636bdaa235e72d0ba5057a3/src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1567) 固定 `kReadyPoolGroupWidth=8`。在本机形成 AIC/AIV/MIX 共 4/8/4 个队列；G 为每类型一个、共 3 个队列。
其 [领取逻辑](https://github.com/poursoul/simpler/blob/fcdc77f299698fdc4636bdaa235e72d0ba5057a3/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L1506) 允许跨组获取 ready 任务，每第四次搜索优先远端以保证公平性。我们的[原领取逻辑](https://github.com/nalinaly/simpler/blob/3f821683ede07db14f09d8df442dfa37df9f54d5/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L1393) 主要在自身 owner 及所属分片内发现任务。
最新两项提交 `2c0accdd0`、`fcdc77f29` 新增常驻链式 FIFO 和私有批链原语；生产代码未包含 `scheduler_linked_queue.h`，不能把本次数据称为链式队列性能。此边界与对方[实施记录](https://github.com/poursoul/simpler/blob/fcdc77f299698fdc4636bdaa235e72d0ba5057a3/tests/docs/a5-hbg-graph-task/README.zh-CN.md)一致。

### 控制实验 A：同一对方实现，G 与 H8

仅将 Host 的分组宽度 8 改为 0，设备执行库逐字节相同。每组 10 次，原数值检查不变。

| 用例 | G Device / ms | H8 Device / ms | H8 相对 G |
| ---- | ------------: | -------------: | --------: |
| Qwen3-14B | 49.413357 | 40.602912 | -17.83% |
| V4 Pro CSA | 3.095556 | 2.901679 | -6.26% |
| DS4.1 Flash SWA | 1.022998 | 0.995286 | -2.71% |

Qwen 的 G→H8 改变带来 8.810445 ms 的差异，量级接近分支主对照的 8.937929 ms。这是队列组织显著影响 Qwen 的直接控制实验，但两个差值不是可相减的严格贡献分解：我们的队列并非对方的 G。

### 控制实验 B：同一对方实现，H8 与已有 owner-inbox 路径

另起同卡批次复测 H8，并仅将 Host 发布的 `SchedulerGraphSummary.ready_pool_offset` 置为 0，选择设备代码中已有的 owner-inbox 路径。保留 H8 的分配布局和初始化，设备执行库完全相同；未移植我们的调度器、未修改计算负载。每组仍为 10 次，全部数值通过。

| 用例 | 本批 H8 / ms | owner-inbox / ms | owner 相对 H8 多耗时 |
| ---- | -----------: | ---------------: | -------------------: |
| Qwen3-14B | 40.650117 | 43.541517 | +7.11% |
| V4 Pro CSA | 2.918532 | 3.167786 | +8.54% |
| DS4.1 Flash SWA | 0.957815 | 1.045894 | +9.20% |

三项均是 H8 更快。因此证据不支持“CSA/SWA 退化主要是新 ready pool 造成，去掉即可恢复”的假设。H8 相对对方自身已有路径是正向贡献；相对我们版本的剩余差距必须继续看其它 resident 路径变化。该 owner-inbox 对照不是我们的基线实现，也不能把两者混称为同一队列版本。

### 任务形态解释了优化敏感性

L1 的实际身份集合 `(task_id, block_idx, block_num, subtask_slot)` 在两分支逐项相同；96 个 core 均有执行。只有 Qwen 的编排调用 `rt_submit_graph`，CSA/SWA 为普通 HBG DAG。

| 用例 | 逻辑计算任务 | 实际 kernel 执行 | MIX 任务 / block | MIX active mask |
| ---- | -----------: | ---------------: | ---------------: | --------------- |
| Qwen3-14B | 10604 | 19208 | 40 / 960 | 7：AIC + AIV0 + AIV1 |
| V4 Pro CSA | 79 | 2308 | 26 / 833 | 3：AIC + AIV0 |
| DS4.1 Flash SWA | 24 | 549 | 1 / 32 | 3：AIC + AIV0 |

Qwen 有大量逻辑任务，层内 ready 发现和分配频繁，G/H8 与 owner/H8 两种消融都确认了队列组织的影响。CSA/SWA 的逻辑任务少得多、SPMD/MIX 的 block 更多，任务内协同与完成协议更值得关注；不能把 Qwen 的收益外推到这些负载。

## 4. 其它实现差异与尚未证实的原因

### 双 lane MIX 的资源范围是优先检查项

我们通过 [`scheduler_mix_resource_mask`](https://github.com/nalinaly/simpler/blob/3f821683ede07db14f09d8df442dfa37df9f54d5/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h#L884) 使用 `active_mask | AIC` 作为参与资源；对方在 [GM MIX 发布](https://github.com/poursoul/simpler/blob/fcdc77f299698fdc4636bdaa235e72d0ba5057a3/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_gm_mix.h#L197) 使用 `LaneMask::all()`，并在[准入成员计算](https://github.com/poursoul/simpler/blob/fcdc77f299698fdc4636bdaa235e72d0ba5057a3/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_admission.h#L207)中对 MIX 使用全部 worker。

对于 CSA/SWA 的 mask=3，这意味着本机参与范围从每 cluster 两个 lane 扩大为三个：全体成员从 64 变成 96，额外的 AIV1 不运行该 MIX 的计算 kernel，却参与资源协调。按源码和捕获中的 MIX 任务数推导，CSA 增加 26×32=832 个任务级成员参与，SWA 增加 32 个；这些是协议范围推导，不是本轮 L4 事件实测。Qwen 的 mask=7 本来就使用三个 lane，不存在这项范围差异。

这与“Qwen 改善而双 lane MIX 负载退化”的分布吻合，也比把回退直接归咎于 Graph 物化更有针对性。但本轮没有执行仅替换 MIX 资源协议的单变量实验，不能断言它解释了全部 15.03% / 11.28%，也不能给它一个耗时贡献比例。其它 admission、claim-seal、topology 检查和热路径代码差异仍是候选因素。

### Graph 参数与完成机制也有变化

- 对方的 [native Graph 布局](https://github.com/poursoul/simpler/blob/fcdc77f299698fdc4636bdaa235e72d0ba5057a3/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_native_graph.h) 使用共享 child plan 和紧凑 node binding；每个实例仍有自己的参数和可变执行状态。
- 对方在 [`bootstrap_ready_graph`](https://github.com/poursoul/simpler/blob/fcdc77f299698fdc4636bdaa235e72d0ba5057a3/src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp#L175) 的节点扫描中并行绑定 child 参数；我们先按独占写回范围并行物化，再进入 ready bootstrap。双方均保留启动屏障。
- 对方的 [`GRAPH_EXIT` fanin](https://github.com/poursoul/simpler/blob/fcdc77f299698fdc4636bdaa235e72d0ba5057a3/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h#L449) 依赖图的 sinks，并由有界控制服务推进；我们对每个 child 递减 `remaining_children`，最后一个完成者发布 EXIT。
- 对方还按 `WithNativeGraph / WithProfiling` 特化执行循环，并复用本轮的只读元数据检查结果。以上是源码事实，尚未逐项隔离其性能贡献。

### L1 定位：执行窗口确有差异，不能拼接为 L0 分解

每分支每例额外采集一次冷调用 L1，均通过原数值检查，不属于正式 10 次计时。单位 ms；执行窗口是首 kernel 开始到末 kernel 结束，包含并行执行和中间空隙，不能当作纯计算时间。

| 用例 / 分支 | 启动到首 kernel | 首末 kernel 窗口 | 最后 kernel 到退出 | kernel 次数 |
| ----------- | --------------: | ---------------: | -----------------: | ----------: |
| Qwen3-14B / 我们 | 未保留 | 50.748927 | 未保留 | 19208 |
| Qwen3-14B / 对方 H8 | 未保留 | 39.975694 | 未保留 | 19208 |
| V4 Pro CSA / 我们 | 0.269465 | 2.039891 | 0.065762 | 2308 |
| V4 Pro CSA / 对方 H8 | 0.358159 | 2.259306 | 0.070319 | 2308 |
| DS4.1 Flash SWA / 我们 | 0.266294 | 0.675759 | 0.067710 | 549 |
| DS4.1 Flash SWA / 对方 H8 | 0.322301 | 0.729959 | 0.076436 | 549 |

Qwen 的首末 kernel 窗口从 50.748927 降到 39.975694 ms，说明改善也出现在执行期间，不能仅用 Host bind 更快解释。其 L1 开启后 Host STRACE 被转入临时 spool，进程退出时已清理，所以未保留该诊断调用的 RunWall 起止；启动/退出明确留空，没有用另一轮 L0 数据补齐。
CSA/SWA 的单次 L1 同时显示启动及执行窗口增量；样本为冷调用且带插桩，不能用它们精确分摊 L0 中位数。原始 kernel 时段可以重叠，其耗时总和也不能当作 Device Wall。

本轮可支持的结论是：对方 H8 对其自身 G/owner 路径均有实测收益；Qwen 的队列敏感性得到直接验证；CSA/SWA 的剩余回退尚未单点归因。双 lane MIX 资源范围与普通 resident 热路径是下一步最有根据的隔离对象，不应直接整套替换当前实现并宣称三个模型均优化。

## 5. 证据、复现与限制

- [主对照汇总及全部样本](../../../../logs/poursoul-hbg-compare-20261103-125412/summary.json)、[150 次调用 CSV](../../../../logs/poursoul-hbg-compare-20261103-125412/samples.csv)、[队列路径消融及全部 60 次样本](../../../../logs/poursoul-hbg-compare-20261103-125412/queue-ablation-summary.json)、[L1 汇总](../../../../logs/poursoul-hbg-compare-20261103-125412/l1-summary.json)。
- [环境](../../../../logs/poursoul-hbg-compare-20261103-125412/environment.json)、[计算镜像与 Host 编排来源](../../../../logs/poursoul-hbg-compare-20261103-125412/payloads.json)、[两分支运行时差异](../../../../logs/poursoul-hbg-compare-20261103-125412/runtime-diff.patch)。
- [构建 native/legacy](../../../../logs/poursoul-hbg-compare-20261103-125412/build_variants.py)、[只重编 Host 编排并复用计算镜像](../../../../logs/poursoul-hbg-compare-20261103-125412/build_payloads.py)、[G 实验](../../../../logs/poursoul-hbg-compare-20261103-125412/build_global.py)、[owner-inbox 实验](../../../../logs/poursoul-hbg-compare-20261103-125412/build_sharded.py)。
- [主测量脚本](../../../../logs/poursoul-hbg-compare-20261103-125412/run-comparison.sh)、[续跑脚本](../../../../logs/poursoul-hbg-compare-20261103-125412/run-resume.sh)、[队列路径消融脚本](../../../../logs/poursoul-hbg-compare-20261103-125412/run-queue-ablation.sh)、[主统计](../../../../logs/poursoul-hbg-compare-20261103-125412/summarize.py)、[消融统计](../../../../logs/poursoul-hbg-compare-20261103-125412/summarize_queue_ablation.py)、[L1 分析](../../../../logs/poursoul-hbg-compare-20261103-125412/analyze_l1.py)。
- 候选包：`bundles/aicore`、`bundles/aicpu`、`bundles/global`、`bundles/sharded`；各包均有 `manifest.json`、`source.patch` 和实际二进制摘要。原始日志在归档 `results/`、`measurements/`、`baseline/measurements/` 和 `queue-ablation/`。

本次设备队列记录：

- `task_20261103_142804_240124628280`：CSA/SWA 主对照全部通过；Qwen 在第一次设备调用前因新增诊断参数与函数参数重名退出，没有产生 Qwen Device 计时。
- `task_20261103_143609_294224926804`：旧诊断目录阻止覆盖，在启动模型进程前退出，没有产生计时。
- `task_20261103_143952_317447021406`：修正并归档前述失败后，在同一设备 5 完成 Qwen、6 次 L1 及全部队列路径消融，exit 0，锁已释放。
- 所有有效样本均保留；两次脚本失败日志在 `failed-driver/` 和队列日志中。没有用失败样本替换成功样本，也没有对既有正式样本做筛选。

复核现有结果（纯日志分析，不触碰 NPU）：

```bash
COMPARE="$(cd ../logs/poursoul-hbg-compare-20261103-125412 && pwd)"
source "$COMPARE/bench_env.sh"
python "$COMPARE/summarize.py"
python "$COMPARE/summarize_queue_ablation.py"
python "$COMPARE/analyze_l1.py"
```

重新上板前需在新输出目录准备独立构建和冻结包，先执行 `.claude/skills/onboard-arch-precheck/check.sh a5`，再通过单卡 `task-submit` 串行运行所需对照；正式计时仍为每组 10 次。不要覆盖本归档。上述 links 指向当前工作区的相邻 logs，二进制和输入不会随 Markdown 自动进入 Git；跨机器复现需一起提供。
本轮没有重跑两分支全仓库测试，也没有新采 L4 协议事件审计。所有实验选路改动仅用于隔离副本且构建后恢复，主工作区运行时代码未变。

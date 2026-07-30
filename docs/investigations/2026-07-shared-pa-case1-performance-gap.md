# Shared PA Case1：standalone 2.3 ms 不是等价基线，五个微优化候选均未保留

**Date**: 2026-07-30
**Verdict**: deferred-pending-equivalent-benchmark

## Question

把 standalone 的 shared TensorMap 与 schema-v5 泳道图迁入 simpler 的真实
Paged Attention（PA）Case1 后，是否还存在一项可删除的软件开销，能够把
`perf-clock` 的完整 Submit 窗口从约 2.7 ms 降到 standalone 的约 2.3 ms？

这个问题容易把“相同的 96 个 worker 和 1,280 个 task”误当成“相同的
benchmark”。本调查先核对两边实际执行的工作和 Claim 拓扑，再只验证能够由
生成代码或泳道图支持的候选，不通过反复跑设备碰运气。

## Measurement scope

迁移基线为本地提交 `18a88a41`：

```text
Update: 迁移 PA 专用 shared TensorMap 与 schema-v5 泳道图
```

真实 PA 的锁定口径为：

- Case1、单 group、B256；
- 96 workers、每核 1,280 次 Submit；
- Alloc 仅 8 个 AIC 候选，QK/PV 为全部 32 个 AIC，
  SF/UP 为全部 64 个 AIV；
- 256 MiB shared heap、8 个 heap shard、8 个 vector Claim cursor shard；
- UP writer history 含 3 条 symbol record，并使用 group CAS 和
  predecessor handoff；
- `perf-clock` 设备计数频率为 1 GHz。

机器上没有可用的 `task-submit` / `perf_lock` 执行面，因此所有 A5 数字都是
**未锁频、未隔离**的 fresh-process 结果。它们只适合判断明显回归或稳定方向，
不适合作为亚百分比差异的最终验收。

冻结基线的五个 fresh-process Submit 样本为：

```text
2665.43, 2679.92, 2746.03, 2818.71, 2968.16 us
median = 2746.03 us
mean   = 2775.65 us
range  = 302.73 us
```

## Why the standalone number is not comparable

standalone 的十个 fresh-process `perf-clock` 样本中位数为
`2304.520 us`，但它与真实 PA 只共享 worker 数和 task 数：

| Dimension | standalone | simpler real PA |
| --------- | ---------- | --------------- |
| Workload | synthetic fixed tiles/repeats | real BF16 PA tensors and dependencies |
| Alloc candidates | 96 workers | 8 AIC workers |
| Per-task candidate topology | synthetic fixed masks | Alloc8 / QK32 / SF64 / PV32 / UP64 |
| Total Claim attempts | 73,728 | 51,200 |
| Arrival shape | synthetic orchestration | data-dependent PA orchestration |

因此 `2746.03 - 2304.52 us` 不是已经被证明的“迁移损失”。改变真实 PA 的
候选掩码或业务工作量去复现 2.3 ms，会违反本阶段已经锁定的调度协议，也会把
benchmark 换成另一个问题。

完整 schema-v5 泳道图也必须同口径比较：真实 PA 为 `2.853980 ms`，
standalone full trace 为 `2.367119 ms`。不能把带完整泳道图的一个数字与
standalone 的低观测 `perf-clock` 数字混用。

## What was tried

所有候选都先检查协议不变量。候选 1、2 由泳道图提出并做有界 A5 测量；
候选 3–5 还用 whole-program O3 IR 证明预期指令或控制流确实发生变化。
代码候选最终均已撤回。

### Common validation and artifact identity

focused protocol test：

```bash
cmake --build build/ut_cpp --target test_fdwic_shared_pa_submit -j2
build/ut_cpp/test_fdwic_shared_pa_submit --gtest_color=no
```

A5 fresh-process 测量命令：

```bash
PTO_ISA_ROOT=<pto-isa-build> \
PYTHONPATH="$PWD/python:$PWD/build/cp312-cp312-linux_x86_64/python/bindings" \
.venv/bin/python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform=a5 --device=0 \
  --case=TestPagedAttentionUnroll::Case1 \
  --fdwic-tensormap=shared --fdwic-profile=perf-clock -q
```

CCEC 检查使用与 `CMakeLists.txt` 相同的 `-O3`、dav-c310 AIC/AIV 架构和以下
关键宏编译 unity orchestration，再把 driver 打印的 `-cc1` 命令从
`-emit-obj` 改成 `-emit-llvm-bc`：

```text
PTO_FDWIC_SHARED_MAP=1
PTO_FDWIC_SHARED_PA_UNITY=1
PTO_FDWIC_PERF_CLOCK=1
PTO_FDWIC_TRACE_ENABLED=0
```

可核对的构建身份：

```text
baseline source: 18a88a419b0c2d90012517106f6229fff3437f00
baseline aicore_kernel.o sha256:
  9e6b0c5303d7fc72ae8c5277e597ee2dbf9342ae6c37cdf4af05e39e0bb27a78
replay-pointer candidate aicore_kernel.o sha256:
  5becdffac8543fb10f18018871e48cb7d5bb7a4fabb3da4c887562dd55e7511c
```

候选 3、4、5 的 A5 输出目录分别为：

```text
typed Claim:
  TestPagedAttentionUnroll_Case1_20260730_220753
  TestPagedAttentionUnroll_Case1_20260730_221016
  TestPagedAttentionUnroll_Case1_20260730_221104
  TestPagedAttentionUnroll_Case1_20260730_221151
  TestPagedAttentionUnroll_Case1_20260730_221237
fixed geometry:
  TestPagedAttentionUnroll_Case1_20260730_221905
  TestPagedAttentionUnroll_Case1_20260730_222049
  TestPagedAttentionUnroll_Case1_20260730_222135
  TestPagedAttentionUnroll_Case1_20260730_222221
  TestPagedAttentionUnroll_Case1_20260730_222308
replay pointer:
  TestPagedAttentionUnroll_Case1_20260730_223731
  TestPagedAttentionUnroll_Case1_20260730_223900
  TestPagedAttentionUnroll_Case1_20260730_223957
  TestPagedAttentionUnroll_Case1_20260730_224054
  TestPagedAttentionUnroll_Case1_20260730_224148
```

候选 1、2 是未提交的早期原型，撤回时没有保留 source snapshot 与稳定的输出
目录映射；候选 3、4 也没有保留可链接 object。这里的描述和数值足以避免重复
追逐已否定的方向，但不能把这些临时 artifact 当作可复现性能基线。若要重开，
必须从 `18a88a41` 按下文描述重新实现，并重新建立完整 artifact identity。

### 1. Relocate the shared Claim cursor

原理是假设 writer-history 发布前后的 cache-line 竞争影响下一次 Claim，因此
原型把 cursor 更新挪到 writer-history 之后，并配套调整 padding/layout。

协议和 golden 校验通过，但五次测量的中位数约从 `2746.03 us` 上升到
`2861.6 us`，约回归 `4.2%`。这个方向被撤回。

### 2. Snapshot occupied state in drain phase B

原理是在一次 drain 中缓存 `occupied_count`，避免无变化时重复读取，并在已知
为空时提前退出。

focused tests 通过；全局 Submit 中位数只变化约 `-1%`，落在未锁频噪声内，
而泳道图中的目标 drain phase 没有同步改善。这个方向被撤回。

### 3. Specialize Claim and `MixedKernels` by task kind

先核对整程序 IR 后发现，原始 Claim wrapper 已经折叠成直接的
`llvm.hivm.atom.MAX.G.s64` 与结果比较，并不存在额外函数调用层。随后仍验证了
按 task kind 固化 role/kernel 字段的原型：

- AIC/AIV replay body 的 `ctpop` 从 `4/4` 变为 `0/0`；
- `MAX.G.s64` 数量不变；
- 最终 `.text` 均为 96,984 B。

候选五次样本为：

```text
2660.82, 2765.74, 2786.35, 2844.70, 2868.38 us
median = 2786.35 us
```

相对冻结基线中位数回归 `40.32 us`（`+1.47%`），且区间重叠。这个方向被撤回。

### 4. Constant-fold the phase-1 geometry

shared Case1 gate 已限定 block size 128 和单 group，因此原型把 runtime 除法
改为右移，并把只执行一次的 group loop 展开。

整程序 IR 证明：

- AIC/AIV 中相关 `udiv` 均变成 `lshr 7`；
- `bn += 64` 的 loop latch 被删除；
- Claim 原子和 `ctpop` 数量不变；
- 最终 `.text` 仍为 96,984 B。

候选五次样本为：

```text
2693.90, 2702.31, 2806.76, 2849.86, 2881.37 us
median = 2806.76 us
```

相对冻结基线中位数回归 `60.73 us`（`+2.21%`）。这个方向被撤回。

### 5. Carry the attached `DistCore` pointer in the replay capability

原实现的 8-byte replay token 只保存 role 和 block id。每个静态 Submit 点仍
读取一次 block-local `g_self`。原型把 attach 时已经验证的 `DistCore` 指针也
放入 token，并在 PA replay 入口一次性验证编译角色。

整程序 IR 证明该原型确实删除了目标工作：

- 两个 replay body 的十个正常 Begin `g_self` load 全部消失；
- 全模块 `g_self` load 从 23 降到 13；
- `MAX.G.s64` 数量保持 15（含声明）；
- 没有新增 `memcpy` / `memmove`；
- 最终 `.text` 从 96,984 B 降到 95,704 B。

但 token ABI 从 8 B 扩到 16 B，并引入额外 context materialization。A5 五次
样本为：

```text
2694.15, 2703.60, 2739.43, 2739.90, 2873.22 us
median = 2739.43 us
mean   = 2750.06 us
```

相对冻结基线中位数只有 `-6.60 us`（`-0.24%`），远小于两组未锁频样本的
波动，不能证明收益足以承担 ABI 和能力生命周期复杂度。这个方向被撤回。

若未来重启该原型，还必须同时满足以下安全条件：

- 捕获的 pointer 只能作为同步 replay 动态范围内的非逃逸 capability，不能向
  orchestration 暴露可修改的 `DistCore *`；
- winner Finish 除了重读 role/block，还必须比较当前 `g_self` 与捕获指针完全
  相同，避免另一份相同 role/block 的 `DistCore` 通过；
- 8 B/align-4 到 16 B/align-8 的 ABI 变化必须触发 caller/callee 全量重编；
  C++ mangled name 本身不编码类尺寸，不能手工混用旧新 object；
- IR 验收必须沿 CFG 确认成功 Begin 路径无加载，不能要求整个文件零
  `g_self`，因为 getter、winner revalidation 和冷错误路径仍应保留。

## Result

| Candidate | Code/profile signal | A5 median versus baseline | Verdict |
| --------- | ------------------- | ------------------------- | ------- |
| Cursor relocation | Work moved; no retained IR identity | About `+4.2%` | Dropped |
| Drain snapshot | Target drain phase did not improve | About `-1%`, within noise | Dropped |
| Typed Claim | `ctpop` removed; MAX and `.text` unchanged | `+40.32 us` / `+1.47%` | Dropped |
| Fixed geometry | `udiv` and one loop latch removed | `+60.73 us` / `+2.21%` | Dropped |
| Replay pointer | Ten hot Begin loads removed; `.text` -1,280 B | `-6.60 us` / `-0.24%` | Dropped |

结论不是“生成代码没有变化”：候选 3–5 都有明确的 IR 变化。结论是这些变化在
当前真实 PA 和未锁频 A5 上没有形成可重复、足以承担复杂度的 makespan 收益。

## Trace evidence

full-trace 的 aggregate core-time 差值用于定位，不等于 wall time，也不能跨
lane 直接相加：

| Partition | Real PA minus standalone aggregate core-time |
| --------- | -------------------------------------------: |
| Claim | +10.190 ms |
| Register | +13.678 ms |
| BetweenSubmit | +9.948 ms |
| EfDrain | +8.884 ms |
| FinalDrain | +24.252 ms |

进一步核对后：

- Register 差值的 `96.42%` 是既定 predecessor handoff 的等待，不是可删除的
  TensorMap 插入代码；
- EfDrain 和 FinalDrain 主要反映依赖完成与 replay 到达分散；
- UP 到下一 Alloc 的平均间隙为真实 PA `550.109 ns`、standalone
  `252.780 ns`，AIC/AIV 的额外部分都约 0.30 us，更像共同的 orchestration
  边界工作，而不是 UP winner 独有的 publish 行为；
- 只对这个 per-lane 边界做极乐观估算：
  `(550.109 - 252.780) ns × 255 transitions = 75.82 us/critical lane`。
  这假设每次差值都串行落在同一关键 lane 且可被全部删除；它不是由上表
  aggregate core-time 相加得到的 wall-time。即使达到这个不现实的上界，也
  不足以把真实 PA 的 2.7–2.8 ms 变成 2.3 ms。

## Why not now

候选 1 是工作搬移，候选 2 的目标 phase 没有改善，候选 3–5 的生成代码按预期
变化；但没有一个得到稳定的 A5 cross-core Submit makespan 收益。继续在当前
未锁频环境中堆叠微优化或增加重复次数，会把噪声当成结论。

更重要的是，standalone 2.3 ms 和真实 PA 基线不是同一个 workload/Claim
协议。当前证据不支持把两者的差值定义为待修复回归，也没有发现足以解释
0.4–0.5 ms 的单项纯软件开销。

## When to reconsider

满足以下任一条件时再打开本调查：

1. 建立 standalone 与 simpler 完全等价的 benchmark：相同业务工作、相同
   candidate mask、相同 Claim 次数与到达拓扑、相同宏和工具链；
2. 明确决定把目标改为复现 standalone synthetic workload，而不是保持真实 PA
   协议；
3. `task-submit` / `perf_lock` 恢复后，用至少 12 个 fresh process、固定构建
   artifact 和 source hash 建立真实 PA 自身基线；
4. 新泳道图把某个纯软件 exclusive partition 证明为稳定的关键路径热点，再为
   该热点设计单一候选。

若选择真实 PA 口径，应以它自己的锁频基线和 correctness/closure 作为验收，
而不是继续把 standalone 2.3 ms 当作硬门槛。

## References

- Migration baseline: `18a88a41`
- `docs/fully_distributed_within_core.md`
- `docs/dfx/l2-swimlane-profiling.md`
- `docs/dfx/l2-timing.md`
- `tests/ut/cpp/a5/test_fdwic_shared_pa_submit.cpp`
- `examples/a5/fully_distributed_within_core/paged_attention_unroll/`

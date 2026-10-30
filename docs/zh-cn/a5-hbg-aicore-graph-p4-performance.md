# A5 resident Graph：性能、资源与启动长尾调查

状态：**首版功能、原生协议及同负载性能/资源报告已交付；独立发布信号在最终 A/B 中未再复现原生 Graph 的启动长尾。并非所有场景或 Host/Device 指标都更快，CSA Host 波动尚不能归因。** 设计见[总设计](a5-hbg-aicore-graph-design.md)，功能和故障证据见[实现过程](a5-hbg-aicore-graph-implementation.md)。第 1–4 节保留初轮失败与修复过程，最终结果见第 5 节；没有覆盖或删去失败样本。

## 1. 对照方法和计时范围

- baseline：开发前 `02335c0f46d6f12a0fd0adaeccf4869b1164ff64`，独立 detached worktree、独立 venv、从自身源码构建。
- native：`81cb667740d85760e80b919ff110c7236aabd758`。本节初轮采样时 runtime 源码无未提交差异；之后的实验另记，不能混入本表。
- 同一 A5 `Ascend950PR_958b`、设备 0；实际 32 cluster、96 个 AIC/AIV worker。每次上卡先通过 ACL 验证芯片和计算利用率；本机无 task-submit，**未获得队列锁**，不能给环境噪声不存在的保证。
- 两边均 Python 3.12、Torch `2.12.0+cpu`、固定 PTO-ISA `418de18dde8fa33773dc5c384fdfe5506463b628`、同一 CANN 安装，CPU 线程数均为 4。禁用的是 Torch NPU 自动导入，不是 simpler 的设备执行。
- 三组原 Graph：原 2D replay、原 AIC/AIV Graph、原全 cluster Mix Graph；另测无 Graph 的冻结 CSA B4S2Seed0。原 kernel、输入、golden、N、容差均不改。CSA 的 58 个数据文件逐个验证 SHA256。
- 计算 kernel 的实际 `code_bytes` 和 `code_fnv1a64` 也相同：三组 Graph 分别 3/2/3 个，CSA 54 个。不是只比较源文件名。
- 每配置每批 105 次，第一轮单列为“本 Worker 首调用”；前 5 次不进入稳定态统计，剩余 100 次全部保留。三批顺序为 AB、BA、AB，每版本每配置 **300 个稳定态样本 + 3 个首调用样本**，共 2520 次调用，每次都检查原数值结果及 fresh backend publication。
- Graph baseline 必须 mode=3，native 必须 mode=1/2；无 Graph CSA 两版都必须 mode=1/2。采样入口拒绝 stale publication、错误后端、跳过 golden 和把 L4 当作 L0 样本。

完整样本、STRACE、版本/源文件/二进制摘要在本地 `outputs/graph-ab-formal.YYfb8s/`，汇总为 `comparison.json`。首调用不含编译和 `Worker.init()`，不能称为“进程从零启动总耗时”。P90/P99 使用 `(n−1)×q` 线性插值；合并样本不作为独立实验重复来构造置信区间，另列三批中位数以暴露批间波动。

| 指标 | 范围 |
| ---- | ---- |
| Host bind | 标准 `chip.run.bind`，含 Host 构图及 bind 内的参数/状态处理 |
| Host 调用 | 标准 `chip.run`，含构图、发布、准备、驱动运行与回收；不是 Device 时间 |
| Python 调用 | `Worker.run()` 的外层墙钟，原始 ledger 保留；不含其后 golden 比较 |
| Device RunWall | 原有 AICPU `RunWall`，覆盖 resident 初始化/物化、执行和退出；不使用第一个 submit 到最后 kernel 的窄窗口替代 |
| GM committed | 既有 `MemoryAllocator` 当前跟踪的设备分配请求字节，含 retained arena、参数、runtime、计算数据；不是驱动页粒度占用或整卡已用 HBM |

Host bind、发布、prepare、runner 子阶段均由原 STRACE 解析，缺失或丢弃记录会拒绝生成报告。L4 只用于另一次诊断，不混入性能分位数。各列的中位数不能相加后当作同一轮的阶段分解。

## 2. 初轮数据：必须保留的回退和长尾

### 稳定态 Device RunWall，单位 µs

| 用例 | 版本 | P50 | P90 | P99 | 最大 | >5ms 次数/300 |
| ---- | ---- | --- | --- | --- | ---- | ------------- |
| Graph 2D | baseline | 227.8 | 237.8 | 248.7 | 279.5 | 0 |
| Graph 2D | native | 274.1 | 309.1 | 327.0 | 331.8 | 0 |
| Graph AIC/AIV | baseline | 203.7 | 216.1 | 228.1 | 258.8 | 0 |
| Graph AIC/AIV | native | 197.1 | 253.5 | 1,300,025.5 | 3,618,738.3 | 14 |
| Graph Mix | baseline | 414.9 | 435.8 | 450.2 | 464.1 | 0 |
| Graph Mix | native | 356.8 | 416.7 | 404,940.8 | 570,879.0 | 17 |
| 无 Graph CSA | baseline | 1,946.0 | 1,985.2 | 2,082.0 | 2,120.6 | 0 |
| 无 Graph CSA | native | 1,951.1 | 1,993.5 | 2,052.9 | 2,069.9 | 0 |

5ms 只是本次微图的长尾计数阈值，不是新增 runtime timeout，也不是通用算子的性能 SLA。**Graph 2D 的 Device P50 退化约 20.3%，必须明确报告；AIC/AIV、Mix 即便中位数较小，也因严重长尾不能宣称性能优化成功。** 无 Graph CSA 的中位数差约 +0.27%，本次没有观察到同等级长尾，不能据此宣布所有无 Graph 负载都无回退。

| 用例 | baseline 三批 Device P50 | native 三批 Device P50 |
| ---- | ------------------------ | ---------------------- |
| Graph 2D | 224.4 / 226.4 / 230.5 | 290.9 / 228.5 / 287.7 |
| Graph AIC/AIV | 209.5 / 187.8 / 201.5 | 194.5 / 202.3 / 194.5 |
| Graph Mix | 418.8 / 410.3 / 415.6 | 356.5 / 355.0 / 358.3 |
| 无 Graph CSA | 1943.8 / 1946.7 / 1947.7 | 1959.4 / 1950.6 / 1943.5 |

### Host 与首调用

| 用例 | baseline Host bind P50 | native Host bind P50 | baseline Host 调用 P50 | native Host 调用 P50 |
| ---- | ---------------------- | -------------------- | ---------------------- | -------------------- |
| Graph 2D | 804.5 | 799.1 | 1611.6 | 1774.2 |
| Graph AIC/AIV | 786.5 | 825.5 | 1601.6 | 1757.4 |
| Graph Mix | 609.4 | 740.9 | 1529.4 | 1754.3 |
| 无 Graph CSA | 71647.4 | 70627.5 | 80156.1 | 79189.2 |

单位 µs。CSA 使用原 Host 输入路径，每轮仍有约 399MiB 级固定数据，Host 调用约 80ms 与 Device 约 1.95ms 是不同范围；不能把后者写成包含输入搬运的总端到端时间，也不能为了缩短 Host 数字改用另一份负载/传参方式。

| 用例 | baseline 首调用 Device P50 [min,max] | native 首调用 Device P50 [min,max] |
| ---- | ------------------------------------ | ---------------------------------- |
| Graph 2D | 698.8 [698.5,708.1] | 453.9 [442.4,461.3] |
| Graph AIC/AIV | 664.1 [656.4,698.5] | 429.7 [416.8,1,209,986.6] |
| Graph Mix | 889.2 [875.7,934.7] | 587.8 [577.6,638.0] |
| 无 Graph CSA | 2176.1 [2122.1,2216.3] | 2214.9 [2168.5,2231.5] |

单位 µs，每格只有 3 个样本；不以这 3 个样本估计稳定的冷启动尾分布。逐批完整样本及 P90/P99 保存在汇总中。

## 3. GM 预算与实际 retained 分配核对

| 用例 | baseline committed B | native committed B | 差值 B |
| ---- | -------------------- | ------------------ | ------ |
| Graph 2D | 8,736,157 | 9,084,188 | +348,031 |
| Graph AIC/AIV | 8,511,829 | 8,850,068 | +338,239 |
| Graph Mix | 7,731,869 | 8,207,196 | +475,327 |
| 无 Graph CSA | 428,442,804 | 428,442,740 | −64 |

每配置的 300 个稳定态样本都为同一个 committed 值，没有逐 run 增长。统计包含旧执行存储/参数池、真实计算数据和新调度区，不能只挑新增的几个控制字报告。它不含 allocator 无法归属的驱动内部页开销。

结合原生捕获的节点/实例/Definition 数量，直接调用生产 `scheduler_plan_layout()`，再按 `SchedulerGraphBinding::reserve()` 的顺序和真实 `sizeof/alignof` 计算 Graph 附加区：

| 用例 | 内部 R / 实例 I / 不同 Definition | 共享 Definition B | 通用 resident 区 B | Graph 附加区 B | 整个 scheduler-state B | 原始分配 B（+127 对齐余量） |
| ---- | --------------------------------- | ----------------- | ------------------ | -------------- | ---------------------- | --------------------------- |
| Graph 2D | 25 / 3 / 2 | 3680 | 342016 | 5888 | 347904 | 348031 |
| Graph AIC/AIV | 15 / 3 / 1 | 1312 | 335104 | 3072 | 338176 | 338303 |
| Graph Mix | 9 / 3 / 1 | 368 | 473472 | 1792 | 475264 | 475391 |

初轮版本的附加区按 `48×R` 映射、每实例 128B 描述及 128B 控制、一个 128B build control、按不同定义保存的 carrier、外部边 `4×E` 和对齐组成；内部 CSR 共享，不为每实例复制。通用区还包括既有 108-worker 容量的 context/payload/ready 结构、每节点 512B task control、16B metadata、既有 trace 占位及协同区；实际运行核仍为 96，不是扩大 cohort。Mix 的 close seals 为 `3×32×128=12288B`，已包含在预算内。下述独立信号候选将 build control 扩为 256B，故每个有 Graph 的 run 另增 128B；不反写这张初轮实测表。

实际 committed 差值恰好等于上述原始 scheduler-state 分配，加上已记录的 callable image 大小差：2D 为 0，AIC/AIV、Mix 为 −64B。无 Graph CSA 的差也只有 callable image 的 −64B，计算 kernel 的大小/hash 不变。**因此本批数据没有隐藏另一份实例 payload/参数池副本；无 Graph 没有新增 Graph 专属 GM 分配。** 预算核对程序保存在同一结果目录 `budget.cpp`，不依赖手工猜测结构体大小。

L0 的 Graph 协议区为零。L4 在通用诊断区之外另加 `128×(I+R+W)` 字节：三例分别 15872/14592/13824B；不能把这些诊断字节混入上面的 L0 committed 对比。Host 侧仍有同容量 retained scheduler-state staging buffer，另有录制器/临时容器；本表不是进程 RSS 报告。

## 4. 长尾定位与尚未成立的优化假设

初轮长尾的 Host bind/发布仍约 1ms，增加的时间在 `runner_run` 与 Device RunWall 内，不是 Python golden 或 JSON 输出。没有删掉这些样本。

- L4：原 AIC/AIV Graph 连续 40 轮，逐轮数值及严格协议通过，未复现大长尾；捕获在 `outputs/graph-tail-l4.Xxapwy/`。这不能反证 L0 没问题，采集可能改变时序。
- L1：60 轮，真实 kernel span 均很短（最大约 28.6µs），但 L1 没有 AICPU 生命周期细节，不能据此确定等待在 kernel 前还是后；捕获在 `outputs/graph-tail-l1.PPgg9t/`。
- L2：60 轮，抓到约 782.38ms / 624.23ms / 19.86ms 的长尾。最慢一轮 `bootstrap_wait` 为 **782.27ms**，发布寄存器后的执行窗口约 **37.26µs**，退出等待约 **10.07µs**。五个 AICPU 线程均已完成 context 发布，等待发生在 Scalar 启动阶段。捕获在 `outputs/graph-tail-l2.iiDULe/`，最慢为 `validation-round-16`。
- 无 Graph ordinary/Mix 对照：两版各 105 次，稳定态 Device P50 为 256.7/264.7µs、最大为 267.9/339.4µs，均没有 >5ms 样本。单批约 +3.1% 需复测，不能先归为噪声；结果在 `outputs/graph-tail-ordinary.fFso8Z/`。

一个待检验假设是 96 个 worker 在新物化屏障持续 atomic poll，使发布或后续 ready 屏障受压。仅复用既有 8→32 次本地退避的试验**尚不能作为修复**：两个微图各 300 个稳定态样本中，AIC/AIV 的 >5ms 次数为 2（最大 17.09ms），Mix 仍为 22（最大 3.04s）。该批同时有 CPU 仿真回归，因此不拿其 Host 数字作正式性能结论；Device 秒级长尾仍足以否定“问题已经解决”。结果在 `outputs/graph-backoff-trial.zPTegK/`。

L3（保留退避试验，Mix 60 轮）进一步抓到 3.199s 的长尾：所有 worker 都在 context 发布后约 3.198s 才进入 ready bootstrap，ready 扫描/初始准备本身只用 18–39µs。因此长尾在物化完成的发布/等待，不在 ready 扫描或真实 kernel。最慢捕获为 `outputs/graph-tail-l3.IxmjOX/captures/validation-round-44/chip_swimlane_records.json`。

### 独立发布信号候选

`SchedulerGraphBuildControl` 的计数/只读字段保留在前 128B，第二个独立 128B 行只放 `materialized` 信号。最后一个 atomic 递减观察到旧值 1 的 owner，在 payload 写回和 barrier 之后用既有 `scheduler_publish_signal()` 发布；等待者读取 bypass 信号，并保留 atomic 零值确认。没有让普通 load 代替原子所有权，也没有让多个 owner 并发普通 store 同一个信号。无 Graph 路径不分配该行。

CPU 反例要求：失败 owner 不发布；暂停 owner 的等待者不能提前通过；只有零计数、没有信号不能通过；只有信号、计数未归零也不能通过。bootstrap/binding/services 三组测试通过。

真机候选各 305 次、前 5 次预热、全部数值验证；没有并行 CPU 仿真负载。结果在 `outputs/graph-signal-trial.17QnQd/`：

| 用例 | 样本数 | Device P50 µs | P99 µs | 最大 µs | >5ms |
| ---- | ------ | ------------- | ------ | ------- | ---- |
| Graph AIC/AIV | 300 | 188.6 | 273.9 | 327.0 | 0 |
| Graph Mix | 300 | 348.7 | 424.1 | 510.4 | 0 |

这个对照说明分离热轮询后，本次可复现的长尾已消失；具体硬件仲裁/队列机制没有单点实测，不能把“同址严格串行”擅自解释成已证明的无限队列或写饥饿。候选在这一阶段仍需完整回归和最终交替 A/B，后续结果见第 5 节，不拿这一批代替最终验收。不得靠减小 N、固定执行核或改变计算负载隐藏长尾。

## 5. 最终对照、验收与复现入口

### 独立信号版本：三批完整 L0 对照

最终运行保存在 `outputs/graph-ab-signal-final.NdEx1F/`，与初轮目录独立。仍为 AB、BA、AB，每批 105 次、前 5 次预热；增加无 Graph ordinary/Mix，每配置每版 300 个稳定态样本，共 **3150 次完整数值验证**。严格分析器确认各批版本/源码/二进制不变、同负载及 fresh backend；计算 kernel 的大小/hash 在五配置的全部批次都一致。

native 采样源码为 `81cb66774` 加独立发布信号差异；`git diff HEAD -- src` 的 SHA256 为 `e558770ef84e6489e57e9a027c90b57920271465b3d1d0c268f604b9bf8ae3a2`。AICore/AICPU/Host 二进制 SHA256 依次为：

```text
4c00bae98890a4896ce07636f4ddc4b5a9be3ec64923afc47ddd3c48ca32a99f
cee0385f0cc3e23e46b3adeb58c20a8a91c2af1f394c082338ba449422f3cd8e
fc9c4e24b17a3ab86325ca9f6a23627c9949d382d9b1e357a35a2205ad6c12bd
```

| 用例 | 版本 | Device P50 µs | P90 µs | P99 µs | 最大 µs | 中位数变化 |
| ---- | ---- | ------------- | ------ | ------ | ------- | ---------- |
| Graph 2D | baseline | 221.1 | 234.8 | 243.9 | 267.7 | — |
| Graph 2D | native | 227.6 | 257.9 | 300.2 | 326.6 | +2.95% |
| Graph AIC/AIV | baseline | 194.9 | 212.6 | 220.2 | 236.9 | — |
| Graph AIC/AIV | native | 183.6 | 200.7 | 245.9 | 276.0 | −5.83% |
| Graph Mix | baseline | 412.5 | 432.2 | 455.6 | 461.4 | — |
| Graph Mix | native | 349.4 | 382.8 | 457.9 | 514.6 | −15.30% |
| 无 Graph CSA | baseline | 1946.1 | 1985.5 | 2036.0 | 2074.6 | — |
| 无 Graph CSA | native | 1949.1 | 1983.2 | 2066.8 | 2106.9 | +0.15% |
| 无 Graph ordinary/Mix | baseline | 263.2 | 290.8 | 353.7 | 423.6 | — |
| 无 Graph ordinary/Mix | native | 261.7 | 290.5 | 342.3 | 405.7 | −0.57% |

各行 300 个稳定态样本的 >5ms 次数均为零。**本批原生 Graph 启动长尾没有复现，不等于所有分位数都改善：2D 的 P50/P99 分别增加约 3%/23%，AIC/AIV 的 P99 也增加约 12%。** Mix 的 Device 中位数改善不能替代 Host 总窗口结论。

| 用例 | baseline 三批 Device P50 µs | native 三批 Device P50 µs |
| ---- | --------------------------- | ------------------------- |
| Graph 2D | 214.8 / 227.3 / 220.0 | 224.0 / 227.6 / 230.4 |
| Graph AIC/AIV | 195.6 / 200.5 / 191.3 | 188.4 / 179.0 / 183.1 |
| Graph Mix | 411.8 / 419.5 / 407.7 | 354.4 / 346.3 / 348.9 |
| 无 Graph CSA | 1940.9 / 1954.8 / 1943.0 | 1945.3 / 1946.5 / 1953.1 |
| 无 Graph ordinary/Mix | 260.0 / 268.8 / 264.0 | 258.2 / 261.7 / 267.1 |

### Host 的额外成本与不稳定样本

下表均为中位数、单位 µs；分阶段中位数不能相加当作单轮分解。

| 用例 | 版本 | bind | publish_image | prepare_execution | runner | Host 总窗口 |
| ---- | ---- | ---- | ------------- | ----------------- | ------ | ----------- |
| Graph 2D | baseline | 847.4 | 46.4 | 56.1 | 478.0 | 1690.9 |
| Graph 2D | native | 788.5 | 153.6 | 55.2 | 447.2 | 1714.7 |
| Graph AIC/AIV | baseline | 712.2 | 41.7 | 59.4 | 440.9 | 1499.3 |
| Graph AIC/AIV | native | 793.2 | 147.4 | 56.0 | 402.6 | 1637.2 |
| Graph Mix | baseline | 659.4 | 44.2 | 56.3 | 637.1 | 1604.9 |
| Graph Mix | native | 690.1 | 239.8 | 54.0 | 566.8 | 1757.8 |
| 无 Graph CSA | baseline | 71696.6 | 463.7 | 88.8 | 2250.4 | 80339.1 |
| 无 Graph CSA | native | 119480.1 | 505.6 | 88.1 | 2257.0 | 132435.6 |
| 无 Graph ordinary/Mix | baseline | 2166.0 | 177.7 | 65.9 | 525.4 | 3916.2 |
| 无 Graph ordinary/Mix | native | 2157.5 | 179.4 | 69.3 | 536.0 | 3906.5 |

**Graph AIC/AIV、Mix 的 Host 中位数分别增加 9.20%、9.53%，不是端到端加速。** 原生 Graph 需要额外发布 resident 元数据；`publish_image` 在三例中增加约 107/106/196µs。[实际发布函数](../../src/a5/runtime/host_build_graph/host/runtime_maker.cpp)包含 prerequisite metadata 与 runtime image 的 H2D，新增存储见下表。数据支持“发布成本抵消了部分 Device 收益”，不支持把整个 Host 差值精确归给一次 memcpy，也不以不同阶段的中位数强行闭合。

CSA 合并 Host 中位数看似增加 64.85%，但原样数据明显非稳定：三批 baseline 为 **139.095 / 80.206 / 78.443ms**，native 为 **136.820 / 133.859 / 78.964ms**；第二批 native 自身也从约 136ms 降到最后 20 次约 84ms，源码/二进制均未变化。两版都出现超过 2.5s 的 Host 单轮，Device 没有对应长尾。不能删掉这些轮次，也不能把合并比例直接判成调度器引入的稳定回退；短批复核另列，原 300 样本始终保留。没有测得具体主机干扰来源，未锁设备/主机的限制仍有效。

短批复核保存在 `outputs/graph-csa-host-check.IMPE8K/`：同源码/二进制，AB、BA、AB，每批每版 25 次、前 5 次预热，共 150 次原样数值通过，每版 60 个稳定态样本。baseline/native Host P50 为 **133.024/132.964ms（−0.05%）**，Device P50 为 **1945.6/1949.4µs（+0.19%）**；Host 三批分别为 78.698/139.261/133.273ms 与 133.112/133.942/99.284ms。它证明高低 Host 档位并不固定绑定某个版本，但仍不证明变化的物理原因，也不能据此给出“所有 Host 回退已排除”的结论。

| 用例 | baseline 首调用 Device P50 [min,max] µs | native 首调用 Device P50 [min,max] µs | baseline / native 首调用 Host P50 ms |
| ---- | --------------------------------------- | ------------------------------------- | ------------------------------------ |
| Graph 2D | 729.2 [710.8,1296269.5] | 469.5 [444.8,477.3] | 41.239 / 40.336 |
| Graph AIC/AIV | 686.8 [676.1,51574.5] | 396.6 [373.8,420.6] | 39.636 / 42.331 |
| Graph Mix | 897.6 [869.2,926.3] | 536.7 [439.6,654.5] | 36.389 / 39.962 |
| 无 Graph CSA | 2160.6 [2127.6,2164.8] | 2153.3 [2138.8,2173.5] | 125.804 / 179.115 |
| 无 Graph ordinary/Mix | 482.1 [456.6,492.7] | 469.3 [467.1,481.1] | 42.384 / 41.981 |

首调用每版只有 3 次。baseline 的 2D/AIC+AIV 首调用长尾也保留，原因未定位；不能用这三个样本声称 native 在全部冷启动场景一定更快。

### PA、容量和泳道的最终核对

原 PA `Case1` 使用 B=256、16 heads、128 head-dim、8192 context，保留原 KV 数据规模、四个计算 kernel 和 golden。每版 8 次全部通过、均为 fresh resident；首调用单列，余下 7 次仅作回归对照，不用小样本声称可靠的尾分布。目录为 `outputs/graph-pa-final.oXZede/`。

| PA 指标 | baseline | native |
| ------- | -------- | ------ |
| Device P50 / P90 / P99 / max µs | 1389.2 / 1399.5 / 1401.2 / 1401.4 | 1402.0 / 1413.5 / 1420.2 / 1420.9 |
| Host P50 / P90 / P99 / max ms | 191.336 / 194.097 / 194.431 / 194.468 | 192.651 / 194.737 / 196.891 / 197.130 |
| 首调用 Device µs / Host ms | 1629.9 / 234.037 | 1623.1 / 228.856 |
| committed B | 1294079716 | 1294079652 |

PA 的 Device/Host 中位数分别为 +0.93%/+0.69%。另存两版各一份 L1 泳道，各 **1024 次真实 kernel 执行**，观察区间约 1287.6/1285.8µs；它们小于完整 Device RunWall，不能拿观察区间替代端到端时间。L1 未采 dep_gen，没有完整依赖箭头，不属于 Graph v8 协议验收。

| 最终容量核对 | baseline committed B | native committed B | 增量 B | native 原始 scheduler-state 分配 B |
| ------------ | -------------------- | ------------------ | ------ | ---------------------------------- |
| Graph 2D | 8736157 | 9084316 | +348159 | 348159 |
| Graph AIC/AIV | 8511829 | 8850196 | +338367 | 338431 |
| Graph Mix | 7731869 | 8207324 | +475455 | 475519 |
| 无 Graph CSA | 428442804 | 428442740 | −64 | 既有布局不变 |
| 无 Graph ordinary/Mix | 11613404 | 11613340 | −64 | 既有布局不变 |

三个 Graph 的分配均比初轮恰好多 128B，等于独立信号行；其余 −64B 仍为 callable image 的差值，计算 kernel 大小/hash 不变。全部稳定态分配值不随轮次增加。原 heap-tail、tensor/scalar pool 没有新增副本，无 Graph 不分配信号行。

最终信号版本的独立 L4 捕获如下，均通过严格 `graph_verified`；单位 µs。这是带插桩的一次实例，不与 L0 分位数混算。物化发布包络在第一条 kernel 之前，计入启动成本；owner 范围是各自同核起止，包络跨 worker 仅用于时序展示，不替代原子票号的因果证明。

| 捕获 | AICPU 生命周期包络 | bootstrap 等待包络 | 单 owner 物化 min–max | 物化及发布包络 | kernel span | 真实 kernel 次数 |
| ---- | ------------------ | ------------------ | --------------------- | -------------- | ----------- | ---------------- |
| Graph 2D | 253.6 | 91.9 | 20.6–21.7 | 30.1 | 79.5 | 13 |
| Graph AIC/AIV | 194.3 | 82.0 | 13.8–15.5 | 24.6 | 33.2 | 9 |
| Graph Mix | 480.1 | 85.1 | 5.5–6.3 | 22.8 | 121.1 | 288 |

本地泳道路径（未加入 Git）：

- `outputs/TestGraphExecutionResident_record_then_replay_2d_20261030_222052/merged_swimlane.json`
- `outputs/TestGraphExecutionAicAivResident_record_then_replay_aic_aiv_20261030_222055/merged_swimlane.json`
- `outputs/TestGraphMixResident_record_then_replay_mix_spmd_20261030_222058/merged_swimlane.json`
- `outputs/graph-pa-final.oXZede/pa-baseline-l1/merged_swimlane.json`
- `outputs/graph-pa-final.oXZede/pa-native-l1/merged_swimlane.json`

另采 baseline 原 Graph Mix L4：`outputs/graph-pa-final.oXZede/graph-mix-baseline-l4/merged_swimlane.json`。两版均为 **3 个实例 × 32 blocks × 3 lanes = 288 次执行**，没有以少运行 blocks 获得性能差异。

### 功能回归与操作入口

| 最终信号版本回归 | 结果 |
| ---------------- | ---- |
| 完整 A5sim HBG | 139 passed，391 deselected，1 个既有 warning |
| 相关 C++ | 23/23 个测试程序通过，bootstrap 包含 15 个用例 |
| 完整 Python UT | 7560 passed，92 skipped；包含新增性能工具 20 个单测 |
| A5 原样 Graph/L4 | 9 类 × 3 轮，27 份原生协议审计通过 |
| A5 相邻参数边界 | 16 场景 × 3 轮，48 份数值/alias/协议通过 |
| A5 调度矩阵 | 188 场景，612 次调用通过 |
| A5 宽 fan-in、双 slot | 6 场景 × 3 轮；双 slot 2 场景/12 次调用通过 |
| A5 实际故障及重建 | 11 场景、33 次正常/失败/恢复调用通过 |

错误用例中的预期 timeout、occupied claim、失败 Worker 拒绝继续提交不是性能样本，也不是正常运行错误。详细首现场和回收边界仍以[专项记录](a5-hbg-aicore-graph-p3-validation.md)为准。

性能工具放在 `tests/fdwic/pa_hbg/spmd/benchmark_graph.py` 和 `analyze_graph_benchmark.py`，后者复用原 CSA 的严格 STRACE 解析；门禁与分位数/错误输入有 20 项 Python 单测。工具可以从另一 checkout 的 venv 调用，但会验证当前工作目录、实际 editable source 和 binding 提交一致。

```bash
# 在对应 checkout 根目录，加载本机 CANN 并完成架构/设备检查后运行。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 SIMPLER_HOST_STRACE=1 \
  .venv/bin/python tests/fdwic/pa_hbg/spmd/benchmark_graph.py \
  --workload graph_cv --expected-backend resident --rounds 105 \
  --output outputs/new-run/b0-graph_cv-native.json

# baseline Graph 使用 legacy-graph；无 Graph 的 baseline 仍使用 resident。
# 文件名和目录必须是新的；输入数据和旧捕获不会被覆盖。
.venv/bin/python tests/fdwic/pa_hbg/spmd/analyze_graph_benchmark.py outputs/new-run
```

本次保留首版“全部实例物化后执行”的明确折中，以及 Host 元数据发布成本。后续若优化，应分别针对 Host 发布和 Device 启动，以本页同负载基线衡量；增量物化必须重新做进度与生命周期证明，不能仅移除启动屏障。本次没有改变业务负载、核数/领取资格或数值容差，也没有引入行为开关。

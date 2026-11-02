# CSA 从约 1.89 ms 到约 2.55 ms 的复核

日期：2026-11-02。对象：simpler A5 `host_build_graph` DeepSeek V4 Pro CSA，固定 B4/S2、seed 0。

本文记录 ordinary pool 回退之前的调查。随后按用户要求完成的回退和 CSA/SWA L1 对照，
见[回退结果](2026-11-a5-swa-ordinary-pool.md#后续ordinary-pool-回退的-csa--swa-l1-对照)。

历史 **1.89 ms 确实存在，且也是完整 Device/RunWall**。此前只引用 README
较早的 2.084 ms 记录不完整。本次重建历史源码并进行同卡交错测试，确认差距同时包含
环境/构建组合变化和运行时源码变化；没有证据将全部差距归因于某一个提交或 PMU 分支。

## 历史记录与计时边界

- [2026-10-31 bounded-context 原始汇总](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-10-31-bounded-context/summary.json)：候选合并 200 个样本，中位数 **1890.006 us**。
- [2026-10-30 earlier-entry 原始汇总](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-10-30-earlier-entry/summary.json)：保留方案 **1893.461 us**，对应可获取的完整提交 `5274c328d60c1157ea54169a3cb05901bfeb94a9`。
- 前者包含当时未提交的候选变化；本次选择后者的精确提交重建，没有把两份源码当成同一份。
- 计时来自 AICPU `AicpuPhase::RunWall`，覆盖设备端启动、调度、kernel 和退出。
  历史到当前没有改成仅计算 kernel span。Host 计时单独列出，不能把两者混用。

历史记录为 Ascend950PR_958b、CANN `9.1.0-weekly-20260708`、PTO-ISA `418de18d`；
本机为 Ascend950DT_9581、CANN `9.3.0`、PTO-ISA `217bb989`，均为 32 AIC / 64 AIV。
历史环境信息来自 [CSA poursoul 原始记录](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-10-27-csa-poursoul/comparison.json)，
其中的 54 个业务 kernel 指纹与 10 月 31 日记录一致。

## 控制条件

- 原始冻结数据共 58 个文件，全部大小和 SHA-256 与原 manifest 相符，未重新生成 golden。
  manifest SHA-256 为 `d1511befaae48a3b692a723320a076064e2ebb1bced193175866c692f9909b45`。
- 每一组比较始终持有同一张卡的 `task-submit` 独占锁；使用交错顺序，预编译后再占卡。
- 每批先跑三轮、每轮七个输出校验，再做五次 warmup 和 100 次正式计时。
  每个版本两批、共 200 个样本；不剔除离群点。正式计时关闭 swimlane 和 PMU。
- 所有重建版本统一使用本机 CANN、PTO-ISA 和已有的 `-fno-jump-tables` 编译修复。
  历史源码的 PTO-ISA pin 被显式替换为当前 pin，这是控制变量，不能声称复现了旧环境。
- 三组对照中，各版本实际上传的 54 个业务 kernel 的长度与 FNV-1a 指纹全部一致。
  历史归档与本机重建的 kernel 只有 1/54 指纹一致，不能把跨环境结果当成相同机器码。
- 每次执行恢复固定 INOUT 状态；这些数字不是整模型逐 token 的连续解码延迟。

## 结果

以下均为合并 200 个正式样本的中位数。不同表之间不能当成一次同卡实验。
Host 样本受共享主机调度影响较大，本次回归归因使用 Device。

### 历史源码 / 当前源码：设备 1，H/C/C/H

| 版本 | 提交 | Device / us | Host / us |
| ---- | ---- | ----------: | --------: |
| H：历史保留源码，本机重建 | `5274c328d` | 2205.280 | 24327.750 |
| C：当前源码 | `4ca09a7d2` | 2551.858 | 22857.345 |

H 两批为 2204.242 / 2205.482 us；C 两批为 2547.9965 / 2554.3775 us。

由此可以分开观察两段差距：

1. 历史源码从旧记录 1893.461 us 到本机重建 2205.280 us，增加 **311.819 us（16.47%）**。
   这归入硬件、CANN/compiler、PTO-ISA 和编译选项的组合差异；未单独测量各因素。
2. 在本机固定环境、同卡、相同业务 kernel 机器码下，旧源码到当前源码增加
   **346.578 us（15.72%）**。运行时源码差距可复现，不能完全归咎于换环境。

两段百分比的分母不同，不能直接相加。这里只是受控比较的分解，不宣称环境与源码没有交互。

### 近期提交：设备 5，A/B/C/C/B/A

| 版本 | 提交 | Device / us | Host / us |
| ---- | ---- | ----------: | --------: |
| A：近期优化之前 | `1b3208eb5` | 3088.207 | 22145.110 |
| B：idle/context/topology 优化之后 | `c8b1fa12b` | 2502.0925 | 21635.505 |
| C：ordinary pool/startup audit 之后 | `4ca09a7d2`，运行时与 `f0722c567` 相同 | 2556.9575 | 29685.830 |

B 相对 A 降低 **18.98%**。C 相对 B 增加 **54.865 us（2.19%）**，超过 2% 回归阈值。
该增量已定位到 `f0722c567` 提交整体；未把它进一步归因于某一个数据结构或分支。
近期两次修改合计仍比 A 快 17.20%，所以“近期 ordinary pool 导致全部退化”不成立。

这些结果把下一步定位重点指向 `5274c328d` 到 `1b3208eb5` 的较早合并范围。
该范围包含 native Graph 兼容、PMU、资源/协议和生命周期修改，尚未完成逐机制隔离。
H 与 A 不属于同一组同卡测试，不据此报告它们之间的精确增量。

### PMU 关闭路径编译特化：设备 1，P/C/C/P，未采用

实验仅在独立 worktree 中把现有 PMU 开关提前到 dispatch loop 入口，生成
`run_ready_dispatch_loop<WithProfiling, WithPmu>` 两种版本。没有新增公开开关，开启 PMU 时保留原功能。

| 版本 | Device / us | Host / us |
| ---- | ----------: | --------: |
| C：当前源码 | 2547.6485 | 25230.980 |
| P：PMU 编译特化实验 | 2528.8760 | 23573.780 |

变化仅 **-18.7725 us（-0.74%）**。这不足以解释历史源码差距，也不足以在本次调查中
采用增加 dispatch loop 特化实例的改动。不能把它描述成完全无影响，更不能声称 PMU 已被证明是主因。
如后续专门评估这个小幅收益，需更长的重复配对实验和代码体积/指令缓存证据。

## 验证、排除项与限制

- 14 批基准测试全部完成，每批三轮七输出 golden 校验通过。
- H、A、B、P 的严格 L4 以及此前同一当前二进制 C 的严格 L4 均通过：
  79 个逻辑计算任务、2308 个物理执行、96 个 worker，原生 AICore 调度身份与依赖协议成立。
- 当前版与 PMU 实验版各跑一次 PMU 开启检查，七个输出通过；各产出 2308 行 PMU 数据，
  每行 `pmu_total_cycles` 为正。
- H/A 的诊断 trace 中 bootstrap_wait 中位数约 129/127 us、exit_wait 约 19/14 us，
  这些阶段不足以解释较大的源码差距。诊断 run 与无 profiling 计时不是同一次执行，不能拿它们做精确扣减。
- 未运行全仓库测试。没有安装旧 CANN 或切换到旧 PR 硬件，所以无法单独判断硬件、编译器或 PTO-ISA 各自造成多少变化。
- 主工作区运行时代码和三个已测运行时二进制保持不变。PMU 候选未采用；没有生产修复可宣称完成。

## 可复核材料与后续定位入口

保存在仓库中的回退前调查材料：

- [汇总、身份和统计](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-11-02-regression/summary.json)
- [全部正式 Host / Device 样本](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-11-02-regression/samples.csv)
- [未采用的 PMU 实验补丁](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-11-02-regression/pmu-specialization.patch)

本机完整材料保存在项目旁 `../logs/csa-regression-20261102-153519/`：
`comparison.json`、`historical-comparison.json`、`pmu-comparison.json`、构建日志、运行脚本、
`pmu-enabled-validated.json`、`evidence/` 中的原始 result/trace/日志及实际运行时二进制。
`archived-path-map.json` 将原临时 worktree 输出路径映射到归档路径；原 JSON 保留原值。
只清理本次创建的四个临时 worktree，不触碰其他会话或原冻结数据。

后续应优先在已固定的本机工具链下继续隔离 `5274c328d` → `1b3208eb5` 中的调度/资源协议变化，
并单独复核 ordinary-pool 提交的 2.19% 增量。恢复到 1.89 ms 需要进一步实测，不能通过选择旧报告数字宣称达到。

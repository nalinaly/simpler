# CSA 公平竞争约束下的准备重叠与硬件计数器诊断

日期：2026-10-29；角色特化补测：2026-10-31。完整 Device 低于 1.5ms 的目标尚未达成。
保留所有原本合法核及 cluster 的竞争资格；不为小 N 任务预选候选集合、固定亲和或静态分区。
2026-10-29 的候选采用当时的三 lane 合同。2026-10-31 的角色候选采用用户后来确认的
实际资源合同：1C1V 两 lane，1C2V 和含 C 协调者的 mask 6 保留三 lane；固定放置、
必需成员的完整 ARRIVED／RELEASED 和原样 CSA 不变。

## 提前准备 payload：未见收益，撤回

基线为 `6a2db9aea`，AICore 二进制为 `20f84347…`。
候选只对已经拥有的 Mix block 提前准备本 lane 的 payload：自身 ARRIVED 且本地槽空闲后，
先占用一个私有 FILLING 槽，再等待另外两 lane 的 ARRIVED ACK。
仍需三 lane 全部到达后才能发布 READY 和进入 kernel；不提前领取额外 block，不扩大候选额度。
复用 `slot_index` 和旧 generation 保存准备状态，发布前重新核对 epoch、错误、槽身份、
metadata 与 callable；inactive lane 也不绕过原到达条件。

同卡串行 B/A 初筛，每批三轮七项 golden、5 次预热及 100 次正式计时：

| 批次 | 完整 Device 中位数 us |
| ---- | --------------------- |
| 提前准备候选 | 2024.819 |
| 恢复基线 | 2005.381 |

候选在这对样本中慢约 0.97%，不支持保留；没有将有限 B/A 称为稳定负收益或最终验收。
候选 L3 的 kernel 窗口 1912.226us、全员无 kernel 区间 519.654us，
不能用这次较短的插桩窗口推翻无插桩完整 Device 的结果。
t23 同 cluster 组间间隔中位 16.943us，输出扇出段 333.011us，未解决宏观大头。

新增 216 项测试在基线全部 RED，覆盖四种 Mix mask、六个 worker 位置、延迟两种 peer、
记录等级 0/3，以及重复准备只发生一次、ACK 前无 READY、延迟 ACK 后准确执行和完成。
还覆盖 12 类错误或陈旧身份，包括 epoch、slot generation、task/block、槽状态、
cooperative、kernel ID、payload callable、metadata flags/mask/N。
原 28 项测试只调整“私有准备是否算进展”的断言，保留 ACK、READY、完成计数保护；
一项 inline service 测试允许私有 FILLING，但仍禁止多余领取、READY 和 Dummy 物理 trace。

候选 21 个 C++ 目标、3574 项 Python 检查、A5/A5sim 编译通过。
原样 CSA 严格 L4 通过 2308 个执行身份、133 条依赖、46733 个协议事件、96 个 worker，
两轮七项 golden；L3 另验七项 golden、完整身份与无丢失记录。
未执行候选完整 HBG 设备回归和故障矩阵，性能初筛后已撤回源码与策略测试。
恢复基线二进制及 A5sim 后，21 个 C++ 目标通过。

该候选的 L3 Dispatch 只记录成功发布那一轮，较早 poll 内的准备不再完整落在同一阶段中。
因此不能同口径比较 Dispatch 的 Scalar 时长；kernel 区间和完整 Device 仍可独立检查。

## 基线 PMU：Scalar 忙碌不等于有效计算

直接复用本机 CANN 9.1.0 SDK `msprof`，使用 `PipeUtilization`、`task-based` 模式，
对未改动的 `20f84347…` 原生二进制捕获八轮 CSA，56 项 golden 全部通过。
采集前后源码、业务 fixture、编排和三件套身份一致，未观察到其他 A5 CLI 重叠。
这不是正式性能验收，也不是逐阶段或逐 core 的计数器捕获。

取后面三轮硬件启动，避开前五轮：

| 范围 | Scalar 忙碌比例 | ICache miss 请求比例 |
| ---- | --------------- | -------------------- |
| 32 个 AIC 聚合 | 约 80.1%–80.3% | 约 2.4%–2.5% |
| 64 个 AIV 聚合 | 约 85.7%–85.9% | 约 2.7%–2.8% |

数据来自 SDK 的 `MetricSummary`，不是仅从四舍五入的 CSV 读取。
使用 SDK `PmuBeanV6` 解码 16 条 128B 原始 C/V 记录；依据捕获配置的事件选择列表，
将 Scalar cycles／total cycles 和 ICache misses／requests 逐条复算，均与 SQLite 匹配。
最后三轮每轮 AIC 总 miss 约 24.1–24.3 万，AIV 约 54.2–55.2 万，均为该类核之和。

这些计数包含业务 kernel、调度和轮询；Scalar 忙碌可能正是在执行等待循环。
miss 比例是请求比例，不是耗时比例，不能据此认定 2.8% 的时间损失或给出可省 us。
硬件 task ID 是这八次常驻启动的编号，不对应 CSA 的逻辑 t23 等任务。
`V6BlockPmu` 表为空，不能将聚合记录解释为已经掌握了逐核瓶颈或 bank 冲突证据。

## 静态栈与热循环体积：只用于构造可验证假设

复用当前 CMake 的真实 CCEC 编译命令，在独立目录增加 SDK 支持的
`-fstack-size-section -fstack-usage`，未替换加载中的 runtime。
`.su` 报告无记录调度循环的静态栈帧为 AIC 688B、AIV 704B；
无记录 `run_resident_scheduler` 为 1432B、1424B；有记录循环为 832B、816B。
这些是独立函数的静态帧，不能当作总运行时峰值，更不能当作 spill 流量或 DCache miss。
DWARF 的固定 CFA 偏移不是该函数实际帧大小，不用它推算栈占用。

ELF 符号表显示基线无记录循环为 AIC 203436B、AIV 206944B。
大体积和非零取指 miss 支持继续检查热路径编译形态，但尚未证明 ICache 是决定性瓶颈。
下面的模式特化实验据此验证实际收益，不以代码尺寸替代设备计时。

## 编译期模式与角色特化

### 准入模式特化：代码缩小，性能未改善

候选只按本次运行固定的 admission 是否存在选择调度循环，
让编译器消去准入模式中不适用的旧填槽、直接 refill 和 SSBUF 完成路径。
无 admission 的运行仍调用通用入口；不按 task 大小、core 或 cluster 选择竞争成员。
错误检查、依赖、执行许可、三 lane 协议和所有原子操作保持原合同。

| 无记录循环 | 基线 bytes | 特化 bytes | 缩小 |
| ---------- | ---------- | ---------- | ---- |
| AIC | 203436 | 175932 | 13.52% |
| AIV | 206944 | 179028 | 13.49% |

同卡 B/A 初筛，每批三轮七项 golden、5 次预热与 100 次正式样本：
候选完整 Device 中位数为 2064.608us，恢复基线为 2006.061us，候选约慢 2.92%。
没有保留；这里没有把一对样本扩展为严格因果归因，也没有完成五批最终验收。
候选 L3 kernel 窗口 1962.263us、全员空窗 561.826us，
t23 同 cluster 组间间隔中位 17.4815us，输出扇出段 351.645us。
这些独立捕获同样没有显示宏观改善，不支持“热循环缩小必然更快”的推断。

执行身份矩阵从 24 扩至 90 项：三 lane、记录等级 0–4、采样开关和三种运行模式，
另加缺少 admission 时必须在轮询前拒绝的保护；共增加 67 项。
三种模式分别是通用入口无 admission、通用入口有 admission、特化入口有 admission。
普通 N=1 在有 admission 的图中仍需原准入尾部许可，不能误断言为 NO_LOG。
首版夹具未满足 metadata 的多 block 标记合同，修正夹具后通过；不算生产缺陷。
初始 RED 是新模板接口不存在导致编译失败，不伪称旧业务行为不正确。

候选 21 个 C++ 目标、3574 项 Python 检查和 A5/A5sim 编译通过；
严格原生 L4、七项 golden 与独立 L3 身份验证通过，业务 kernel 指纹保持一致。
未执行候选完整 HBG 设备回归或故障矩阵；生产改动与候选测试已撤回，原生二进制恢复基线。
恢复后 A5sim 重新构建，21 个 C++ 目标通过。
补丁、二进制和尺寸表归档为 `idle-admission-mode-*`，不将它们作为已保留的优化成果。

### 固定 C/V 角色特化：初筛无收益，撤回

基线为 `5274c328d`，原生 AICore 为 `d51c2fcd…`。候选只把启动时已验证的不可变 C/V 角色
传入内部调度循环，并向 Ready 轮询、填槽和 GM 组服务传播；默认动态 helper 保留。
角色与拓扑不一致时，在轮询／所有权转移前拒绝；没有在 WEAK `aicore_execute` 中加入
架构预处理分支，没有新环境变量、宏或 CLI，也没有改动竞争成员、业务 kernel 或共享 ABI。

同卡 A/B 初筛，每批三轮七项 golden、5 次预热、100 个正式样本：

| 批次 | 完整 Device 中位数（us） |
| ---- | ------------------------ |
| 固定角色候选 | 1891.690 |
| 恢复基线 | 1890.691 |

差异不到 1us，没有可保留的收益信号。只完成一对初筛，不把它外推为所有角色特化都稳定变慢。
两批没有并行本会话编译、其他测试或设备查询，未观察到其他 A5 CLI 重叠；不是系统级独占证明。
200 个正式样本全部保留，54 个业务 kernel 指纹一致。没有将该候选计作新的性能检查点。

最终 ELF 中，基线两个无记录循环体为 189388／183840B；候选按地址排序的四个循环体为
187040／177340／189664／177160B，声明的模板角色依次为 AIC／AIV／AIC／AIV。
默认动态循环被两个角色实例取代，静态总量接近翻倍；不能只报告变小的一侧。
这些名称是模板角色，不单凭名称认定对应的物理核 PC 归属。全部选中函数通过旧 demo 严格解码器的
完整函数体、机器字和非法指令检查，但不是全 ELF 覆盖，也不是动态 I-cache 或耗时归因。

新增 204 个候选检查：入口 132 项、Mix 72 项；入口／Mix 目标共 261／1420 项通过。
覆盖记录等级与采样、迟到 Ready／EXIT／错误、角色错配、四种 Mix mask、N1／2／65、sync_start，
以及两种入口产生相同执行身份、槽 generation、物理子块和完整关闭。
新接口的初始 RED 是不可编译，不称为旧业务缺陷。
首轮正向区间两臂均漏做 completion flush，32 项全局计数断言失败；补 flush 后保留原断言。
一个角色拒绝区间漏接 cluster 拓扑，进入空轮询后主动中断自己的子进程；补齐拓扑及有限退出保护后通过。
Python 首次命令漏设既有后端禁用变量，误加载不匹配的系统 `torch_npu`，出现 8 个收集错误，
随后进程以 139 退出；原日志保留。补回原环境后 3989 项通过，不将该收集错误归为调度回归，
也不声称已定位退出 139 的全部内部细节。

候选 21 个 C++ 目标、A5/A5sim 构建通过。严格 L4 保留 2308 个执行身份、133 条依赖、
3808 个成员义务、40193 个协议事件及两轮七项 golden；未运行候选完整 HBG 回归或故障矩阵，未补采 L3。
生产候选和候选专用测试均已撤回；原生三件套恢复基线，A5sim 重建，恢复后 21 个 C++ 目标通过。
只恢复本次改动，不撤回前一检查点已保留的当前义务先行规则。

## 可复算证据

[公平竞争候选与基线 PMU 摘要](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-10-28-idle/prepare-arrival-pmu-summary.json)
保留两个候选及各自恢复基线的四批各 100 次完整 Device 样本、业务 kernel 指纹、二进制身份、L3 宏观分段、
L4 验证范围、PMU 原始计数及派生值、静态栈信息和源文件校验和。

以下路径相对仓库根目录：

- `../reports/csa_poursoul_20261027.Osnaam/`：`idle-prepare-arrival-{production,tests}.patch`、
  `runtime-idle-prepare-arrival/` 和 `artifacts/idle-prepare-arrival-*`。
- `../reports/csa_macro_l3_20261029.W2dr2E/`：候选 L3、分析脚本和 `summarize_prepare_pmu.py`。
- `../reports/csa_pmu_20261029.1uXaXD/`：未改动基线的 SDK 原始 PMU 捕获、SQLite 和配置。
- `../reports/csa_stack_20261029.Dfaa9U/`：隔离编译命令、诊断对象、日志和 `.su` 文件。
- `outputs/csa-prepare-arrival-*.log`：失败、修正、通过、计时和恢复检查日志。
- 同一归档根目录的 `idle-admission-mode-{production,tests}.patch`、`runtime-idle-admission-mode/`、
  `artifacts/idle-admission-mode-*`，以及 `outputs/csa-admission-mode-*.log` 与 `*-symbols.txt`。

下一步优先解释并压缩三处宏观成本：t23 同 cluster 的重复周转、中段任务交接空窗、
输出扇出后 pack 的等待；不再考虑缩小合法候选范围的方案。

角色特化的[完整样本与撤回证据](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-10-31-core-role/summary.json)
由 `../reports/csa_macro_l3_20261029.W2dr2E/summarize_core_role.py` 复算。
同一归档根目录的 `idle-core-role-{production,tests}.patch`、`runtime-core-role/`
保留候选；`core-role-restored-state.json` 保存恢复时的源码和二进制身份。
`outputs/csa-core-role-*.log` 保留接口 RED、测试区间修正、Python 收集失败、通过、计时及恢复日志。
未来若扩展特化传播范围，需新的串行路径证据和独立完整 Device 对照，不能根据本次静态尺寸推定收益。

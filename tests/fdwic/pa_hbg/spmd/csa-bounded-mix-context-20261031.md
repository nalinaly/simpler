# CSA Mix：有界调用内复用固定上下文校验

日期：2026-10-31。基线提交为 `7f698d25b`，生产实现与 Graph 集成后的 `47c55ec57` 相同。
完整 Device 对照显示约 0.71% 的局部改善，相关常规、故障和 SPMD 回归通过，保留此改动。
低于 1.5ms 的目标及最终五批门禁均未达成。

## 实现与不变量

一次 `scheduler_service_gm_mix` 可以连续完成退役、领取、发布、到达及准备。
这些步骤反复检查同一组固定绑定：graph、当前 task、Mix 标志、mailbox、cluster 和本 lane 资格。
首次 close 仍进行完整校验；如果 close 结束当前义务，服务立即返回。
否则同一有界调用的后续步骤可复用该绑定。内部没有等待循环或 kernel 调用，
也没有改变当前 task／成员绑定的路径；下一次服务调用重新完整检查。

仅这段调用链使用内部模板参数；其他独立调用默认完整检查。没有新增状态字段、ABI、
环境开关、用户选项或跨调用缓存。每一步仍检查错误状态及当前 run 的 epoch，
phase、descriptor 身份、slot generation、取号、关闭、完成计数和全部必需 ACK 均保持原检查。
epoch 仍是原有 READY 后冻结的配置读取，不宣称增加了跨 run 的并发重置能力。

保持全部合法 cluster 公平竞争，1C1V 使用两 lane、1C2V 使用三 lane；
mask 6 仍包含只协调的 C。全部必需成员 ARRIVED 才执行、全部 RELEASED 才复用。
没有增加在飞 block、提前进入 kernel 或改变 sync_start。

## 完整 Device 对照

原样 B4/S2/seed0、Torch 2.12.0+cpu，54 个业务 kernel、79 个 kernel task、2308 次执行和七项 golden 不变。
A/B/B/A 串行，每批三轮七项 golden，随后 5 次预热和 100 次正式计时：

- A1 候选：1889.0520us；B1 恢复基线：1901.2645us。
- B2 恢复基线：1904.1455us；A2 候选：1892.0285us。
- 每版合并 200 个正式样本：基线 **1903.5215us**，候选 **1890.0060us**。
- 差值 **13.5155us／0.7100%**，两对方向一致；不剔除慢样本。

这是包含启动、调度、kernel 和退出的完整 Device/RunWall，不是 kernel 窗口或 Host 时间。
每次切换同时恢复源码和匹配的三个原生二进制；正式采样不并行编译、测试或设备查询。
四批均未观察到其他 A5 CLI 重叠，但不能据此证明系统级独占。全部样本、分位数、Host 分布、
源码／二进制摘要和 kernel 指纹见[可复算摘要](perf_data/2026-10-31-bounded-context/summary.json)，
由下文报告目录的 `summarize.py` 核对。Host 批次漂移明显，不宣称相同比例的端到端收益。

两份无记录循环按 ELF 地址顺序，大小由 219428／238676B 变为 220056／222504B。
一份略增、一份缩小；不把静态尺寸变化当作动态 I-cache 或耗时证据。

## 泳道交叉检查

候选严格 L4 保持 v8：40193 个事件、96 个 worker、2308 个执行身份、61 个协同任务、
3808 个成员义务、1457 个 block 和 133 条依赖，原有校验器全部通过；两轮七项 golden 通过。

独立 L3 的原始观察并没有显示宏观空窗改善：

- kernel 窗口：基线 1815.046us，候选 1848.544us。
- 全员无业务 kernel 区间总量：527.247us → 547.648us。
- t23 结束至 t56 结束之间的无 kernel 区间：297.203us → 316.443us。

两次独立捕获不能精确归因，但这些结果不支持把完整 Device 的 13.52us 改善说成解决主要空窗。
这里的无 kernel 区间不等于 Scalar 不工作，L3 与无插桩版本也不能相减分摊成本。

同协调核时钟的 L4 中，t23 的 480 次相邻组交接，自己的 kernel 返回至下一组入口的中位数
由 28.681us 变为 27.871us。这包含详细记录开销，不能当作无插桩单次交接成本。
各分量中位数不可相加，也不将跨捕获差值精确归给某条检查或 atomic。

## 回归与测试适配

新增 44 项通用保护在基线通过，覆盖服务各状态、非法上下文，以及健康调用之后的错误／epoch 变化。
另增 42 项候选检查，在四种资源 mask 的 420 个子场景中，确认固定绑定的复用不缓存健康许可。
候选 29 个 C++ 目标及 1434 项 Mix 检查通过；初轮相关 Python 检查为 4392 项。
这些是优化护栏，不把接口或编译变化伪称原有业务缺陷。

完整回归发现两个 Graph 集成后的测试适配问题，先保留失败日志并单独核对：

- 错误身份探针没有传入 graph，生产错误写出会将 public/subtask ID 均写成 UINT32_MAX；
  旧 golden 仍把两者所在索引 33 的 uint64 字当作零填充。候选及恢复基线在 A5/A5sim 单独运行均失败。
  修正完整 expected image，并用 C++ offsetof 断言固定字段位置；不忽略该字或放宽比较。
- Graph 生命周期用例直接注册 slot 0，与共享 Worker 已有的 callable 冲突。
  原基线单独运行通过，预占 slot 0 的单测复现冲突。改用 Worker.register 分配的本地 slot，
  最终只 unregister 自己持有的 handle；保留跨 slot、旧 token、epoch 和输出检查。

新增 54 项测试先失败，修正后相关 390 项通过；修正后的两个设备用例在原基线 A5/A5sim 均通过。
首轮大回归因这两项失败主动中断，不能称为完整通过；真机中断还触发了 resident 发布计数的 teardown 断言，
该日志不覆盖。修正后的最终验证为：

- C++ 29 个目标通过，其中 Mix 目标 1434 项；相关 Python 4446 项通过。
- A5 所选 HBG 常规回归 118 通过、4 跳过、4 未选；resident 故障 58 项和 SPMD 聚合 1 项通过。
- A5sim 常规 96 通过、30 未选；有界故障 46 通过，按已有平台规则排除 12 项不可返回 kernel；SPMD 聚合通过。
- 真机三轮回归前后生产源码和三个运行库摘要相同，未观察到其他 A5 CLI 重叠。
  故障矩阵的干扰中断保护未触发。常规回归另有 8 条 Python 多线程 fork 弃用警告，原日志保留。

## 可复算证据与后续

路径相对仓库根目录：

- `../reports/csa_bounded_context_20261031.2qvC6S/`：生产／测试补丁、候选原生与仿真运行库、
  `analyze.py`、`summarize.py`、`validated-state.json`、基线与候选 L3/L4 分解。
- `../reports/csa_poursoul_20261027.Osnaam/artifacts/idle-bounded-context-{a1,b1,b2,a2}/`：400 个正式样本。
- 同目录 `idle-bounded-context-after-l4/` 及 `idle-handoff-graph-baseline-after-l4/`：两份严格 L4。
- `../reports/csa_macro_l3_20261029.W2dr2E/l3-bounded-context/`：候选 L3；
  同级 `l3-ordinary-pool-graph-baseline/` 为基线。
- `outputs/csa-bounded-context-*.log`：构建、初轮失败、基线核对、测试红绿及后续回归。

生产补丁 SHA256 为 `d42f96e56d36340c295e8feca81cbd59cb97523119e276f93662400254dfb944`，
候选 AICore 为 `b08a719078f83106c4683f9c7f457cb8250d3639daa2224db906caaf87973e66`。
Host/AICPU 未变化。测试适配不改变上述生产补丁或 CSA kernel。

下一项普通 Mix 提前进入 kernel 的实验已单独询问用户；尚未得到答复，未实施。
当前仍完整保留全部 ARRIVED／RELEASED 入口与复用条件，不以本次小幅收益替代主要空窗优化。

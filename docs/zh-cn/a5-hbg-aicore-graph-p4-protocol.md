# A5 HBG AICore Graph：原生协议与泳道验证

日期：2026-10-30。依据：[设计文档](a5-hbg-aicore-graph-design.md)、[实现过程](a5-hbg-aicore-graph-implementation.md)。本页记录以 `b22cbd89a` 为起点的 L4 接入及后续信号兼容约定；最终故障和性能结论分别见[专项验证](a5-hbg-aicore-graph-p3-validation.md)及[性能/资源报告](a5-hbg-aicore-graph-p4-performance.md)，不从 L4 的带插桩时间推断 L0 性能。

## 1. 实现与证据口径

### Host 冻结与身份

- `SchedulerGraphBinding::freeze_protocol` 保存展开的调度元数据、依赖、公共范围、实例到 Definition 的关系；每个不同 Definition 只留一份模板/CSR 快照。不在 Host 物化 descriptor/payload 或绑定子任务参数。
- 复用原 `SchedulerProtocolTaskSpec` 和普通图冻结逻辑；共用 JSON 有界读取/导出工具，不另建一套普通执行协议。快照与可复用的 Host 编排镜像分离，修改原镜像不影响已冻结内容。
- 内部队列仍使用 int32 稠密 ref；公共身份仍是 GLOBAL/SUB_TASK。泳道任务、执行身份和 Scalar 记录使用原公共 TaskId，Graph v8/v9 同时保存 ref、父 ID、子 ID。显示参数另有十六进制 ID，避免只靠超过 JavaScript 精确整数范围的十进制值辨认子任务。
- 独立 Graph 分支原来输出 Graph v8、无 Graph 输出 v7；目标分支另有资源范围 v8。合并版本的 Graph run 输出 `hbg-native-protocol-v9`，无 Graph 保持资源范围 v8。读取器不把这两种 v8 的字段混为一个合同。

| schema 与形态 | Graph 证据 | Mix 资源范围与逐 lane 前驱 |
| ------------- | ---------- | -------------------------- |
| v1-v7 | 无 | 沿各旧版合同 |
| v8，含 `graph` | 必须完整，首错误带父/子身份 | 原 Graph 分支的完整三 lane 资源协议 |
| v8，不含 `graph` | 无，不能报告 Graph verified | 必须按实际资源范围，lane 带 `previous_lane_seq` |
| v9 | 必须完整，不能省略 `graph` | 同时要求资源范围及 `previous_lane_seq` |

以上分支仍进行精确字段集合、身份和因果校验。只删除 Graph v8 的 `graph` 字段会因首错误字段不匹配被拒绝；把普通资源范围 v8 直接改名为 v9 也因缺少 Graph 证据被拒绝。没有对旧文件补写或推测缺失的 lane 前驱。

### 设备有界记录

| 记录 | 写者与时间边界 | 用途 |
| ---- | -------------- | ---- |
| 实例物化 | 本实例唯一 owner；物化前后、发布前后 | 核对 Definition、实例、owner 与物化状态 |
| 构造发布票号 | `remaining_instances` 递减返回的旧值 | 证明各实例恰好贡献一次，最终零值可被观察 |
| worker 启动等待 | 每个 worker 的等待起止、读取次数、最终观察值 | 合并轮询为一段，不逐次写 raw |
| 子任务完成贡献 | 唯一完成者；`remaining_children` 递减前后与旧值 | 证明全部子任务终态汇总到唯一 EXIT |
| ENTER/EXIT/Dummy | 复用已有无 kernel TaskDone 记录 | 不生成伪 kernel 执行记录 |

记录使用原 worker 私有的事件序号；没有新增全局 trace atomic。新事件区只在 Graph + L4 时分配，三种 record 均为独立 128B 行，不与调度控制字混写。L0 不分配这些区、不写这些记录。实例控制为 128B；构造控制现为 256B，其中独立 128B 行保存功能性的物化发布信号，不属于 trace。

启动等待先读取隔离行的 bypass 信号，再以 atomic 确认 `remaining_instances==0`；`remaining_observed` 仍来自实际 atomic，因果证明与旧 v8 一致。`poll_count` 是等待循环的探测次数，不等于 atomic 指令次数。最后完成者的发布 before/after 包含计数递减和发布信号，未增加逐次轮询事件。

设实例数为 I、展开节点数为 R、worker 数为 W，新增 Graph 事件区为 `128 × (I + R + W)` 字节，不随轮询次数增长。2 个 1024-child 实例在 A5 上为 I=2、R=2052、W=96，对应 275,200B；**这只是新增事件区，不是完整 scheduler/heap/retained 分配的总占用**。

### 严格因果审计

审计同时检查冻结定义、设备实际映射、完整 worker 序号、实例终态及 kernel 身份；数值正确或次数相等都不能替代协议证明。

- 同址 atomic 的票号决定发布/贡献顺序，线性化点限制在真实 before/after 事件之间；不靠不同核的 cycle 大小排序。
- 构造 owner 的 `publish_after` 可能晚于其他 worker 观察到零；审计使用发布区间内的点，不错误要求消费者等待这个诊断返回戳。
- 子任务依赖释放先于父计数贡献，后继可能抢先完成。审计不要求 remaining 的票号与子任务拓扑序一致，但要求全部唯一贡献先于 EXIT。
- CSR 与 ENTER gate、公共父子身份、共享 Definition 地址、实际存储范围、owner/epoch、剩余计数和公共完成数必须相符。
- 失败/不完整记录保留原值供诊断，严格审计拒绝通过。Graph v8/v9 首现场同时导出公共父 ID 与子 ID；无 Graph 的旧版本不悄悄增加字段。

## 2. 失败复现与修复过程

1. 接入前，同一 `record_then_replay_2d --enable-chip-swimlane 4` 数值通过，resident 门禁因 `mode=3` 失败。没有先删除回退来制造“原生通过”。
2. 先添加快照、L4-only 布局、真实 bootstrap 事件和导出测试，分别确认缺失能力的失败，再接实现。普通 v7 导出回归同步保留。
3. Python 合成记录用于审计器反例：错父子 ID、错误 CSR/模板、重复/缺失票号、旧 epoch、错 owner、缺事件、提前 EXIT、局部事件顺序矛盾等。它们不替代真机证据。
4. 首份 A5sim/A5 原生 v8 都通过数值、fresh resident 与协议审计。随后检查发现包装器强制 3 轮时沿用同一输出目录；新增 4 项反例确认覆盖与弱门禁，再改为逐轮独立路径并强制审计。
5. 最终每个场景前两轮写入 `validation-round-1/`、`validation-round-2/`，末轮保留标准根目录；写入前拒绝已有 raw。计算 kernel、输入形状、原 N 和 golden 未改。
6. predicate 的非执行分支在 Worker View 改为 `predicate_skip` 控制标记；Scheduler View 保留真实控制区间，kernel duration 标为 0。另从已审计的冻结映射恢复 kernel ID，可配合既有 name map 显示函数名，不必猜测或额外运行 dep_gen 才能辨认 kernel 编号。

本 worktree 原先只构建 A5/A5sim，全量 Python 单测中 5 项因缺少 A2/A3 sim 的 HBG/TMR 动态库失败；日志明确是 `Pre-built runtime binaries not found`，包括 3 个本地测试 daemon 场景和 2 个 ABI 场景。已在自己的 `.venv` 中补建 `build_package_a2a3sim` 后复跑，没有修改这些测试或借用其他 worktree 的二进制。

## 3. 验证结果与复现

### 原样 L4 矩阵

每项在 A5sim/A5 **各 3 轮**；每轮先验数值与新 resident publication，再由真实 raw 完整审计。合计每平台 27 份、两平台 54 份独立捕获通过。下表的执行数量来自 A5 实际记录，而不是按模板推算。

| 场景 | 实例数 | 定义数 | Graph 子任务数 | 实际 kernel 次数 | predicate skip |
| ---- | ------ | ------ | -------------- | ---------------- | -------------- |
| 原 1D Graph | 3 | 2 | 18 | 13 | 0 |
| 原 2D Graph | 3 | 2 | 18 | 13 | 0 |
| 原 AIC/AIV DAG | 3 | 1 | 9 | 9 | 0 |
| 原 Mix/SPMD Graph | 3 | 1 | 3 | 288 | 0 |
| 原 sync-start Graph | 1 | 1 | 2 | 106 | 0 |
| 原 predicate replay | 4 | 1 | 20 | 13 | 7 |
| 无边界、2 个 Dummy | 1 | 1 | 2 | 0 | 0 |
| 无边界、多实例 Dummy | 16 | 1 | 128 | 0 | 0 |
| 无边界、最大 body | 2 | 1 | 2048 | 0 | 0 |

kernel 次数包含普通外部任务及各 block/lane 的实际执行；子任务数是 Graph 内逻辑节点数，两者不能直接相减归因。predicate 的 20 条 dispatch 身份中有 7 条 `executed=false`，不能统计成 20 次 kernel。

### 相关回归

| 检查 | 结果 |
| ---- | ---- |
| Graph Host binding / bootstrap / completion services | 14 / 12 / 13 项 PASS |
| A5 C++ 相关测试可执行文件 | 23/23 PASS，包括普通 ready/dispatch、SPMD、Mix、admission、协议导出和错误身份 |
| 协议与泳道 Python 重点回归 | 显示与身份补充后 1179 passed、8 skipped；覆盖新 Graph、旧协议、执行身份、converter 和 TaskId 合同 |
| 完整 A5sim HBG 场景 | 127 passed、391 deselected、1 个既有 `load_module` warning |
| 全量 Python 单测 | 补齐 A2/A3 sim 构建后 7527 passed、105 skipped；skipped 不计入通过 |
| 无 Graph 的 A5 L4 | `ordinary_mix_interleaved` 数值与原生审计通过；仍为 v7，1992 个事件、32 条执行记录；没有虚假的 Graph verified 标记 |

构建和原样矩阵命令：

```bash
source .venv/bin/activate
source <CANN_ROOT>/set_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 CMAKE_BUILD_PARALLEL_LEVEL=4 MAX_JOBS=4 \
  python -m pip install --no-build-isolation --no-deps -e . \
  --config-settings=build.targets=build_package_a5sim \
  --config-settings=build.targets=build_package_a5

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python tests/st/a5/host_build_graph/graph_execution/test_graph_execution_resident.py \
  -p a5sim --manual include --enable-chip-swimlane 4

# 上卡前确认 A5 和设备空闲，设置新的 ASCEND_PROCESS_LOG_PATH。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python tests/st/a5/host_build_graph/graph_execution/test_graph_execution_resident.py \
  -p a5 -d 0 --manual include --enable-chip-swimlane 4
```

本机 `npu-smi`/`task-submit` 缺失，预检脚本无法识别；按已获允许的单卡方式，改用 ACL 确认 `Ascend950PR_958b` 及运行前 Cube/Vector/AICPU 利用率为零。没有声称获得排他锁，本轮只作为功能与记录正确性证据。

## 4. 本地泳道图与如何阅读

A5 矩阵日志目录：`outputs/aicore-graph-p4-matrix-a5.hvs3jw/`。每个 case 的实际 raw 与图放在下表目录中，最终展示文件统一为 `native_graph_v8_swimlane.json`；原 `merged_swimlane.json`、raw 和前两轮子目录都保留，未覆盖。全部生成物均在 Git 忽略目录，不提交泳道图/raw。

| 场景 | `outputs/` 下的目录 |
| ---- | ------------------- |
| 1D | `TestGraphExecutionResident_record_then_replay_1d_20261030_195825/` |
| 2D | `TestGraphExecutionResident_record_then_replay_2d_20261030_195826/` |
| AIC/AIV | `TestGraphExecutionAicAivResident_record_then_replay_aic_aiv_20261030_195832/` |
| Mix/SPMD | `TestGraphMixResident_record_then_replay_mix_spmd_20261030_195838/` |
| sync-start | `TestGraphSyncResident_Case1_20261030_195845/` |
| predicate | `TestGraphPredicateResident_PredicateAcrossGraphReplays_20261030_195848/` |
| 2 Dummy | `TestGraphNoBoundaryResident_instances1_children2_20261030_195853/` |
| 16 实例 | `TestGraphNoBoundaryResident_instances16_children8_20261030_195853/` |
| 2048 Dummy | `TestGraphNoBoundaryResident_instances2_children1024_20261030_195853/` |

- **Graph Execution**：从真实物化开始到 EXIT 完成的父区间，并含物化、ENTER、EXIT 子区间；零 kernel 的 Graph 也可见。
- **AICore Scheduler / Graph detail (same Scalar)**：物化、发布、启动等待、Dummy、predicate skip、完成贡献。它是同一个 Scalar 的细节轨道，不是额外并行线程；等待同时显示精确 poll 次数。
- **Worker View / Scheduler View**：保留真实子任务执行/控制区间，显示 `g<parent>t<local>`、block、lane、slot、generation；父子 ID 和 dense ref 不混用。
- **Graph 因果 flow**：ENTER 到 root、最后贡献到 EXIT，依据已经审计的协议关系生成。没有 dep_gen 时仍有这些边界 flow；完整任意任务依赖箭头仍需 dep_gen。
- 不用笼统的 `Scalar.poll` 填补没有采集证据的间隙，也不把整段 Graph 生命周期当成某个 Scalar 持续忙碌。

最大 Dummy 场景的 A5 raw 为 2,629,471B，约 2.51 MiB；其 8400 个有效序号包含控制服务和有界 Graph 事件，没有逐次 poll 展开。这不是完整运行内存占用。

## 5. 版本与后续验证

本次 A5 验证二进制 SHA256：

```text
aicore_kernel.o     a66469036abf694c98294bdc58b8d625bdb8061c235875b3d803e50005d29f82
libaicpu_kernel.so  396de9b49313372adc2183e45edc19ab288b099300f31a04ea6bdbaba0dadd89
libhost_runtime.so d936ac420b06b4930a082ec0b75382132352f528425dc902ff0e9630dba098ae
```

本页初次接入 L4 时，真实故障、参数边界和完整性能资源仍未验收；后续已完成设计范围内的专项与对照，结果见页首两个链接。L4 只证明协议和捕获矩阵，不自动证明端到端性能收益；最终报告仍保留首版启动成本、Host 回退及测量限制。仅作阶段性本地 commit，不主动 push。

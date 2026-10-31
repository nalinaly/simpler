# A5 HBG AICore Graph 实现过程

设计依据：[AICore Graph 支持设计](a5-hbg-aicore-graph-design.md)。起始提交 `02335c0f46d6f12a0fd0adaeccf4869b1164ff64`，工作分支 `feat/a5-hbg-aicore-graph`。

## 实施状态

用户于 2026-10-30 要求实现设计并记录过程。设计范围内的首版已经实现，功能、协议、资源和同负载性能报告已交付；首版仍为单层 Graph、启动前完成全部物化，不新增嵌套 Graph/deferred completion 或多 block predicate。P5 只合入有证据的启动长尾修复，不承诺普遍性能提升。本文件记录已发生的工作，不把计划写成结果。

| 阶段 | 当前状态 | 完成证据 |
| ---- | -------- | -------- |
| P0 合同/来源门禁 | 已完成 | 基线数值通过而 resident 来源失败；当前 22 项门禁单测通过，包含逐轮独立采集与原生 L4 强制审计 |
| P1 数据/引用/布局 | 已完成 | GM 载体、实例/引用布局、Scalar 物化和启动发布已接线；framing 跨编译器差异已修复，A5 复测通过 |
| P2 最小 native Graph | 已完成 | 原样 1D、2D、AIC+AIV Graph 在 A5sim/A5 上通过；真机各 3 轮，数值与 fresh resident 来源均通过；扩展边界见 P3 |
| P3 能力与故障矩阵 | 核心功能、有界设备故障及参数发布边界已验证 | 188 场景/宽 fanin、epoch/owner、双 slot；新增每平台 11 个实际故障/恢复用例和 48 轮参数/alias；详见[专项记录](a5-hbg-aicore-graph-p3-validation.md) |
| P4 协议/泳道/性能资源 | 已交付，测量限制明确保留 | v8、父子身份及原生 L4；3150 次最终 A/B、150 次 CSA 短批复核、PA 真机及精确 GM 预算；详见[性能/资源记录](a5-hbg-aicore-graph-p4-performance.md) |
| P5 有证据的优化 | 独立发布信号修复通过 | 启动热轮询与 atomic 计数分离，保留原子最终确认；最终 900 个原生 Graph 稳定态样本未复现 >5ms 长尾；不继续扩展增量物化 |

## 2026-10-30：隔离构建与原样门禁

- 核对 worktree 只包含前一轮设计文档和导航变更，没有其他 session 的 runtime 修改。
- 在本 worktree 的 `.venv` 内安装依赖，由当前源码构建 `build_package_a5sim`；没有复制其他目录的 runtime 二进制。构建完成。
- 新增 `tests/st/a5/host_build_graph/graph_execution/test_graph_execution_resident.py`：继承原 Graph 的 CALLABLE、CASES、计算 kernel、输入和 golden；逐轮先做数值比较，再检查新发布的 resident 模式；至少运行三轮，拒绝 skip-golden。
- 新增 `tests/ut/py/test_hbg_graph_resident_gate.py`，验证来源门禁不能被正确数值、陈旧 mode、跳过 golden 绕过；此单测只验证验收器，不证明设备 Graph 已支持。

日志保存在本次构建/测试输出中；生成二进制、日志和泳道图不提交到 Git。未授权 push。

### 门禁的实际结果

| 检查 | 结果 | 说明 |
| ---- | ---- | ---- |
| 原 `TestGraphExecutionHostBuildGraphA5::record_then_replay_2d`，A5sim，3 轮 | PASS | 原 kernel、输入、golden 和计算负载未改 |
| 新 `TestGraphExecutionResident::record_then_replay_2d`，A5sim | 预期失败 | 数值比较先通过，随后报告 `mode=3`，即整轮 AICPU Graph 回退 |
| Python resident 门禁 | 11 passed | 包括陈旧 publication、fallback mode、数值错误和 skip-golden 反例 |

复现命令（在本 worktree 根目录、已安装当前 A5sim runtime 的 `.venv` 中）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tests/st/a5/host_build_graph/graph_execution/test_graph_execution.py \
  -p a5sim --case TestGraphExecutionHostBuildGraphA5::record_then_replay_2d --rounds 3

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tests/st/a5/host_build_graph/graph_execution/test_graph_execution_resident.py \
  -p a5sim --case TestGraphExecutionResident::record_then_replay_2d

TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python -m pytest \
  tests/ut/py/test_hbg_graph_resident_gate.py -q
```

第二条在基线阶段按预期失败；后续生产接线后已用同一命令转绿。门禁没有改成接受 mode 3/4，也没有跳过 golden。

## 2026-10-30：稠密引用、共享读取器和父完成服务

### 已实现内容

- Host `SchedulerGraphPlan` 按顶层提交生成范围，普通任务占 1 个 ref，Graph 占 ENTER、N 个子任务和 EXIT。Host 规划只存每个公共任务的范围，不为每个 worker 建一份展开表。
- `SchedulerGraphView` 增加独立 `node_map_address`，原 `reserved` 必须为零的约束不变。未映射的普通 DAG 继续按原 storage stride 寻址。
- 映射项保存真实 descriptor/payload 的地址、公共 TaskId、fanin 来源和所属实例控制区。shape 校验比较完整公共身份，不把 SUB_TASK 的 local_id 当全局 ref。
- ready 注册、完成唤醒和 Host 协议快照统一读取 fanin；不再因子任务 payload 的 fanin 为零而误当 root。内部宽 fanin 上限与顶层 128 的限制分开。
- 内部非 root 可直接引用共享 Definition 的 uint16 CSR，读取时加实例的子任务 ref 基址；root 的 ENTER gate 是映射项内的单个 ref。不为每个实例复制一套转换后的内部边数组，普通外部依赖仍可使用 int32 稠密引用。
- EXIT 从 bootstrap root 中排除。完成者在原 wake-list 唯一关闭后贡献一次实例计数；只有最后一个子任务把 EXIT 放入已有 inline 服务队列，不递归执行，不构造超长 fanin。
- 实例计数占独立 128B 行，设备只通过既有原子原语更新。重复完成由原 task 状态/wake-list 所有权拒绝；错误实例身份和计数下溢明确报错。

基础阶段仅验证 helper，未发布新地址。后续接线将视图扩展为 48B（增加公共完成控制地址），映射项仍为 48B、实例控制为 128B；生产接线和真机状态见下节。

Definition 的公共 POD 类型和原有 Host 数组读取器抽到 `src/common/host_build_graph/graph_definition.h`。`GraphDefinition` 仍为 104B，`SubTaskDefinition` 仍为 80B，scalar 来源仍为 4B；旧 `graph_execution.h` 引用同一份定义，不维护第二份 wire schema。AICore 的 `scheduler_graph_definition.h` 读取对齐的 run-owned GM 对象，检查 framing、对象内范围和溢出；它不接收或保存其他 AICPU 线程的 launch-package 指针。

### 测试证据

| 测试组 | 数量/结果 | 覆盖 |
| ------ | --------- | ---- |
| `test_a5_hbg_graph_plan` | 9 项 PASS | 多父实例同 local_id、GLOBAL→EXIT、内部边转换、1024 子任务、空 run 和非法范围 |
| `test_a5_hbg_graph_view` | 8 项 PASS | 非连续 storage、公共身份错绑、独立 CSR、宽 fanin、服务节点无 payload、实际参数池、共享 CSR 的实例基址、无边分配的 root gate |
| `test_a5_hbg_graph_services` | 9 项 PASS | ENTER 外部 gate、全部子任务完成后唯一 EXIT、独立 Graph 重叠、重复完成、下溢、1024 子任务、两个 worker 并发最后贡献 32 轮、串实例错误、共享 CSR 菱形的逐前驱等待迁移 |
| `test_a5_hbg_graph_definition` | 3 项 PASS | 公共字段/大小保持不变、scalar 来源、Host 数组边界 |
| `test_a5_hbg_graph_definition_view` | 4 项 PASS | GM 原位读取、framing/key、未对齐/溢出、跨对象越界和失败时不改目标 |
| 原 resident 回归 | 7 个测试可执行文件 PASS | scheduler contracts、protocol export、ordered execution、GM Mix、inline services、error identity、scheduler bootstrap |
| 共享 Definition 拆分回归 | 6 个旧测试可执行文件 PASS | A2/A3 Definition arena、recording bounds、cache、boundary match、activation，以及 A5 activation |

服务测试在接入完成聚合前已执行：5 项全部按预期失败，表现为 EXIT 被误当零 fanin root、子完成不递减父计数。接入后转绿，再补 1024 子任务、并发最终贡献和错误实例测试。以上是真实共享 helper 的 CPU 执行证据，**不是 A5 cache 发布链的上卡验证**。

```bash
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build --target \
  test_a5_hbg_graph_plan test_a5_hbg_graph_view test_a5_hbg_graph_services \
  test_a5_hbg_graph_definition test_a5_hbg_graph_definition_view -j4
ctest --test-dir tests/ut/cpp/build \
  -R 'test_a5_hbg_graph_(plan|view|services|definition|definition_view)$' --output-on-failure
```

A5sim 和 A5 onboard runtime 已在自己的 worktree 构建通过。另用动态映射地址的编译探针直接实例化 GM Definition 读取、scalar 来源 accessor、公共身份校验和子完成聚合，`dav-c310-cube`/`dav-c310-vec` 两种 CCEC 目标均通过；避免仅因生产 `node_map_address=0` 被编译器裁掉而漏查新分支。编译通过仍不代表真机执行/缓存一致性通过。

最终源码重建后的 A5sim 场景回归：原 Graph replay、Graph Mix/SPMD、Graph predicate、Graph sync-start 和普通 single-block Mix 等默认选定场景为 `6 passed, 2 deselected`；两个 deselected 来自原有 `manual=[a5sim]` 标记。随后对 Graph AIC/AIV 和 vector example 使用 `--manual include` 单独运行，`2 passed`，合计 8 个原场景通过。严格 native Graph 用例再次在数值比较后因 `mode=3` 失败，说明尚未接通的生产路径没有被基础测试掩盖。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest \
  tests/st/a5/host_build_graph/graph_execution/test_graph_execution.py \
  tests/st/a5/host_build_graph/graph_execution/test_graph_execution_mix_spmd.py \
  tests/st/a5/host_build_graph/graph_predicated_dispatch \
  tests/st/a5/host_build_graph/graph_sync_start_early_dispatch \
  tests/st/a5/host_build_graph/single_block_mix \
  --platform a5sim --device 0-1 --max-parallel 2 --pto-session-timeout 600 -q

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest \
  tests/st/a5/host_build_graph/graph_execution/test_graph_execution_aic_aiv.py \
  tests/st/a5/host_build_graph/vector_example \
  --platform a5sim --device 0-1 --max-parallel 2 --pto-session-timeout 600 --manual include -q
```

基础阶段的 Markdown、Python ruff、按仓库配置的 C++ lint、版权头和 `git diff --check` 检查通过；该阶段没有单独 commit/push。后续真机验收见下节。

## 2026-10-30：原生接线与真机首测

### 生产路径接线

- Host 打包 Definition 后先做 backend 能力选择。native Graph 将一份共享 Definition block 追加到既有 scheduler-state 分配；legacy 仍使用 AICPU launch section，不把 Definition 同时发两份。
- 新增 `SchedulerGraphBinding`，将顶层提交展开为稠密节点，复用原 heap-tail 规划，并从已压紧/重定位的 shared-memory image 取得本次边界地址。所有乘法、对齐和总长度使用 checked layout；参数及输出不在 Host 构造为子任务。
- worker context 在已有配置行的未使用空间中增加映射、实例和构造控制地址，`bootstrap_task_count` 的 384B 偏移不变。无 Graph run 的地址/数量为零，不分配上述 Graph 专属区。
- Scalar 动态领取实例，一个实例一个临时 owner；在原存储内写 descriptor/payload、tensor/scalar pool，进行范围写回和 barrier 后再发布物化完成。所有实例完成后才开放现有 bootstrap/ready 路径。
- 内部退休数仍覆盖 ENTER/子任务/EXIT；独立公共完成计数只由普通顶层任务和 EXIT 贡献，Host 已完成节点计入初值。退出前核对公共完成总数。
- 本阶段仅开放单 block、单 lane、无 predicate 的 Graph 子集。Mix、多 block、sync-start、predicate 以及需要新版 Graph L4 证据的 run 在这一阶段仍整轮 legacy，后续逐项验收后开放；这不是最终范围缩减。

新增物化测试 6 项，验证边界/内部 heap 重绑定、固定与继承 scalar、重复实例化刷新、非法参数/存储拒绝及 predicate 不误变为无条件执行，并用同一份 Definition 对照实际 legacy materializer 的字段结果。binding 测试 6 项验证 Definition 共享、两实例地址隔离、GLOBAL producer → EXIT、非法边、predicate 元数据矛盾、分配溢出和映射查询越界；公共完成计数另有专门测试。连同前节，当前 Graph 基础 C++ 测试为 46 项。

### A5sim 原样门禁转绿

| 用例 | 轮次 | 数值 | resident 来源 |
| ---- | ---- | ---- | ------------- |
| `TestGraphExecutionResident::record_then_replay_2d` | 3 | PASS | 每轮新 publication，mode 1/2，通过 |
| `TestGraphExecutionResident::record_then_replay_1d` | 3 | PASS | 每轮新 publication，mode 1/2，通过 |

1D 在 A5sim 上原本标为 manual，命令需额外传 `--manual include`；没有修改原用例的负载、输入、golden 或容差。

### 真机首测：framing 跨编译器布局差异

本机缺少 `npu-smi` 和 `task-submit`，架构预检查脚本无法检测，改用已安装 ACL 做只读查询：SoC 为 `Ascend950PR_958b`，设备数 1；运行前 Cube/Vector/AICPU 利用率均为 0。按用户允许的直接单卡方式执行，未获得队列锁；本次仅是功能验证，不把时延作为性能结果。

首次真机日志保存在本地未提交的 `outputs/aicore-graph-native-a5.sfnYJp/`。失败发生在 bootstrap 的 Scalar 物化：`task=9 status=6 site=132`，任务执行尚未开始；Host 的 `507018` 是此初始化失败的外层错误。runtime 的故障退出执行了既有 reset 并确认设备已恢复，没有手动修改设备配置。

用 CCEC 编译期断言复现原因：`GRAPH_DEFINITION_OBJECT_ALIGN = alignof(std::max_align_t)` 在 Host/AICPU 为 16，在 CCEC 为 8，导致同名 framing header 分别为 32B/24B。Scalar 从 image 前倒退 24B，读到了错误的 magic 位置；CPU 仿真无法暴露这一差异。

修复将 framing 固定为 16B 对齐、32B header，并加入各编译目标都会检查的 `sizeof == 32` 断言。Host/AICPU 的原编码不变；CCEC 布局断言已经由红转绿。

### 修复后的真机结果

| 场景 | 轮次 | 数值和逐轮 resident 门禁 | 本地日志目录 |
| ---- | ---- | ------------------------ | ------------ |
| 原 Graph 2D | 3 | PASS | `outputs/aicore-graph-framing-a5.jbiLUW/` |
| 原 Graph 1D | 3 | PASS | `outputs/aicore-graph-pure-cv-a5.HJTaic/` |
| 原 AIC/AIV Graph DAG | 3 | PASS | `outputs/aicore-graph-pure-cv-a5.HJTaic/` |

`test_graph_execution_resident.py` 复用一个来源门禁 mixin，分别继承原 Graph 和原 AIC/AIV Graph 的 CASES、CALLABLE、输入和 golden，不复制或缩减计算 kernel。对应 pytest 来源门禁单测为 `12 passed`。原 1D/2D 类的默认 scheduler marker 从 legacy 更新为 resident；未开放的组合仍由原 legacy 用例验证。A5sim 新旧最小 Graph 测试为 `3 passed`（pytest 条目），保留的 Graph Mix/SPMD、predicate、sync-start 与普通 Mix 回归为 `5 passed`。

真机复现命令，需先加载本机 CANN 环境并给 `ASCEND_PROCESS_LOG_PATH` 设置新的日志目录：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tests/st/a5/host_build_graph/graph_execution/test_graph_execution_resident.py \
  -p a5 -d 0 \
  --case TestGraphExecutionResident::record_then_replay_1d \
  --case TestGraphExecutionResident::record_then_replay_2d \
  --case TestGraphExecutionAicAivResident::record_then_replay_aic_aiv
```

本次仅证明最小原生功能链路，日志中的运行时间不作为 Graph 性能结论；尚未完成同负载 legacy/native A/B、完整 L4 及全部 P3 能力。

最终构建再次把上述三个真机场景各跑 3 轮，全部通过，保存在独立目录 `outputs/aicore-graph-p2-final-a5.CzvoCO/`，没有覆盖前面的失败或成功日志。扩展 C++ 回归为 29/29 个可执行文件通过，包含 A2/A3 旧 Graph 回归与 A5 普通 ready/dispatch、Mix、SPMD、协同准入等路径。

完整 A5sim HBG 回归命令及结果：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest examples tests/st --platform a5sim --runtime host_build_graph \
  --device 0-1 --max-parallel 2 --pto-session-timeout 600 --exclude-level 4 -q
```

结果为 `115 passed, 391 deselected, 1 warning`，耗时约 251s。deselected 包括平台、级别和原 manual 过滤，不计入通过；warning 为既有 Python `load_module()` 弃用提示。此组结果是功能回归，不是性能样本。

提交检查发现两处 Host `optional` 查询未经显式检查。补充越界查询反例后复现为错误地返回 `UNSUPPORTED_SHAPE`，已改为 `INVALID_TASK_ID`；重新构建后的针对性场景为 `8 passed`，三个真机场景又各 3 轮通过，保存在 `outputs/aicore-graph-checked-a5.uW4M8U/`。全部 pre-commit 检查通过，没有跳过 hook。

此版本的 A5 onboard 二进制 SHA256：

```text
aicore_kernel.o     d4eae306584d5db98a28bd740814d132238789933254c82513f414993f6f44dc
libaicpu_kernel.so  fa0bec99a948e2c4a44516a3e76934105f6947116ab760c572642856cb01f1e6
libhost_runtime.so a51d5fb5fccd02c78188aaeb364c5b87f63aa26cf72ece6a35a02134129bf0bb
```

## 2026-10-30：能力扩展、故障检查与生命周期

详细实现、红绿证据、命令和日志目录见[能力与故障验证记录](a5-hbg-aicore-graph-p3-validation.md)。

已开放多 block/Mix/sync-start/单 block predicate；188 个执行场景、6 个宽 fanin 场景在 A5sim/A5 均通过。后续补齐旧 epoch、owner 唯一性、错位控制地址、父子首错误身份，以及空边界、超容量拒绝和双 slot prepare/finalize 复用。相关 C++ 为 20/20 个可执行文件通过，完整 A5sim HBG 为 127 passed。

该阶段尚缺设备实际故障/回收、缓存边界专项和完整性能资源验收；前两项的后续验证见下节，性能资源验收仍未完成。不主动 push。

### 原生协议与父子泳道

已接入 Graph 专属 v8 协议和真实 Scalar 事件，9 类原样场景在 A5sim/A5 各运行 3 轮，54 份 raw 逐轮通过严格审计；最后一轮还生成标准泳道图。真实子任务显示公共父/子身份，ENTER/EXIT/Dummy 是控制服务，predicate skip 不是 kernel span。普通图继续输出 v7。

采集包装器不再将三轮写入同一个 raw：前两轮放入独立子目录，末轮保留标准目录，且已有文件拒绝覆盖。完整 A5sim HBG 回归为 127 passed；23 个相关 C++ 测试可执行文件通过。实现、失败复现、全量 Python 检查及本地泳道图路径见[协议与泳道验证记录](a5-hbg-aicore-graph-p4-protocol.md)。

### 实际故障和相邻实例发布

补齐 Scalar 物化失败的具体 child 定位；相应 C++ 反例先失败后转绿。A5sim/A5 各通过 11 个实际故障/恢复用例（每用例正常、失败、重建后正常三次调用），覆盖物化失败和纯 C/V、Mix 的有界子任务超时。失败 Graph 不释放 EXIT/外部后继，失败 Worker 不接受继续提交或提前 free 借用的 buffer。

AIC/AIV × 16/65 个相邻实例 × 1/7/9/15 scalar 的 16 场景，各平台均 48 轮通过完整数值/alias 与 v8 审计。真机观察到 16/65 个不同物化 owner、共享一份 Definition、写回范围互不交叠。完整 A5sim HBG 回归增至 139 passed。详细口径、启动前退出与正常退出的区别、故障回收耗时和日志见[专项记录](a5-hbg-aicore-graph-p3-validation.md)。

性能对照采用开发前 `02335c0f` 的隔离 detached worktree，独立 venv 和 runtime 构建；不改现有工作分支、不复制其他 session 的二进制。当时 P4 的完整性能/资源报告尚未完成，不能从上述功能通过推断性能收益；后续结果见下节。

### 初轮性能资源对照与长尾

已完成三组 Graph 和冻结 CSA 的三批交替 A/B，共 2520 次原样数值验证；同一计算 kernel 的实际大小/hash 一致。无 Graph CSA 稳定态 Device 中位数约 1.946/1.951ms，GM 差仅 callable image 的 64B；Graph 的新增 GM 与实际 scheduler-state 预算完全对上。

但原生 AIC/AIV、Mix 微图有严重长尾，不能按中位数宣布完成。L2 抓到 782ms 中有 782.27ms 位于启动等待，真正任务执行约 37µs。仅增加既有本地退避未解决 Mix 长尾，继续细分启动阶段。全部初轮数据、未通过项、计算口径和实验记录见[性能/资源与启动长尾调查](a5-hbg-aicore-graph-p4-performance.md)。

### 最终启动发布修复与交付

L3 进一步把长尾限制到物化发布/等待阶段。采用独占 128B 行的单 writer bypass 信号，最后一个 atomic 递减者发布，所有等待者看到信号后仍 atomic 确认零计数；错误路径不发布成功。每个有 Graph 的 run 增加 128B，无 Graph 不增加分配，不改计算 kernel、N 或公平领取资格。

最终三批原样 A/B 共 3150 次数值和后端来源通过。Graph 2D、AIC/AIV、Mix 的 Device P50 为 227.6/183.6/349.4µs，相对 legacy 为 +2.95%/−5.83%/−15.30%，900 个原生 Graph 稳定态样本没有 >5ms 长尾；Host 发布成本和尾分位数并非都改善。无 Graph CSA Device 约 1.946/1.949ms，ordinary/Mix 约 263.2/261.7µs，原 PA 约 1.389/1.402ms。CSA Host 两版都有约 80–140ms 批间波动，额外 150 次短批复核也不能建立稳定的 Host 差异，明确保留此测量限制。

完整 A5sim HBG 为 139 passed；23 个相关 C++ 测试程序、7560 个 Python 单测通过。最终信号版本的 A5 原 Graph/L4、48 轮参数边界、188 场景调度矩阵、宽 fan-in、双 slot 和 11 个实际故障恢复场景通过。独立性能/泳道、版本摘要、GM 对账和复现入口集中在[最终报告](a5-hbg-aicore-graph-p4-performance.md)。

尚未实现的嵌套 Graph、deferred completion、多 block/Mix predicate 和增量物化属于设计明确排除/后续范围；不把 legacy fallback 算作 resident 成功，也不将有界 timeout 测试写成全部硬件 trap 的证明。未主动 push。

### 主线用例兼容性复核（2026-10-31）

用例来源核对到 `upstream/main@74ddf236d776d0b9194ed8ca4d3fa4d3bb102587`，运行实现为 `feat/a5-hbg-aicore-graph@9381db6b7ae03bc14a7cf642b0f40602ad1ff07b`，没有合并主线或改动运行时代码。18 个实际编排/计算 kernel 源文件与主线逐字节一致；5 个原始测试文件去除说明文字和额外 scheduler 来源标记后，AST 一致。沿用原参数、输入、golden、数值容差和核数，不把测试包装器的新增校验当作新的业务负载。

| 用例 | A5 真机 | A5sim |
| ---- | ------- | ----- |
| 1D replay | 3/3 通过 | 3/3 通过 |
| 2D replay | 3/3 通过 | 3/3 通过 |
| AIC+AIV Graph | 3/3 通过 | 3/3 通过 |
| Mix SPMD | 3/3 通过 | 3/3 通过 |
| predicate 重绑定 | 3/3 通过 | 3/3 通过 |
| sync_start | 3/3 通过 | 3/3 通过 |

两个平台各 18 次、共 36 次调用全部通过原始数值比较、逐轮 fresh resident 来源检查及严格原生 L4 审计。每个平台日志均为 `5 passed`：这是 5 个 pytest 类节点，其中 `TestGraphExecutionResident` 包含 1D/2D 两个业务用例，不是少测一项。最终后端均为 mode 1，没有 mode 3/4 的 legacy 回退；36 份独立 raw 的 `graph_verified` 与 `protocol_verified` 均为真。

首轮使用 `--case Class::` 筛选时，被排除的 `TestGraphNoBoundaryResident` 仍被收集并在运行时跳过，随后 scheduler-mode teardown 因零 publication 报错。改用明确的 pytest 类节点，只收集需要的用例，全部通过；没有关闭来源门禁或修改源码。该筛选/teardown 问题尚未修复，保留初轮失败日志。AIC+AIV 在 A5sim 上原本标记为 manual，本次用 `--manual include` 显式纳入，没有把默认 deselect 算作通过。

复现命令如下；真机执行前仍须加载 CANN、进行架构/设备检查。本机缺少 task-submit，使用 ACL 确认 A5 和空闲后直接运行，未获得队列锁。A5sim 改为 `--platform a5sim --pto-session-timeout 600`；两个平台顺序运行，避免相同用例的秒级输出目录名称相撞。

```bash
graph_fixture=tests/st/a5/host_build_graph/graph_execution/test_graph_execution_resident.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 SIMPLER_REPORT_SCHEDULER_MODE=1 \
  .venv/bin/python -m pytest \
  "$graph_fixture::TestGraphExecutionResident" \
  "$graph_fixture::TestGraphExecutionAicAivResident" \
  "$graph_fixture::TestGraphMixResident" \
  "$graph_fixture::TestGraphSyncResident" \
  "$graph_fixture::TestGraphPredicateResident" \
  --platform a5 --device 0 --max-parallel 1 --pto-session-timeout 1200 \
  --manual include --enable-chip-swimlane 4 -v -s
```

本地证据目录为 `outputs/mainline-graph-check-20261031.ZDGeoY/`：`a5-selected.log`、`a5sim-selected.log` 是最终成功结果，`a5.log`、`a5sim.log` 保留筛选问题，`mainline-test-source.diff` 保存来源差异。成功日志列出每份泳道的独立路径。本次只是功能/协议复核，带 L4 的执行耗时不作性能结论；原始日志、泳道、构建产物均不提交 Git。

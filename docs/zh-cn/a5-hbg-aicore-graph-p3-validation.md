# A5 HBG AICore Graph：能力与故障验证

设计和总进度见[实现过程](a5-hbg-aicore-graph-implementation.md)。本页保留能力开放及后续故障检查的独立证据，不把 CPU 反例或正常真机执行当作全部故障路径的证明。

## 实现和红绿证据

- 先给原 Mix、sync-start、predicate 用例增加逐轮 resident 来源门禁：原数值检查之后，三者均因 `mode=3` 失败。继承原 CASES、输入和 golden；跨目录 wrapper 只把 source 路径解析到原文件，不复制或修改 kernel。
- Graph 子任务的多 block/Mix/sync-start 元数据接入既有 resident 执行器，不新增领取/协同机制。Host binding 的 mask/N/sync 矩阵先失败，开放后转绿；设备容量约束仍由原 runtime maker 检查。
- Scalar 为 predicate 复用普通 tensor 的边界/内部 heap 重绑定计算，只写入本实例 payload 的既有 `DispatchPredicate`。Host 对每个不同 Definition 检查 operand 来源、dtype/宽度、比较操作和引用范围；Scalar 检查本次实例的地址/偏移/缓冲区范围，拒绝自身或未来任务的输出。没有额外 GM scratch，也没有把整份 TensorMap/Graph 搬进栈。
- predicate 仍只开放单 block、单 lane；合法但超出该范围的组合保持整轮回退。非法 Definition 不作为“合法回退”接受。新增 predicate 地址/操作/宽度/溢出反例先失败，修复后通过；同一 Definition 的原 AICPU materializer 与新 Scalar materializer 的参数、predicate 字段一致。
- 原 A5 Mix、sync-start、predicate 用例在 A5sim 和 A5 上各 3 轮通过后，才把对应默认 pytest marker 改为 resident。本节能力开放版本的 Graph L4 当时仍回退；后续接入情况见[原生协议记录](a5-hbg-aicore-graph-p4-protocol.md)。

## 有真实阻塞的执行矩阵

新增 `tests/fdwic/pa_hbg/spmd/test_graph_ordered_execution.py` 和专用编排，**复用已有 ordered/sync kernel 和完整 oracle**。同一计划可放入一个 Graph 的多个子任务，也可放入多个 Graph；事件序号来自设备诊断 atomic，不靠不同核的时钟比较，也不作为性能数据。

第一次扩展矩阵暴露测试编排误用了 Graph boundary 的外部 `set_dependencies`，在 Host 以 `INVALID_ARGS` 拒绝，尚未 launch。当前公共 Graph API 不接受这种边界；修正测试为真实输出 view 和输入 view 经 TensorMap 生成跨 Graph 边，Graph 内部仍用已支持的子任务显式依赖。**没有为测试放宽公共 API**。两个失败场景单独转绿后，完整矩阵重新运行。

| 场景组 | 每平台场景数 | 具体约束 |
| ------ | ------------ | -------- |
| 原 N/分波、慢核、依赖与复用 | 116 | mask 1–7；N=1/2/7/8/9/31/32/33/63/64/65/512；原 block_idx/block_num；恰好一次执行；无 inactive lane 伪执行；同图与跨图边；头/中/尾慢 block；512→1→65→1→512 复用 |
| sync-start 与局部推进 | 72 | 7 种同 mask、4 组异构 mask；同图多个 roots/跨 Graph；真实核内 rendezvous；独立任务尾 block 等待另一任务先进入；真实依赖不得提前；sync→ordinary 和 ordinary→sync |
| 合计 | 188 | A5sim 和 A5 均 PASS；每场景至少 3 轮，含复用子序列后每平台实际 612 次调用；每次均检查 fresh resident |

其中 A5 的纯 AIV 同步 cohort 为 64 block，其余为 32；A5sim 分别为 16/8。ordinary 的超容量任务分波，sync-start 的核内等待检查“所有成员进入后才能返回”；同图/跨图 overlap 检查不能被全图串行完成绕过。

另增 `test_graph_wide_fanin.py`：纯 AIC/AIV 分别验证内部 fanin=128、129、1023，最大 body 为 1024 个子任务。依赖全部生产者的内部校验任务与 Graph 外部后继分别写两个独立 summary，后者不能覆盖前者的错误。复用原 `record_slot_*`/`verify_fanin` kernel 和数值 oracle；6 场景在 A5sim/A5 均各 3 轮通过，确认内部 CSR 没有误套顶层 128 的内联容量。真机日志保存在 `outputs/aicore-graph-p3-wide-a5.sNehgO/`。

## 回归结果与复现

| 检查 | 结果 |
| ---- | ---- |
| Graph materializer / Host binding | 8 / 9 项 PASS；连同此前 Graph helpers 共 51 项 |
| A5 C++ 相关回归 | 17/17 个测试可执行文件通过，包括 ready、dispatch、SPMD、Mix、ordered、admission |
| Python 来源门禁与原 ordered/sync oracle 反例 | 386 passed，其中 Graph wrapper 门禁 18 项 |
| 完整 A5sim HBG 场景回归 | 118 passed，391 deselected，1 个既有 `load_module` 弃用 warning |
| 原 Mix / sync-start / predicate | A5sim/A5 均各 3 轮 PASS |
| 新 Graph ordered / sync 矩阵 | A5sim/A5 均 188 场景、612 次调用 PASS |
| 内部宽 fanin / 最大 body | A5sim/A5 均 6 场景、18 次调用 PASS |

在本 worktree 独立 `.venv` 内重建 A5sim 和 A5 runtime 后运行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tests/fdwic/pa_hbg/spmd/test_graph_ordered_execution.py -p a5sim

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tests/fdwic/pa_hbg/spmd/test_graph_wide_fanin.py -p a5sim

# 加载 CANN 环境、确认设备，并先给 ASCEND_PROCESS_LOG_PATH 设置新的日志目录。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tests/fdwic/pa_hbg/spmd/test_graph_ordered_execution.py -p a5 -d 0
```

真机首轮扩展验证保存在 `outputs/aicore-graph-p3-a5.RNMGjH/`，最终完整矩阵保存在 `outputs/aicore-graph-p3-matrix-a5.xcawWM/`，均为本地忽略目录，没有覆盖或提交旧泳道图/raw。每轮真机工作前通过 ACL 确认 `Ascend950PR_958b`、设备 0 三类计算利用率为 0；本机仍没有 `task-submit`，结果只用于功能验证，不作为性能 A/B。

上面的能力开放版本基于 `0e63f5ee0` 的工作区修改；故障修复前的 A5 onboard 二进制 SHA256 为：

```text
aicore_kernel.o     891645f5bcbfd7ccf9ae24eb47cd66c7cc204cc1f241eb6021290fca38e7c80e
libaicpu_kernel.so  fa0bec99a948e2c4a44516a3e76934105f6947116ab760c572642856cb01f1e6
libhost_runtime.so 5568c8092b1c60e1b4d88cfbc958423a1cd2424b1a89caab741ec579168c84d5
```

## 2026-10-30：epoch、首现场与 slot 生命周期

### 反例复现和修复

直接调用生产 bootstrap 和 completion 函数的测试确认：

- build/instance 的旧 epoch 原本仍会写出参数或贡献完成计数；现在启动前对照本轮只读 summary，子任务贡献前再核对所属实例。
- 控制区地址偏移 8B 时，原顺序先写 owner/state，再由 materializer 拒绝，已改为任何 atomic 写之前检查地址对齐。
- 重复 ticket 原本能覆盖已有 owner；现在通过零值到本 worker 的 CAS 取得唯一物化权，已有 owner 或非初始状态均拒绝。
- 首错误原本只有 dense ref。新增公共父任务 ID、子任务 local ID 和有效位，复用原 8B 对齐空隙，RunControl 仍为 512B、旧字段偏移不变。未知子任务使用 UINT32_MAX，不冒充 local_id=0；首错误之后的错误不能覆盖它。日志可直接看到 `public_task=74 sub_task=0`。v7 导出合同尚未扩展，这不是原生 Graph L4 已完成的声明。

`test_hbg_graph_bootstrap` 的 8 项覆盖成功发布、旧 build/instance epoch、错位控制地址、重复 owner、失败 owner、暂停 owner 的等待者和超时。等待者必须看到整个物化完成，错误和超时都不能增加公共完成数。新增旧实例完成反例使 Graph helpers 合计 60 项；它们是 CPU 生产函数执行证据，不是芯片上的缓存可见性故障注入。

### 端到端边界与复用

| 场景 | A5sim | A5 | 检查内容 |
| ---- | ----- | -- | -------- |
| 无 tensor 边界 | 3 场景各 3 轮 PASS | 同左 | 1/16 个实例、最多 1024 个 Dummy 子任务，逐轮 fresh resident；不产生计算 kernel |
| 原 predicate 复测 | PASS | 3 轮 PASS | 边界、view、内部 heap 的条件地址及 skip 语义 |
| native lifecycle | 2 场景、12 次调用 PASS | 同左，实际双 slot prepare | body 130→2、2→1024、1024→130；输出与内部/外部 summary 完整校验 |
| 超容量拒绝 | 8 个子测试 PASS | 同左 | body=1025；7 种 mask 的超容量 sync-start；错误码/诊断正确，backend publication 不增加 |

`test_graph_native_lifecycle.py` 使用既有 prepare/launch/wait/finalize 接口：A5 上前驱保留执行 claim 时，后继可以在另一个 slot 准备，两个 arena bank 的 GM heap 不同；后继提前 launch 被拒绝，且同一 token 在前驱 finalize 后仍可正确执行。完成后的旧 token 被拒绝，后续不同大小 run 不串写。**这里证明的是并发准备及保留所有权，不声称两个整算子并行在设备执行。** A5sim 的 backend 不支持并发 prepare，只证明顺序复用；测试明确打印该能力差异。

最初生命周期测试在第二个 case 把 lease generation 重置为 1，runtime 正确拒绝 stale generation。测试改为跨 case 单调推进自己的 lease，没有放宽 runtime 的 generation 检查。容量用例最初只匹配 release 分支的 warning 字样，debug 分支先抛 assertion，因此改为匹配二者共有的 finalize 诊断，并继续严格要求原错误码与 publication 不变。

无 tensor 边界本来就是公共 Graph 支持的情况，共享文档中的旧“空 boundary 不支持”描述同步纠正，公共 API 没有扩展。

### 修复后复测

- 重新构建本 worktree 的 A5sim/A5；20/20 个相关 C++ 测试可执行文件通过；Python 来源及 oracle 反例仍为 386 passed。
- 完整 A5sim HBG 回归：127 passed、391 deselected、1 个既有 Python 弃用 warning。
- 正向 ordered/sync 188 场景、612 次调用与宽 fanin 6 场景、18 次调用，均在 A5sim/A5 重新通过，保持原 N、kernel 和 oracle。
- A5 生命周期/空边界/predicate 日志：`outputs/aicore-graph-p3-lifecycle-a5.SxaYqA/`。
- A5 容量拒绝日志：`outputs/aicore-graph-p3-capacity-a5.XF2l5s/`。资源调度器把这 8 项分为独立子进程，父进程的 `no tests ran` 不表示未执行；验收读取 8 个子任务 PASS 和整体退出码 0。
- A5 最终正向矩阵：`outputs/aicore-graph-p3-checked-a5.BdAlkQ/`。上述目录均为新建且 Git 忽略，未覆盖旧证据，不提交 raw/泳道图。

故障修复后的 A5 onboard 二进制 SHA256：

```text
aicore_kernel.o     9a98cb25d46acf78932dcd73c30c2d433f59590f61c850d77963d8eb10e6b737
libaicpu_kernel.so  ad123246e91e6a383add474b438f2bf665e7b5756187fc08a5e238eb655793a8
libhost_runtime.so 9797267a0576d353f5b4c8a8c29a4a5b8fca4bee56901dde1803946616a8a640
```

复现命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tests/fdwic/pa_hbg/spmd/test_graph_native_lifecycle.py -p a5sim

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tests/st/a5/host_build_graph/graph_execution/test_graph_execution_resident.py \
  -p a5sim --case TestGraphNoBoundaryResident::instances2_children1024

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest tests/st/a5/host_build_graph/graph_execution/test_graph_capacity.py \
  --platform a5sim --device 0 --pto-session-timeout 600 -q
```

上卡时将平台改为 a5，先检查 A5 设备，设置新的 `ASCEND_PROCESS_LOG_PATH` 并指定设备。本机无 task-submit，这些上卡记录只作为功能证据，不作为性能 A/B。

## 尚未完成的验收

1. 设备实际故障下的退出收敛与失败回收；不能拿 CPU 注入或 Host 预提交拒绝替代全部设备错误场景。
2. 相邻实例参数写回/alias 的专项边界测试；已有多实例结果与原 view/predicate 测试不能宣称穷尽所有交错。
3. 完整内存预算与端到端数据。原生 v8、父子泳道和严格 L4 功能矩阵已在后续通过，见[协议与泳道记录](a5-hbg-aicore-graph-p4-protocol.md)。
4. 同负载 legacy/native 和无 Graph 固定 CSA 的性能对照。未获得数据前不宣称 Graph 性能收益。

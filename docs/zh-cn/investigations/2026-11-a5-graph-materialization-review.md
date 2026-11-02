# A5 Graph 物化提案复核：认可启动成本，不接受原样删除屏障

**日期**：2026-11-02。**复核源码**：`f0722c567`。
**结论**：暂缓按原提案改运行时；先补齐失败契约、发布/可见性和进度证明，
再测量候选。不是“实测无收益而放弃”，本次没有新增性能测量。

对象是[物化策略提案](../../../tests/fdwic/pa_hbg/a5-hbg-graph-materialization.md)。
原文基线为 `47c55ec57`；本次对照当前实现修正其中的事实和方案边界。
legacy 完成计数还核对了原文指定的 `bad268384`，两个原子计数同样存在。
`47c55ec57` 对象不在本地，本文不声称在该历史提交上重跑过测试；
原生调用时序与测试结论均以 `f0722c567` 为准。

**后续方案修订**：已选定先保留启动屏障、并行构造大实例，
再有条件研究验证前置与按实例发布。详见[正式物化优化方案](../a5-hbg-aicore-graph-materialization-plan.md)。
本记录对原样移动 ENTER 的异议仍成立，不代表否定这条修订后的路线。

## 1. 接受与异议

| 判断 | 复核结果 |
| ---- | -------- |
| P1：单实例由一个核全量物化 | 接受。实例级并行度上限是 `min(instance 数, worker 数)`；大单图有串行构造段。 |
| P2：所有实例物化结束前不能执行任务 | 接受。屏障位于 ready bootstrap 前，独立的普通任务也要等待。 |
| P3：原生路径有共享完成计数器 | 接受存在该计数器；未证明它是性能瓶颈。 |
| P3：legacy 没有完成计数器、只靠 wake list | 错误。legacy 同样逐 child 更新完成和退休计数。 |
| A：搬到 ENTER 即可解决 P1/P2 | 只能改善多实例重叠；单实例仍全量串行，不能使其首批 child 提前执行。 |
| A：已有唯一 owner，因此无需协议改变 | 不成立。owner 不等于发布、缓存可见性、失败行为或进度保证。 |
| B：只在 prepare 前失效 child storage 即可 | 不充分。MIX 更早解析 payload，参数池和最终执行核也有读取。 |
| native/legacy 对照可以直接量出物化策略收益 | 不成立。它比较完整调度后端，不能单独归因于物化。 |

这些结论不替旧实现的性能背书。原有全量物化确实支付启动成本，
只是尚无依据把一种更复杂的协议直接当作已经证明正确、收益明确的修复。

## 2. 代码依据与反例

### 完成计数与 wake list 是两回事

[`GraphExecution`](../../../src/common/host_build_graph/graph_execution.h)
同时保存 `remaining_tasks` 和 `retired_tasks`。
`graph_execution_complete_sub_task()` 执行 `remaining_tasks.fetch_sub(1)`；
`graph_execution_retire_sub_task()` 执行 `retired_tasks.fetch_add(1)`。
[A5 legacy 的完成路径](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler.h)
先释放内部 wake list，再调用这两个函数；A2/A3 使用相同流程。

原生路径的
[`scheduler_graph_child_completed()`](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h)
也先完成该 child 的内部释放，再贡献 `remaining_children`。
最后一个贡献者发布 EXIT。原子的指令、内存层级和竞争者数量不同，
不能仅凭两边都有计数器证明性能相同；同样不能以“legacy 没有计数器”证明它更好。

### 延后物化会改变已有失败行为

[`bootstrap_graph_instances()`](../../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp)
在成功发布全部实例前，不会进入 `bootstrap_ready_graph()`。
物化失败会发布 first error 并终止启动。

现有[物化失败场景](../../../tests/st/a5/host_build_graph/graph_execution/test_graph_device_failure.py)
在一个 run 中保留健康实例并使另一个实例失败，明确检查：

- 健康和失败实例的输出仍为原值；
- `protocol["executions"]` 为空；
- `resolved_task_count == 0`，没有 child 完成贡献；
- 错误标明父 Graph 和 child；Worker 被 poison，未退休的存储不能提前释放。

方案 A 允许健康实例的 ENTER 先结束并执行 child，再由另一个 ENTER 发现错误。
这与上述启动失败行为不等价。仅证明 child 在自己的 ENTER 之后读数据，
不能排除这个反例。这里讨论的是物化阶段发现错误的保证，
不是声称运行时对所有 kernel 错误提供事务回滚。

### 可见性证明不能只盯一个 dispatch 入口

本次检查了三个 Graph storage accessor、payload parser、直接 storage 地址使用，
以及普通/MIX dispatch 的调用路径：

| 位置 | 当前行为及延后物化的要求 |
| ---- | ------------------------ |
| `scheduler_bootstrap_route_task` / fanin accessor | 原生 node map 路径读 Host 预建的依赖元数据，不读 child storage；这部分原文判断成立。 |
| `scheduler_classify_task_shape` | 当前生产调用在 Host 的 binding、runtime maker 和 protocol graph 路径；未找到设备侧分类调用。 |
| 普通 dispatch | prepare 内调用 payload parser，随后读取分离的 scalar pool。 |
| `scheduler_dispatch_mix_ready` | **先解析 payload，再调用 prepare**；只在 prepare 中失效已经太晚。 |
| `scheduler_materialize_task_payload_resolved` / `scheduler_materialize_mix_arguments` | scalar 值来自独立 pool；Tensor 参数复制的是描述符地址，不是描述符内容。 |
| 实际执行核 | 会通过参数中的地址读取 Tensor 描述符；只保证调度核可见，不代表执行核可见。 |

对应实现见 [Graph accessor 与参数组装](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h)、
[MIX dispatch](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_mix.h)、
[缓存原语](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_memory.h)。
[CANN API 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0177.html)
也将整个 DCache 刷新定义为核内操作，不能把物化核刷新理解为替其他消费核刷新。

检查没有发现正常原生 child 在 ENTER 就绪之前消费其 storage 的路径。
这支持继续研究方案 A，却没有证明“只刷新物化核即可”或“B 已足够”。
逐行失效可以是候选，但必须覆盖所有读取区域和 reader，且先有 producer 写回及发布顺序。

### 单 owner 不等于进度保证

legacy 除 prepare queue 外，
[`graph_execution_materialize_slice()`](../../../src/common/host_build_graph/device/graph_execution.cpp)
还有 `materialize_busy` CAS。整实例延后物化可以复用 ENTER 的唯一 ready owner，
但不能因此省略其余不变量。

AICore 调度/执行是协作式循环，长时间执行 ENTER 会占用该核。
[原生 Graph 设计 §3.4](../a5-hbg-aicore-graph-design.md)
已要求增量构造证明必要的构造、发布和释放不会被后续阻塞 kernel 卡住；
[旧 Graph/drain 调查](../../investigations/2026-09-hbg-graph-release-vs-drain-reorder.md)
说明仅调整控制逻辑位置不足以保证这种进度。
这是尚须证明的风险，不是本次已经复现了方案 A 的死锁。

## 3. 已采用的后续设计方向

1. 优先按写回范围并行构造大实例，保留启动屏障。它改善单实例构造吞吐；
   单实例提前执行仍需要分片发布，多实例重叠则属于后续 A 的目标。
2. 保持“物化失败前零 kernel 进入”。第一阶段靠全 run 构造成功后放行保证；
   后续延后生成必须先完成所有可失败验证，不选择放宽现有失败行为。
3. 在 native 基线上建立完整 producer/consumer 发布关系，覆盖普通/MIX、
   参数池、跨核执行、重复 replay、旧 epoch、失败退出和诊断记录。
4. 先测物化本体、写回、等待及首 kernel 开始，再决定是否优化完成计数。
   cluster 分片计数需要处理动态任务归属和未 claim 的 child，不能预设没有成本。
5. native/legacy 只作完整后端对照。`get_last_scheduler_runtime_mode()` 及
   `get_scheduler_mode_publish_count()` 用于确认本轮选择；
   `benchmark_graph.py --expected-backend` 是断言，不是强制切换器。
   物化策略的归因需要同一 native 基线下只改变该策略的交替 A/B。

已有[性能报告](../a5-hbg-aicore-graph-p4-performance.md)
同时保留稳定态和首调用样本，并明确三次首调用不足以估计冷启动尾分布。
其隔离发布信号实验没有改变物化策略，因此不能把等待长尾直接归因为串行写入，
也不能用它证明本提案无收益。本次不重写或冒充那些历史测量。

## 4. 本次修改与验证

- 修正原提案的计数器对照、单 owner 论证、A 的收益边界、B 的消费者范围、
  失败契约、历史测量表述和 A/B 方法。
- 保留运行时实现，未新增环境开关，未以删断言或放宽故障行为来让方案成立。
- 从本工作区重新构建六个相关 C++ target：A5 materialize、bootstrap、services、
  binding，以及 A5/A2A3 graph activation；**64 个 GoogleTest 用例全部通过**。
- 这是生产 helper 的 CPU 协议检查，不是 A5 缓存一致性或性能验收。
  本次没有实现增量候选，也没有新增上板性能结果。

复现命令：

```bash
source .venv/bin/activate
cmake --build tests/ut/cpp/build --parallel 2 --target \
  test_a5_hbg_graph_materialize test_a5_hbg_graph_bootstrap \
  test_a5_hbg_graph_services test_a5_hbg_graph_binding \
  test_a5_hbg_graph_activation test_a2a3_hbg_graph_activation
ctest --test-dir tests/ut/cpp/build --parallel 2 --output-on-failure \
  -R '^test_a(5|2a3)_hbg_graph_(materialize|bootstrap|services|binding|activation)$'
```

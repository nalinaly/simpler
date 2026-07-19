# 对 `make_replay_faster.md` 第 7/8.2 节的实测回应：QK 单回调 Lazy Submit

> cleanroom 归档状态（2026-07-19）：本文由
> `wip/qk-lazy-pmu-migration-20260718@cb799c4d` 迁入当前目录；新实现基线为
> `origin/fdwic-swimlane-deps@6caa269c`。本文在进入 cleanroom 前的 A/B/C、
> production、STOP 和 PMU 数字均为历史实验记录，不是 `6caa269c` 的性能基线，
> 当前分支必须重新构建和采样。历史 `docs/make_replay_faster*.md`、
> `tests/atomic_probe/pa_lazy_qk_pmu/`、d112 增量 probe 及 `/atomic/sfd/gpt/`
> 产物没有随附；旧 `/private/3-gpt/` 工件也位于本 worktree 之外。下文明确
> 标出的历史路径不能冒充当前仓内可复跑入口。

> 验证日期：2026-07-17～2026-07-19
> Production 业务代码基线：`a3f5ecc2186fb8a06cded6d26886a4ab802009c0`
> PMU 诊断 host HEAD：`76df85ced5a43099d268cd6bcd5a988d03d9cca7`；
> 两者之间 PA/runtime 相关路径无 diff
> 用例：`TestPagedAttentionUnroll::Case1`
> 平台：真实 A5；A5sim 只做流程验证
> 候选状态：性能不合格，production 修改已回退

## 1. 结论

本轮验证了 private TensorMap 下这条控制流/结构路径可以运行：

```text
execute-first drain
→ claim
→ builder callback 恰好一次
→ winner/loser 按角色求值参数
→ private output/map 必要工作
→ winner build / loser replay
```

这里的“可以运行”只表示一次 submit/一次 callback/一次 claim、claim-first
顺序与结构 oracle 跑通，且真实 A5 未报 507015/507018。因为本轮命令使用
`--use-example-exec-time` 跳过数值 golden，它不能扩大成 PA 端到端数值语义
已经等价。

它没有把一次业务 submit 拆成两次，也没有让 winner 再调用一次
`winner-bind` callback。为了让 orchestration 访问 runtime TU 内的
`static [[block_local]]` 状态，候选把完整 `dist_engine.cpp` 与 PA
orchestration 融合到一个 CCEC TU；因此历史对照同时改变了参数求值策略和
TU/codegen 形态，并不是代码变量上的单变量实验。

同一 HEAD、同一 108-worker Case1 的真实 A5 production A/C 配对为：

| 版本 | 三轮全局 Submit span | 中位数 |
| --- | --- | ---: |
| eager | 5.218171 / 5.168611 / 5.205719 ms | **5.205719 ms** |
| QK lazy + fused TU | 5.510962 / 5.630435 / 5.525736 ms | **5.525736 ms** |

候选回退 **0.320017 ms（6.15%）**；候选最好一轮仍比 eager 最差一轮慢
0.292791 ms。这足以判定当前 fused-lazy 整体候选不能提交，但不能单凭这组
A/C 数据把回退归给 fused TU、lazy front 或 I-cache 中的任意一项，也不能
把结论扩大成“第 8.2 节的一次 callback 惰性方案不可行”。

为拆开变量，2026-07-18 又构造了 `A=split eager`、`B=fused eager`、
`C=fused lazy` 三种机器形态，以及语义对称的局部
`total/scalar/icache request/icache miss` 窗口。三种 STOP r1 预检已通过内容
oracle；后续归档虽补齐 21 行 manifest，但三种 QK 的 total counter 全部为
0，当前严格解析器以 `QK_TOTAL_ZERO` 拒绝，仍未形成有效 EMPTY/QK PMU
三向差分。2026-07-19 又完成冻结 A/B/C STOP 的三轮 96-worker 泳道复测；
它只支持同口径诊断形态的性能方向与内容 oracle，不是有效 I-cache counter，
也不是 production 性能构建。历史无埋点 Submit span 仍是性能验收数据；带
PMU 的诊断 ELF 只用于解释计数差异，不能用它的 wall-time 替换上面的
`5.205719/5.525736 ms`。

## 2. 为什么先选择 QK

此前 PV two-phase 已经给出一个反例：PV loser 虽然把 finish 主体从
1.377 us 降到 1.268 us，但 prepare/finish 外围从 0.922 us 增到
1.321 us，完整 AIV loser Submit 反而从 2.647 us 增到 3.012 us。其输入
本来就是现成 Tensor 描述符，没有逐 task 的高价值 `view`；参数节省太轻，
很容易被拆分固定成本覆盖。因此本轮没有继续拿 PV 论证 lazy 收益，而是选择
确实包含 `qi = Tensor::view(...)` loser 冗余的 QK。

Case1 的关键参数为 `batch=256`、`N_UNROLL=64`。每个 scope 只有一个 QK，
每个 QK 由 108 个 worker 重放，但只有一个 AIC winner 真正执行。

QK 的参数为：

```text
INPUT  qi
INPUT  key_cache
INPUT  block_table
OUTPUT sij_buf
SCALAR n_blocks
SCALAR block_offset
```

其中 `qi = Tensor::view(query, ...)` 是最明确的 loser 冗余；QK 没有
`INOUT/OUTPUT_EXISTING`，输入均为外部输入，因此 QK fan-in 为 0。

按 Case1 控制流、108 个 replay worker 与 winner 条件推导，候选应减少的
源码层工作次数为：

| 工作 | eager | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| `qi` view | 27,648 | 256 | -99.07% |
| `qi + out_view` 总 view | 55,296 | 27,904 | -49.54% |
| input thunk/Arg 写入 | 82,944 | 768 | -99.07% |
| scalar thunk/Arg 写入 | 55,296 | 512 | -99.07% |
| materialize 扫描的 tensor slot | 110,592 | 28,416 | -74.31% |

这里的“一次 callback”需要准确理解：它是每个 worker、每个 QK replay 调用
一次，所以总 outer callback 次数仍是 `108 × 256 = 27,648`；只是 loser
不再求值其中的 input/scalar thunk。

private TensorMap 下以下工作不能因输掉 claim 而删除：

- fresh `OUTPUT` 的确定性物化；
- 每核一致推进 `heap_next`；
- 生成可供下游使用的 `TaskOutputTensors`；
- 推进本核 private map 的 retire 窗口；
- `INOUT/OUTPUT_EXISTING` 的 producer 登记（QK 本身没有这两类参数）。

因此本轮目标不是让 private loser 变成零成本，而是只跳过纯消费侧参数。

## 3. 候选实现的具体方式

### 3.1 orchestration 侧

QK 从 eager 代码：

```cpp
params.reset();
params.add_input(qi, key_cache, block_table);
params.add_output(sij_buf_ci);
params.add_scalar(n_blocks, block_offset);
auto qk_outs = rt_submit_aic_task(FUNC_QK_MATMUL, params);
```

改为一次内部/codegen 专用调用：

```cpp
auto qk_outs = rt_submit_private_aic_qk_codegen_once(
    FUNC_QK_MATMUL,
    params,
    [&](PrivateAicQkBuilder &builder) {
        builder.add_input([&] { return lazy_qi_view(); });
        builder.add_input([&] { return key_cache; });
        builder.add_input([&] { return block_table; });
        builder.add_output([&] { return sij_buf_ci; });
        builder.add_scalar([&] { return n_blocks; });
        builder.add_scalar([&] { return block_offset; });
    }
);
```

`qi` 在 scope 内按核缓存：某核第一次赢得该 scope 的 QK 时才构造；Case1
每个 scope 只有一个 QK，所以全局恰好构造 256 次。`out_view` 服务后续
`INOUT`，仍在公共路径由所有核构造。

### 3.2 单 submit、单 callback 的真实执行顺序

```text
rt_submit_private_aic_qk_codegen_once(...)       # 业务源码只调用一次
  ├─ 查询 fatal 状态
  ├─ 取得本核 DistCore，分配确定性 task_id
  ├─ drain_block_won + drain_phase_b             # execute-first
  ├─ claim 前校验 task_id / kernel_id
  ├─ AIC 对 cube cursor 做一次 claim
  │   AIV 固定为 loser，不做 atomicMax
  ├─ claim 后 reset 复用的 L0TaskArgs
  ├─ 调用 builder callback 一次
  │   ├─ winner：INPUT + OUTPUT + scalar
  │   └─ loser：仅 OUTPUT
  ├─ 打包 16 B ticket
  │   └─ submit_trace_start + task/kernel/won 元数据
  └─ 调用一次 noinline finish
      ├─ 重建 DistSubmitCtx；不再分配 task_id，不再 claim
      ├─ 所有核物化 fresh OUTPUT、推进 heap 和 map retire
      ├─ winner-only fan-in
      ├─ producer register
      └─ winner build / loser replay
```

因此它同时不同于 E1 的 prepare/finish 两次 runtime 边界，也不同于 E2 的
winner `Prepare + WinnerBind` 两次 dispatcher：

- outer builder callback 只调用一次；
- claim 也只执行一次；
- orchestration 只有一次 submit API 调用；
- 源码/调用形态是 inline front 最后 `return` 一个 `noinline` finish；本机
  无法解码 HiIPU 指令，不能进一步声称编译器一定生成了机器码 tail-call。

### 3.3 为什么使用 fused TU

当时的直接阻塞是：`g_self`、`g_dist` 等 CCEC worker 状态定义为 runtime
TU 内的 `static [[block_local]]`。独立 orchestration TU 不能直接用同一份
对象完成 claim-first。

同时，`tests/atomic_probe/AICore嵌套Lambda捕获与模板调用验证.md` 已经证明
caller 栈地址跨
特定未内联调用边界存在 CCEC codegen 敏感性。候选因此临时生成：

```cpp
#define PTO_FDWIC_FUSED_AICORE_TU 1
#include "dist_engine.cpp"
#include "paged_attention_orch.cpp"
```

构建器在该 opt-in 用例中移除原本单独编译的 `dist_engine.cpp`，避免重复
定义；AIC/AIV 分别以 `dav-c310-cube`、`dav-c310-vec` 编译这个 wrapper。
A5sim 不使用 fused CCEC 路径，而是以 winner 语义调用 callback 一次后回到
旧 eager API，所以 A5sim 只能补充验证流程，不能证明真实 loser fast path。

这一方案解决了状态同源和 callback inline 问题，但也让 CCEC 在 `-O3` 下
同时看到完整 runtime、PA orchestration 和所有旧 submit 调用。源码上没有
改 alloc/SF/PV/UP，最终二进制形态却已经改变；这违反了第 7 节“保留旧
`rt_submit_*` 路径”的性能含义。

### 3.4 方案演进：哪些尝试做过、为什么停止

这一轮先后出现过几种容易混淆的“拆分/融合”形态。这里必须把三种次数分开：

- **业务 submit 次数**：orchestration 对一个 task 发起几次 submit；
- **builder/dispatcher 次数**：业务参数构造逻辑实际被调用几次；
- **跨 runtime 调用边界次数**：热路径从 orchestration 进入独立 runtime
  函数几次。

三者不是一回事。一次对外 submit 仍可能在内部先后调用 `prepare` 和
`winner-bind` 两次 dispatcher；一次 callback 也仍可能因为 `begin/finish`
产生两次跨 runtime 调用。按真实代码、补丁和实机结果，本轮演进如下：

| 阶段 | 参数/调用形态 | submit / callback（或 dispatcher）/ runtime 边界 | 结果与停止原因 |
| --- | --- | --- | --- |
| E0：原始 eager | orchestration 先完整构造 `L0TaskArgs`，再进入 split runtime 的 `dist_submit_impl` | 1 / 0 / 1 | 正式性能基线；loser 也完成 input/scalar 打包 |
| E1：早期 PV two-phase | split runtime 先 `prepare/claim` 返回 ticket；orchestration 按 winner/loser 构造参数；再 `finish` | 1 个业务 task / 无 builder callback / 2 | outputs-only loser 的 materialize/register 局部变快，但完整 PA 主体稳定回退约 6.16%；额外调用、context 重建和外围控制超过节省，已回退 |
| E2：split weak dispatcher | 一个对外 lazy submit；runtime 先调 `Prepare`，winner 在 claim 后再调 `WinnerBind`；caller 栈 context 跨未内联边界 | 1 / loser 1、winner 2 / 1 个 submit 边界，内部再跨 dispatcher 边界 | 违反“一次 callback”；PA 设备路径报 507018。对应最小 probe 把核心边界收敛为 caller 栈地址跨特定 noinline weak-context 调用时可稳定报 507015；两者错误层级不同，不能声称 fault PC 相同 |
| E3：fused single-callback | inline front 在融合 TU 内 drain/claim；本地 callback 一次；把 16 B ticket 和已物化的 `L0TaskArgs` 交给一次 noinline finish | 1 / 1 / 1 个 finish 边界 | 一次 callback、一次 claim 的结构和 A5 流程跑通；但 production 相对 eager 中位数回退 6.15%，因此整体方案不合格 |
| E4：A/B/C 归因实验 | 保持对称 PMU 窗口，分别构造 split eager、fused eager control、fused lazy | 每个形态仍只有一次业务 submit；C 为一次 callback | 用于拆分 E3 的 fusion/codegen 代价与 lazy bundle 代价；它是诊断矩阵，不是三版依次准备提交的业务实现 |

E1 与 E2 是两种不同的失败：E1 的主要证据是**增加一次 runtime 往返后性能
回退**；E2 同时存在**winner 两次 dispatcher**和 caller 栈地址跨特定
noinline 边界的 CCEC codegen 敏感性。E3 规避了 E2 的跨边界 caller context，
并把 callback 恢复为一次，但为此引入完整 fused TU，最终又出现不可接受的
整体性能回退。

### 3.5 A/B/C 各自具体回答什么

A/B/C 是在 E3 已被 production 判定回退后补造的定因控制组：

```text
A = split eager
    原始 orchestration eager 打包
    → 原始 split dist_submit_impl

B = fused eager control
    与 A 相同的 eager orchestration entry
    → 在诊断构建中挂入与 C 相同的关键 fused helper/tail 和布局组合
    → QK 动态仍走 legacy dist_submit_impl

C = fused lazy
    claim-first inline front
    → 本地 builder callback 一次
    → noinline finish(ticket, L0TaskArgs)
```

因此：

- `B-A` 回答当前诊断构建里 fusion、outlining、额外 helper 链入和布局变化的
  **组合动态代价**；
- `C-B` 回答在关键 runtime helper 的静态形态已核验一致时，legacy eager
  submit 切到 lazy front/finish、claim/materialize 重排和条件参数求值的
  **组合动态代价**；
- `C-A` 与历史 production A/C 的 6.15% 总回退交叉核对。

B 不能被解释为“只切换一个 fused 开关”的纯变量，C-B 也不能被解释成
“lambda 自身成本”；精确边界仍是上述 bundle。正式 PMU 数据出来前，这三项
只能作为实验问题，不能预填原因结论。

### 3.6 待选 D：保留 split runtime，但不能多一次热路径往返

如果 `B-A` 证明当前 fusion/outlining/layout 是明确回退项，就需要增加
`D=split lazy`。但 D 不能直接照搬下面的双 runtime 调用：

```text
submit API
  → runtime begin/claim 返回 POD ticket
  → orchestration callback 一次
  → runtime finish(ticket, args)
```

这虽然保持一次 callback，却把 eager 的一次 runtime 边界变成 begin + finish
两次，已经改变热路径，且 E1 已证明这种固定拆分成本可能覆盖参数节省。因此
它只适合作为语义清晰的原型或对照，不是首选性能实现。

当前更严格的 D 设计目标是：

```text
一个对外 submit API
  → runtime/codegen 专用 inline front 通过已由独立 probe 验证的 AIC/AIV 专用
     external [[block_local]] 状态（组合 submit 仍待验证）
     完成 task-id、execute-first drain 和 claim
  → 业务生成代码只提供 callback；front 在 caller 内调用它恰好一次并物化
     L0TaskArgs
  → 只跨一次 TU/runtime 边界调用 finish(ticket, L0TaskArgs)
```

这样在“业务 submit / callback / runtime 边界”三个维度分别保持 `1 / 1 / 1`，
不会仅为了 split 又新增一次 runtime 往返。不过它仍有两项尚未验证的变化：

1. claim 从 eager 的完整参数 materialize 之后移动到参数求值之前；这是 lazy
   的必要顺序变化，不是“原流程完全不动”；
2. runtime/codegen 内部 header 导入跨 TU external `[[block_local]]` 状态，必须
   先由最小 AIC/AIV probe 证明对象同源和 CCEC 生成形态；cursor/claim 不暴露
   给业务 orchestration，不能由源码可编译直接推定状态同源。

截至 2026-07-18，`nested_lambda_claim_first_once`（callback once + split
finish）和 `block_local_cross_tu`（AIC/AIV external `[[block_local]]`）都已
通过当前 CCEC 的编译、链接、ELF 符号检查和各自的真实 A5 oracle：

- `nested_lambda_claim_first_once`：4 blocks × 64 rounds，1/1 PASS；每轮唯一
  winner，callback 次数、winner-only input/scalar、finish digest 与 guard
  均精确匹配；
- `block_local_cross_tu`：AIC 5/5、AIV 1/1 PASS；每次 4 blocks × 257 rounds，
  `concurrent_block_arrivals == 4`，runtime/orchestration 看到的地址、终值、
  cookie、轮数和 guard 均精确匹配。

它们仍是两个独立 probe，因此只能称“两个前置边界分别上板通过”，不能称
“external state + callback once + finish 的 D 组合路径已经上板通过”。

虽然两个独立 probe 的设备 oracle 已通过，external state + callback once +
finish 的组合 submit oracle 仍未验证；即便组合 oracle 后续通过，D 与 A 的
直接差分仍会混入
`static [[block_local]] → external [[block_local]]` 的 state-export 改动。若
真正实现 D，必须同时重编一个 `A′=split eager + 同一 state-export patch`：

```text
D  - A′ = 同一 state-export patch 下，split 形态的 lazy front/finish、
          claim/materialize 重排与条件求值 bundle 的组合变化
A′ - A  = 其余源码/编译参数完全一致时，state-export 的组合变化
C  - B  = fused 形态下同一 lazy bundle 的组合变化
D  - C  = 保持 lazy 顺序时，state linkage + split/fused codegen 的组合变化
```

这样才形成可解释的 A/B/C/D 四角对照；A′ 是消除 D 特有 linkage 混杂的附加
控制，不是第五个业务候选。

所以 D 当前是**有硬门槛的候选设计**，尚不是已实现或已测得收益的结论。
是否进入 PA，要先由 A/B/C PMU 判断融合与 lazy bundle 各自的动态方向；两个
前置 probe 的设备门槛已经通过，下一项正确性门槛是二者组合后的真实 submit
oracle。任何 D 最终仍须用无 PMU 的真实 A5 production 三轮 A/B 验收。

## 4. 验证方法与口径

### 4.1 运行条件

历史 production 两组均满足：

- HEAD：`a3f5ecc2186fb8a06cded6d26886a4ab802009c0`；
- 108 个 replay worker；
- `TestPagedAttentionUnroll::Case1`；
- 真实 A5；
- 参数包含 `--enable-l2-swimlane --use-example-exec-time`；
- PTO-ISA 使用本机已有目录，没有下载；
- 每轮独立 pytest 进程。

测试命令形态：

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/python:$PWD/build/cp312-cp312-linux_x86_64/python/bindings"
export PTO_ISA_ROOT=/home/q00473782/atomic/codex/simpler-fully_distributed/build/pto-isa

python -m pytest \
  examples/a5/fully_distributed_within_core/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --case Case1 \
  --enable-l2-swimlane --use-example-exec-time -s -v
```

`--use-example-exec-time` 在 A5sim 的 `DIST_SIM_HOST_CLOCK` 路径才以参考时长
busy-wait；真实 A5 仍执行实际 incore kernel。此开关同时使框架跳过数值
golden，因此这里的 PASS 表示调度/执行流程完成，不能扩大为数值 golden 已
比较。候选 A5 三轮均 PASS，未出现 507015/507018。

A5sim 对应样本也 PASS，但其 host 调度时间和 Submit span 不参与任何性能
结论。

### 4.2 主指标

主指标严格沿用当前 PA 分析口径：

```text
所有 worker 中最早的 Submit.start
→ 所有 worker 中最晚的 Submit.end
```

它覆盖用户关心的片上 scalar orchestration/replay 主体，不包含 pytest wall
time、host 初始化、设备装载以及最终收尾时间；因此不会拿整段十几毫秒或
A5sim 的 host 调度时长冒充约 5–6 ms 的优化目标。

每核角色按 `fdwic_events[row][2]` 的 lane 判断：`0=AIC`、`1=AIV0`、
`2=AIV1`。不能用本批 JSON 的 `metadata.core_types` 做细分，因为当前 exporter
在 `fdwic_swimlane.cpp` 中按 `core_idx % 3` 生成该字段，它与实际 worker id
排列不一致。

### 4.3 三形态局部 PMU 对照

历史 eager/fused-lazy 对照同时改变了 runtime 编译形态和 QK 参数路径。局部
PMU 因此不再只做两个点，而是构造三种最终 AICore 形态：

| 形态 | orchestration | runtime 形态 | QK 参数处理 | 用途 |
| --- | --- | --- | --- | --- |
| A：split eager | 原始 eager | 原始 split TU | claim 前 eager 构造 | 基线 |
| B：fused eager | 与 A 相同的 eager 代码 | 链入与 C 相同的关键 fused helper/tail | claim 前 eager 构造 | `B-A` 测当前 fusion/outlining/layout 组合影响 |
| C：fused lazy | 单 callback lazy front | 与 B 相同的关键 fused helper/tail | claim 后按 winner/loser 求值 | `C-B` 测 eager→lazy 调用、claim/materialize 重排及参数条件化的组合变化 |

B/C 不是整个 ELF 字节相同；其可比性来自关键 runtime helper 的地址、尺寸和
combined object 机器字节相同，差异集中在 orchestration front。A/B 的 eager
orchestration entry 在 combined object 中也逐字节相同。因此三组差分的准确
含义为：

```text
B - A = 当前诊断构建中 fused TU、runtime outlining、额外 helper 链入与代码布局的组合影响
C - B = 已核验关键 helper 静态形态相同条件下，legacy eager submit→lazy front+finish、claim/materialize 重排与条件参数求值的组合影响
C - A = 当前 fused-lazy 候选相对 split-eager 的总变化
```

### 4.4 三个语义对称窗口

旧诊断窗口不能复用：旧 eager 在参数构造完成后才开窗，而旧 lazy 在 callback
之前开窗，覆盖范围不等价。本次每个 QK 累计三个互不连续的窗口：

| 窗口 | eager 覆盖 | lazy 覆盖 | 明确排除 |
| --- | --- | --- | --- |
| scope prepare | `qi` shape/offset 与 `Tensor::view` | `Tensor qi` 与 `qi_ready` 初始化 | 公共 `out_view` |
| caller args | `params.reset/add_input/add_output/add_scalar` | 对应 caller 位置为空；thunk 延后到 runtime | 其他 task 参数 |
| runtime | `EfDrain` 后的 materialize、claim、register、build/replay | `EfDrain` 后的 claim、唯一 callback/thunk、finish、register、build/replay | 前序 kernel 和 `EfDrain` 等待 |

Case1 每核有 256 次 QK，每次三个窗口，所以每核严格为 768 对 gate，108 核
合计 82,944 对。窗口只覆盖用户关心的 QK 参数/submit 路径；公共
`out_view`、Alloc、SF/PV/UP 以及前序 kernel 不计入。

### 4.5 STOP、EMPTY、QK 与逐核配对

每种机器形态各构建三个模式：

| 模式 | 行为 | 用途 |
| --- | --- | --- |
| STOP | 不打开成对测量窗口；不执行 `metrics_prof_start()`，但保留入口/出口的 stop-only 清理、selector/counter 读取和 synthetic row 写出 | 四项必须逐核精确为 0，证明 gate 未开启时没有形成有效计数窗口；STOP 本身仍有诊断开销 |
| EMPTY | 每个窗口入口立即 start/stop | 测相同次数 gate 的固定计数 |
| QK | start/end 包住真实目标逻辑 | 得到目标逻辑加 gate 的 raw count |

净值先在同一轮、同一物理核上计算，再做 lane 聚合：

```text
net(core, round) = QK(core, round) - EMPTY(core, round)
per-worker-QK    = net / 256
```

STOP 只作 oracle，不参与数值扣除；有符号的 `QK-EMPTY` 小幅负噪声原样保留，
不得截成 0。AIC/AIV0/AIV1 各 36 核分开报告。物理 miss rate 只对 raw
样本使用 lane 内 `SUM(raw miss)/SUM(raw request)`，不对 per-core ratio 求
平均。`QK-EMPTY` 和 A/B/C 差分只分别报告 signed request/miss，二者的比值
不是物理命中率，本文不计算。

### 4.6 v7/v8 PMU oracle 及证据边界

生产 `--enable-pmu 2` 由 AICPU 配置 PipeUtilization selector；AICore 只用
本地 `CTRL` SPR bit0 开关窗口，并通过 `ld_dev` 读取累计 counter。v7 在真实
A5 的 108 个 sub-core 上逐核实读得到：

```text
入口 selector2/6/7 = 0x1 / 0x34 / 0x35
出口 selector2/6/7 = 0x1 / 0x34 / 0x35
AICore ld_dev(MMIO PMU_CTRL_0) = 0（入口/出口均如此）
```

本机资料只规定 MMIO `PMU_CTRL_0/1` 由 AICPU 配置，没有给出 AICore 对
`PMU_CTRL_0` 的读回语义。因此这里仅把“读到 0”记录为当前机器/访问路径的
实测现象，不能据此声称它是 write-only、存在独立视图或实际没有启用计数。
v8 不再把该读值当门禁，而是同时校验：

- 108 核，AIC/AIV0/AIV1 各 36；
- 每核 1280 个 Submit、其中 256 个 QK；
- 432 条 synthetic PMU row，即 `108 × 4`；
- 入口/出口 selector 均为 `0x1/0x34/0x35`；
- AICore 本地 `CTRL` SPR bit0 在入口/出口均为 0；
- AICore `ld_dev` counter 读取语义可识别，末尾重复读取符合该语义；
- STOP 的 total/scalar/request/miss 逐核精确为 0；
- raw 数据满足 `scalar <= total`、`miss <= request`。

三种 v8 形态的 STOP r1 预检已分别上板，全部满足上述内容 oracle，flags
均为 `0x504d00fc`，且每核四项计数精确为 0。r1 没有 artifact/host/method
manifest；截至 2026-07-18 早期记录，带完整 manifest 的正式样本只完成
`split_eager STOP r2`，另一次 `fused_eager STOP r2` 因 `PEER_OVERLAP` 被拒绝。
后续虽然补齐了 21 行 manifest，但三种 QK 样本的 total counter 在全部 96 核上
仍为 0，当前严格解析器以 `QK_TOTAL_ZERO` 拒绝，因此不能把该归档当作有效 PMU
矩阵，也仍不能据此计算任何正式 `QK-EMPTY` 三向动态差分。

本次 AICore `ld_dev` 对 counter 的实测是 nondestructive，因此用入口快照从
结束快照中做差；这与 AICPU `read_reg` 路径的 read-to-clear 语义不能混为
一谈。selector 只在入口/出口核验，不能发现中途被改后又恢复的瞬时干扰，
所以正式 EMPTY/QK 仍要求设备独占。

### 4.7 2026-07-19 迁移机 96-worker A/B/C STOP 泳道复测

为先复现 simpler 路径的性能方向，2026-07-19 在迁移机真实 A5 device 0 上，
使用 clean `76df85ced5a43099d268cd6bcd5a988d03d9cca7` host 和冻结的 A/B/C STOP
工件，重新采集三轮泳道。host signature 为：

```text
67ac5a584799bd06c449595d46b6dcb9836b66c9e2c89083fa91f0f1e3dc7846
```

三轮均使用 `TestPagedAttentionUnroll::Case1`、
`--enable-l2-swimlane 1 --enable-pmu 2 --use-example-exec-time`，每个样本为独立
pytest 进程；顺序分别为 `A -> B -> C`、`C -> B -> A`、`B -> C -> A`。
迁移机拓扑为 96 worker，即 AIC/AIV0/AIV1 各 32 核，不能与第 5 节的 108-worker
production 数值混算。runner 没有报告当前用户可见进程的 peer overlap；这不是
驱动级全系统枚举，不能扩大成 device 0 绝对空闲证明。

主指标仍严格取所有 worker 中最早的 Submit start 到最晚的 Submit end：

| 轮次 | 顺序 | A split eager | B fused eager | C fused lazy | `B-A` | `C-B` | `C-A` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A、B、C | 5.143684 ms | 5.233890 ms | 5.398408 ms | +1.754% | +3.143% | +4.952% |
| 2 | C、B、A | 5.310360 ms | 5.110000 ms | 5.404423 ms | -3.773% | +5.762% | +1.771% |
| 3 | B、C、A | 5.211785 ms | 5.124709 ms | 5.447396 ms | -1.671% | +6.297% | +4.521% |

按形态分别取三轮中位数后再相减：

| 形态 | Submit span 中位数 | 中位数差 |
| --- | ---: | ---: |
| A | 5.211785 ms | — |
| B | 5.124709 ms | `B-A = -0.087076 ms / -1.671%` |
| C | 5.404423 ms | `C-B = +0.279714 ms / +5.458%`；`C-A = +0.192638 ms / +3.696%` |

“各形态中位数相减”不能与“逐轮配对差再取中位数”混写；后者的绝对差分别为
`B-A=-0.087076 ms`、`C-B=+0.294423 ms`、`C-A=+0.235611 ms`。

内容 oracle 与时间戳完整性为：

- 9/9 个 pytest 进程通过；`--use-example-exec-time` 在真实 A5 仍执行真实 incore
  kernel，但跳过数值 golden，因此这里只能写调度/执行流程 PASS；
- 每个样本都是 96 核、每核 1280 个连续且唯一的 task id `0..1279`，合计
  122,880 个 Submit；
- 每个 Submit 均满足正时间戳、`end > start`，每核按 task id 严格递增且相邻
  Submit 不重叠；
- 每次 runner 均重新校验 artifact SHA256 与 host signature，未出现失败样本或
  output 集合歧义。

`C-B` 的 per-core Submit span p50 进一步显示回退集中在 AIV：

| lane | 第 1 轮 | 第 2 轮 | 第 3 轮 |
| --- | ---: | ---: | ---: |
| AIC | +2.001% | +2.122% | +2.827% |
| AIV0 | +6.146% | +6.505% | +6.474% |
| AIV1 | +6.152% | +6.324% | +6.508% |

因此本次复测可以确认：`C-B` 三轮方向一致地回退，形态中位数回退 5.458%，且
两个 AIV lane 三轮均超过 5%；但全局第 1 轮只有 3.143%，不能写成“全局每轮
稳定超过 5%”。`B-A` 方向混合，不能从该组数据判定 fusion/layout 的稳定净
影响；完整 `C-A` 三轮都低于 5%，中位数为 3.696%。与 2026-07-18 的单份 STOP
样本相比，本轮 B/C 中位数只分别漂移 +0.059%/+0.223%，而 A 漂移 +2.370%，
所以 `C-B` 回退方向的复现强于 `B-A` 和 `C-A` 的量级复现。

这组数据仍然只是带泳道、PMU owner、stop-only 清理与 synthetic row 的 STOP
诊断工件。它可以比较冻结 A/B/C 在同一观察口径下的方向，不能冒充无诊断
production 性能，也没有提供有效 I-cache counter，更不能单凭回退把原因归给
lambda 或 I-cache。

## 5. 真实 A5 production 对照

### 5.1 原始样本

| 版本 | 输出目录 | 全局 Submit span |
| --- | --- | ---: |
| eager | `eager_baseline_a3f5ecc2/..._220609` | 5.218171 ms |
| eager | `eager_baseline_a3f5ecc2/..._220925` | 5.168611 ms |
| eager | `eager_baseline_a3f5ecc2/..._221205` | 5.205719 ms |
| fused lazy QK | `simpler/outputs/..._234917` | 5.510962 ms |
| fused lazy QK | `simpler/outputs/..._235222` | 5.630435 ms |
| fused lazy QK | `simpler/outputs/..._235441` | 5.525736 ms |

| 统计 | eager | fused lazy QK | 差异 |
| --- | ---: | ---: | ---: |
| 三轮中位数 | 5.205719 ms | 5.525736 ms | **+0.320017 ms / +6.15%** |

历史文档中的 5.115620 ms 中位数和 5.096685 ms 最好值来自 96-worker 快照，
只能做纵向参考，不能和本轮 108-worker 样本混成同条件对照。上表各样本的
运行条件相同；它们是同 HEAD、同条件的重复对照，但没有逐轮交错配对，也不是代码
变量上的单变量实验，因为 candidate 同时改变了参数路径和 TU/codegen 形态。

### 5.2 按真实 lane 的关键路径

下表不是把 108 个核直接混在一起求一次均值，而是：每轮分别对 36 个 AIC、
36 个 AIV0、36 个 AIV1 的 per-core `CoreSpan` 取 p50，再对三轮 p50 取中位数。
这样既保留角色差异，也避免某一个异常核支配结论。

| lane | eager CoreSpan | 候选 CoreSpan | 差异 |
| --- | ---: | ---: | ---: |
| AIC | 4.959469 ms | 5.116599 ms | +157.130 us / +3.17% |
| AIV0 | 5.040790 ms | 5.395352 ms | +354.562 us / +7.03% |
| AIV1 | 5.037745 ms | 5.402776 ms | +365.031 us / +7.25% |

三轮全局最后结束的 critical core 均为 AIV1。两个 AIV lane 都系统性回退，
不是单个异常核或 host 抢占样本。

### 5.3 QK 阶段分解

下表是每核 256 个 QK 的三轮累计中位数，单位 us：

| 阶段 | AIC：候选 / eager / 差值 | AIV：候选 / eager / 差值 |
| --- | ---: | ---: |
| Submit | 825.653 / 712.536 / **+113.117** | 878.383 / 593.735 / **+284.648** |
| EfDrain | 60.184 / 50.700 / +9.484 | 37.480 / 20.569 / +16.911 |
| Materialize | 116.395 / 129.255 / -12.860 | 152.859 / 130.534 / +22.325 |
| PrepareMap | 56.120 / 53.713 / +2.407 | 68.447 / 54.154 / +14.293 |
| Claim | 139.169 / 110.166 / +29.003 | 45.284 / 25.334 / +19.950 |
| Fanin | 6.996 / 8.163 / -1.167 | 0 / 0 / 0 |
| Register | 2.148 / 46.977 / -44.829 | 10.584 / 30.459 / -19.875 |
| 未标记 residual | 427.973 / 308.254 / **+119.719** | 563.432 / 332.550 / **+230.882** |

这里的 residual 是 `Submit` 减去其中已单独记录、互不重叠的 runtime phase，
主要覆盖 callback/thunk、未单独埋点的 submit 框架、call/return 及其最终
codegen 开销。

逐次调用更直接：

- AIC loser：Submit +401 ns，其中 residual +455 ns；
- AIC winner：Submit +1,896 ns，其中 residual +1,871 ns，包含 winner-only
  `qi.view`；
- AIV loser：Submit **+1,105 ns**，其中 residual **+909 ns**。

AIV 不参与 QK 的 cube-cursor `atomicMax`。所以上表中 AIV Claim 区间增长
不能解释成 atomic 竞争；它是角色判断、trace 和邻近固定路径的计时区间。

当前 fused-lazy 整体候选中，本应最轻的 AIV loser 每次 QK 多付约 1.1 us，
累计 256 次，可覆盖大部分 AIV CoreSpan 增量。它证明候选的 QK residual
增长超过了所省参数工作的可见收益，但不能把 residual 全部归因给 lazy
callback；历史 A/C 同时包含 fusion/codegen 变化。

### 5.4 非 QK、gap 与背压

每核非 QK Submit 差异为：

| task | AIC | AIV |
| --- | ---: | ---: |
| Alloc | -7.9 us | -20.8 us |
| SF | +40.5 us | +34.7 us |
| PV | +1.9 us | +42.9 us |
| UP | +1.4 us | -107.5 us |

AIV 的 Submit 外 gap 还累计增加约 150 us，其中下一批 Alloc 前增加约
91.5 us。最终 task 的 start 波宽从 256.155 us 增至 427.970 us。

候选三轮还分别出现 3/3/4 次 ring-slot `RingBp`，eager 为 0。它们全部
发生在跑得相对更快的 AIC，而不在最终 critical core；更合理的解释是 AIV
滞后后导致的背压症状，而不是把 RingBp 时长直接当作全局回退根因。

四类 kernel 的累计时间只从约 32.468 ms 变为 32.509 ms，差约 0.13%。
因此 kernel 时长变化不足以解释主要回退，证据更支持
orchestration/submit、Submit 外 gap 与由核间相位差引起的背压。

### 5.5 与历史优化序列的关系

此前已经记录过如下演进，它说明 private TensorMap 的既有优化确实把主体时间
从约 6.17 ms 压到约 5.1 ms；本轮不是重新定义这些历史收益：

| 历史阶段 | A5 主体时间 | 相对最初累计提升 |
| --- | ---: | ---: |
| 原始四个参数容器 | 6.170876 ms | — |
| 四个 kernel 复用一个容器 | 5.720638 ms | 7.30% |
| 未使用 TensorRef 槽延迟初始化 | 5.645826 ms | 8.51% |
| alloc 与四个 kernel 共用 scope 内唯一容器 | 5.577570 ms | 9.61% |
| `e3b748b4` runtime 热路径优化 | 5.115620 ms | 17.10% |

但是最后一行和更早阶段来自 96-worker 历史快照；本轮 HEAD 自动解析出了
108 个 replay worker。因此历史表只用于说明优化演进，不能把 5.115620 ms
直接当作本轮 candidate 的分母。对 fused lazy QK 唯一有效的定量结论仍是
同 HEAD、同 108-worker、同命令的 `5.205719 → 5.525736 ms`，即回退 6.15%。

## 6. CCEC 二进制形态发生了什么

尺寸只统计可执行 `.text` 或具体 FUNC symbol，不使用包含 DWARF、符号表的
`.o` 磁盘大小。

| 对象/符号 | eager | fused lazy QK | 变化 |
| --- | ---: | ---: | ---: |
| AIC `aicpu_orchestration_entry` | 3,244 B | 16,640 B | +13,396 B / +412.95% |
| AIV `aicpu_orchestration_entry` | 3,220 B | 16,736 B | +13,516 B / +419.75% |
| AIC QK finish | 无 | 11,420 B | 新增 |
| AIV QK finish | 无 | 11,520 B | 新增 |
| AIC combined `.text` | 96,408 B | 121,208 B | +24,800 B / +25.72% |
| AIV combined `.text` | 110,160 B | 135,248 B | +25,088 B / +22.77% |
| 最终 ELF `.text` | 206,672 B | 256,592 B | +49,920 B / +24.15% |

还有一个比“总量变大”更重要的变化：baseline 的 `dist_submit_impl` 约
42.2 KiB；fused 编译后该符号缩小到约 19.7 KiB，同时新出现 AIC/AIV 各
约 22.5 KiB 的 `dist_submit_build_winner_task` out-of-line 函数。这证明
CCEC 不只是增加了 QK helper，而是重新决定了旧 runtime 的 inlining、
outlining 和代码布局。

因此 alloc/SF/PV/UP 虽然源码没有修改，也不能再当成“二进制完全不变的
对照路径”。本轮 fused 实验不是只改变 QK 参数求值的纯单变量实验。

这也是对 `make_replay_faster.md` 第 7 节的一处实质回应：

> “旧 API 源码仍在”不足以叫作“旧路径保留不变”。性能优化还需要检查最终
> ELF 的符号尺寸、调用边界和旧 task 的泳道阶段。

当前没有栈不足证据：候选三轮都正常完成，没有 507015/507018；性能回退
主要由 QK residual 与 Submit 外 gap 共同构成。HiIPU 对象的 DWARF relocation 也使本机
`readelf` 难以把每条 FDE 可靠映射到具体 production symbol，因此本文不以
某个逐函数 CFA 数字作为根因证据。

### 6.1 三形态 v8 诊断 ELF 的机器码控制

为避免把历史 A/C 的两个变量继续混在一起，v8 对 exact QK artifact 又做了
A/B/C 三形态审计。下表只统计可执行 `.text` 和 FUNC size：

| 对象/符号 | A：split eager | B：fused eager | C：fused lazy |
| --- | ---: | ---: | ---: |
| AIC orchestration entry | 3,344 B | 3,344 B | 16,808 B |
| AIV orchestration entry | 3,320 B | 3,320 B | 16,748 B |
| AIC `dist_submit_impl` | 42,196 B | 20,004 B | 20,004 B |
| AIV `dist_submit_impl` | 42,124 B | 20,028 B | 20,028 B |
| AIC lazy finish | 无 | 11,464 B | 11,464 B |
| AIV lazy finish | 无 | 11,564 B | 11,564 B |
| AIC combined `.text` | 98,384 B | 110,168 B | 123,632 B |
| AIV combined `.text` | 112,208 B | 124,240 B | 137,552 B |
| final ELF `.text` | 210,768 B | 234,576 B | 261,200 B |

combined object 的原始机器字节进一步确认：

- A/B 的 eager entry 逐字节相同（AIC/AIV 均为 0 B diff）；
- B/C 的 AIC `dist_submit_impl` 地址、尺寸、SHA256 完全相同；
- B/C 的 AIV `dist_submit_impl` 地址、尺寸、SHA256 完全相同；
- B/C 的 AIC/AIV lazy finish 地址、尺寸、SHA256 完全相同；
- B/C 的 entry 起始地址相同，尺寸分别由 `3344→16808 B`、
  `3320→16748 B`。

因此 B/C 已核验的 `dist_submit_impl` 与 lazy finish helper 静态形态已经
对齐，B→C 的静态增量主要集中在 orchestration entry；动态上 B 的 QK 仍走
legacy `dist_submit_impl`，C 的 QK 则走 lazy front + finish，这正是要测的
变量。A→B 的 final `.text` 增加
23,808 B（11.30%），同时 `dist_submit_impl` 被重新 outlining，代表 fusion/
codegen 形态变化。

final ELF 中 B/C 的对应 helper 仍有少量链接 fixup 字节差异：AIC
`dist_submit_impl/finish` 分别 17/6 B，AIV 分别 11/6 B。因为 combined
object 的地址、尺寸和原始代码完全一致，这些差异与最终地址重定位/链接
fixup 相符；本机 objdump 无法解码 HiIPU 指令，因此这只是当前证据支持的
解释，不能声称已逐指令证明语义同一。

v8 的 selector oracle 使 B/C `dist_core_main` 共同增加少量代码，但没有改变
上述 B/C helper 的地址、尺寸或 hash，也没有改变 A/B eager entry 的机器字节。
所以诊断门禁不会破坏三向比较的静态控制关系；诊断 ELF 的绝对 `.text` 数字
则不能冒充历史 production ELF。

## 7. 代码段变大为什么可能影响 I-cache

### 7.1 本机资料能确认到什么程度

当前实机记录为 `Ascend950PR_958b`，本机 CANN platform config 给出
`NpuArch=3510`、AIC `dav-c310-cube`、AIV `dav-c310-vec`。

本机 DAV3510 simulator 资料存在两类配置：

- 9599 CA 默认的 `1982_cloud_config.toml`：AIC scalar I-cache 32 KiB，
  AIV 16 KiB；二者均为 4-way、128 B cache line；
- `cpro_light_config.toml` 等变体：AIC/AIV 均为 16 KiB，同样 4-way、
  128 B line。

本机没有找到“958b 精确选择哪份 PEM 配置”的映射。因此不能把“958b AIC
一定是 16 KiB”或“一定是 32 KiB”写成已证事实。AIV 16 KiB 在这些相关
配置中是一致的；AIC 应同时按 16/32 KiB 两种边界分析。

对应的本机原始字段是：

```text
1982_cloud_config.toml:291-299
  aic_ic_size=32768, aiv_ic_size=16384, 4-way, 128 B line

cpro_light_config.toml:117-125
  aic_ic_size=16384, aiv_ic_size=16384, 4-way, 128 B line
```

也就是说，“scalar I-cache 是 16K”对 AIV 有一致的本机资料支持；对这块
958b 的 AIC，当前资料还不足以排除 32K。这里区分 AIC/AIV 很重要，因为
本轮最终关键路径和最大回退恰好出现在 AIV。

### 7.2 容量和冲突风险如何产生

I-cache 缓存的是实际取过的 scalar 指令行，不是把整个 ELF 一次性装入。
函数调用不会主动清空 I-cache；只有 callee 取指映射到相同 set、替换了
caller 的 line，返回 caller 后才需要重新取指。

候选静态尺寸为：

```text
orchestration entry: 16,640 B（AIC）/ 16,736 B（AIV）
QK finish:          11,420 B（AIC）/ 11,520 B（AIV）
```

按 128 B line 计算，AIV entry 静态覆盖约 131 条 line，已经略多于 16 KiB
4-way cache 的 128 条总 line；entry 与 finish 合计约 221 条 line。若这些
symbol 的 line 在短复用窗口中都被执行：

- 按 16 KiB、32 set、4-way，entry 单体已经会让少数 set 出现第五条 line；
  entry+finish 则所有 set 都会超出四路；
- 即便按 32 KiB、64 set、4-way，entry+finish 的 221 条静态 line 平均也有
  约 3.45 line/set，给其他 hot helper 留下的冲突余量很小；实际哪些 set
  超过四路仍取决于两段起址映射，不能固定写成 29 个 set。

PA 每个 batch 反复执行 QK→SF→PV→UP。候选在 orchestration、QK finish、
通用 `dist_submit_impl` 和被 outline 的 winner helper 之间切换，所以容量
miss 与 set-conflict miss 风险都比单看一段顺序代码更高。2 KiB 预取单元
来自前述相关 DAV3510 simulator model，只能说明可能的实现边界，不能直接
断言 958b 实机采用相同配置；即使存在，它也只能缓解顺序取指，不能保证跨
多处分支/call 的热工作集不被置换。

这说明为什么 AIV 必须纳入分析：虽然 QK kernel 只由 AIC winner 执行，
每个 AIV worker 仍重放完整 orchestration，并对每个 QK 走 loser submit。
实测 AIV CoreSpan 回退 355–365 us，显著大于 AIC 的 157 us；AIV loser 的
QK residual 每次增加 909 ns。这种 lane 分布值得检查 AIV 的动态前端计数，
但在 B-A/C-B PMU 数据出来前，不能把它写成 I-cache 已经导致回退的证据。

需要强调，“超过 16 KiB”带来的不是一个编译期硬错误，也不是调用时整段
函数一次性装入 cache。它只是意味着：如果下一次复用某条旧指令之前，期间
实际取过的同 set 指令超过四条，该旧 line 会被替换；随后再执行它才产生
miss。候选每个 worker 反复执行 256 组 QK/SF/PV/UP，因而这类“跨 submit
返回后重取旧热代码”的代价可能累计放大。是否真的发生，仍取决于动态分支
覆盖、符号地址映射、替换策略和预取，不能由静态字节数单独决定。

### 7.3 静态尺寸为什么不能替代实机 PMU

静态 symbol size 只是动态 footprint 的上界：

- 未走到的 cold branch 不会成为动态工作集；
- entry 覆盖整段 PA，单次 QK 只执行其中一部分；
- finish 内也有 winner/loser 条件分支；
- replacement policy 和实际地址映射不能由 ELF 尺寸还原；
- baseline 的 `dist_submit_impl` 本身约 42.2 KiB，却仍比候选快，直接说明
  “单个大函数超过 cache 容量”不是充分根因。

另一个反证式提醒是：若只把 `entry + 被调 tail` 的整个静态 symbol 相加，
baseline 约为 3.2+42.2 KiB，候选约为 16.6+11.4 KiB，候选反而更小。
这再次说明必须比较实际执行的 line，而不能按函数总字节数直接判定 miss。

因此当前证据强度应写成：

| 判断 | 证据强度 |
| --- | --- |
| fused TU 显著改变了代码布局和调用边界 | 已证实 |
| orchestration entry 膨胀约 5.1 倍 | 已证实 |
| AIV QK residual 与全局关键路径稳定回退 | 已证实 |
| fusion 是否增加动态 I-cache 压力 | 必须看 `B-A` 的 request/miss 与 `total-scalar` |
| lazy front 是否增加动态 I-cache 压力 | 必须看 `C-B`，不能拿整个 `C-A` 代替 |
| 本轮 6.15% 主要由 I-cache miss 导致 | 当前 production A/C 与静态尺寸均不能证明 |

除 I-cache 外，至少还要保留这些候选原因：

- lambda/front 带来的额外角色判断、trace、参数分支和 call/return；
- inline 后活跃值增多导致寄存器 spill；
- CCEC 重新 inlining/outlining 后的分支、对齐和调度变化；
- claim-first 改变各 worker 的相位，放大 AIV 长尾；
- QK finish 本身的固定成本。

## 8. 局部 PMU 的指标含义与定因方法

第 4.3～4.6 节的局部 gate 已经直接围住 QK 参数/submit，不再拿整颗
persistent kernel 的 task-based PMU 代替局部证据。A5 PipeUtilization 中使用：

```text
0x1   scalar_instr_busy
0x34  icache_req
0x35  icache_miss
fixed PMU_CNT_TOTAL0/1
```

### 8.1 四项 raw count 分别代表什么

| 指标 | 本文中的含义 | 不能解释成 |
| --- | --- | --- |
| total cycles | 三个开窗区间累计经过的 AICore PMU cycle | 不是整颗 persistent kernel wall-time，也不能按 JSON 的 1 GHz trace 时钟直接换算 ns |
| scalar busy | scalar pipeline busy cycle | 不是动态 scalar 指令条数；等待 dependent atomic 返回也会计入 |
| I-cache request | scalar 前端发出的取指请求 | 不是源码行数，也不是每执行一条指令就请求一次 |
| I-cache miss | request 未命中 I-cache 的事件数 | 不是具有严格固定 penalty、可直接相加成 wall-time 的事件 |
| raw `total-scalar` | 同一 raw 样本内未被 scalar-busy 覆盖的剩余 cycle | 不是纯 I-cache stall，还可能包含其他等待或前端空洞 |

仓内独立最小 probe 已在同一 A5 上确认两个判读边界：

- dependent `atomicAdd` 相对 scalar control 增加的 total cycle，几乎 100% 同步
  增加到 `scalar_instr_busy`；
- 同一 8 KiB scalar target 的额外 68 次 I-cache miss 带来约
  `2309..2312` total cycle，但 scalar busy 只增加 48 cycle，说明 refill
  等待的绝大多数落在 `total-scalar`。

它们帮助区分方向，但不是互斥、完备的 stall 分类，也不能把 PA 的 miss 数乘
约 33 cycle 当作精确贡献。远端 `640efe50` 又用 96 核 cold/warm、每个 cold
trial 严格增加一个 CNT7 miss 的方法完成 15 轮标定，给出约
`T_icache_est_ns = CNT7_miss_total × 90` 的一阶等效标尺。该值可以回答数量级，
但仍受预取、miss 重叠、下级命中位置和并发排队影响，不能作为严格可加的
关键路径 stall。净 non-scalar 按
`(QK.total-QK.scalar)-(EMPTY.total-EMPTY.scalar)` 计算；由于两项来自独立
ELF，它允许出现小幅负值，只是校准后的诊断量，不是严格非负的 stall cycle。

### 8.2 三向差分如何判读

| 观测 | 更支持的解释 | 仍需保留的边界 |
| --- | --- | --- |
| `B-A` scalar 增加，miss/request 基本不变 | fusion/outlining 带来额外控制、等待或 scalar 路径变化 | scalar busy 不能区分执行与 atomic 等待 |
| `B-A` miss 与 `total-scalar` 同向增加 | fusion/code layout 增加前端压力 | 可用 `miss × 90 ns` 看一阶量级，不能按静态 `.text` 或该标尺精确分摊关键路径 |
| `C-B` scalar 下降 | 与 winner-only 参数求值减少工作方向一致 | C-B 同时切换调用路径并重排 claim/materialize，不能唯一归因某个 thunk |
| `C-B` scalar 上升 | lazy 调用 bundle 的成本超过所省参数 | bundle 含 callback/角色分支/ticket/finish、claim 时机与 atomic 竞争相位，需结合 AIC/AIV 和 Claim/Residual |
| `C-B` miss 与 `total-scalar` 上升 | lazy front/entry 布局增加 I-cache 压力 | 只适用于当前 fused 机器形态，不否定 lazy 语义 |
| PMU 局部变化很小但 production span 回退 | 回退主要在窗外 gap、核间偏斜或背压 | 检查 inter-submit gap、RingBp 和 critical lane |

除逐核 `QK-EMPTY` 净值外，还保留 raw QK 的 A/B/C 直接差分。原因是 EMPTY
与 QK 是不同诊断 ELF，虽然 gate 次数相同，局部 stop 的位置仍会带来少量
机器布局差异。只有 raw 与净差方向一致、三个逐轮 paired delta 的方向与量级
相对 run-to-run IQR 也稳定时，才把结论写成当前机器实现的机制证据。

### 8.3 统计口径

每一轮先按相同 core index 计算 `QK-EMPTY`，再除以 256 得到每个
worker-QK 的值。每个 lane 同时报告：

- 36 核 sum / `(36×256)`，表示每次 replay attempt 的 lane 平均硬件工作；
- 36 核 per-core p50 与空间 IQR；
- 三个独立进程的中位数与 run-to-run IQR；
- raw 样本的聚合物理 miss rate：`SUM(raw miss)/SUM(raw request)`；
- 净计数和 A/B/C 差分只报告 signed request/miss，不计算二者比值。

不同 run 的 36 样本不池化，也不做“两个中位数相减”。第一轮按
`A-empty/A-QK → B-empty/B-QK → C-empty/C-QK` 成对采样；第二轮同时反转
shape 顺序和 pair 内顺序，第三轮再旋转 shape。这样可观察 EMPTY/QK
先后顺序带来的漂移，但三轮样本的 IQR 仍只是描述量，不能代替逐轮原值。

## 9. 对 `make_replay_faster.md` 第 7/8.2 节的正式回应

### 9.1 已被验证的部分

1. **当前 fused 候选保持了单 callback。** 其每个 worker 的每次 submit
   只调用一次 builder，winner/loser 由这一次 builder 决定哪些 thunk
   求值；这不代表待选 D 的 split 组合边界已验证。
2. **当前 fused 候选的 claim-first 控制流可以运行。** task id、
   execute-first drain、claim、参数求值、output/map、winner build 的顺序
   已由真实 PA 流程与已持久化的结构 oracle 跑通。两个独立最小 probe 的
   设备 oracle 也已分别通过，但它们没有覆盖组合 submit；本轮 PA 也没有
   数值 golden，不能据此声称组合边界或端到端数值已经等价。
3. **private loser 只能跳消费侧。** fresh output、heap、返回句柄和 private map
   一致性仍须保留。
4. **源码层理论工作量下降。** 按 Case1 控制流推导，QK 的 view、
   input/scalar 写入和 slot 扫描均大幅减少；这不等同于 PMU cycle 或最终
   wall-time 已下降。

### 9.2 当前实现为什么不合格

当前能够直接下的结论只有：

1. 历史 production A/C 同时改变 fusion/codegen 与 lazy 参数路径，不是单变量
   实验；
2. 当前 fused-lazy 整体候选的真实 A5 中位数回退 6.15%，因此不能提交；
3. 回退集中在 orchestration/submit，AIV loser 每个 QK 多约 1.1 us，kernel
   累计时间基本不变；
4. 源码层面的参数工作量虽然下降，但不等价于 scalar cycle 或 wall-time
   一定下降；
5. 静态机器码证明 fusion 改变了 outlining/layout、lazy 扩大了 entry，但
   不能据此判定二者各自的动态代价，更不能先验归因 I-cache；
6. 2026-07-19 的 96-worker STOP 泳道复测中，`C-B` 三轮均回退
   `3.143%/5.762%/6.297%`，两个 AIV lane 的 per-core p50 三轮均回退超过 5%；
   但 `B-A` 方向混合、完整 `C-A` 三轮均低于 5%，且 STOP 不是 production
   性能构建。

正式原因必须由三形态数据拆开：`B-A` 回答当前 fusion/outlining/layout 组合
的净影响；`C-B` 回答在已核验 helper 静态形态相同条件下，从 legacy eager
submit 整包切换到 lazy front+finish、claim/materialize 重排与条件参数求值
的净影响，不能进一步拆成 lambda 捕获、角色判断、ticket/finish、atomic
相位或参数节省中任一项的唯一贡献；`C-A` 再与历史 production 总回退交叉
核对。局部 PMU 只能把原因收敛到当前 bundle 的 scalar/front-end 类别；最终
是否有性能收益仍由无 PMU 的真实 A5 Submit span 验收。

所以本轮最大的价值不是得到了一版可提交优化，而是同时得到两个边界：

> “一次 callback”不需要退回两次调用；当前这版“完整 runtime +
> orchestration fused TU + lazy front”的组合实现性能不合格。三形态有效 PMU 完成
> 前，不能把这个结论扩大成单独否定 fused TU 或 lazy 语义。

### 9.3 下一版硬门槛

- 保持原 `dist_engine` 为独立 TU；旧 eager control 及 alloc/SF/PV/UP 的
  `dist_submit_impl` 调用形态必须稳定。D 的 QK 有意改走 inline front +
  finish，不能要求它同时仍动态调用旧 `dist_submit_impl`；
- alloc/SF/PV/UP 不仅源码不变，最终符号尺寸和关键阶段也不得系统性漂移；
- QK 仍必须是一次业务 submit、一次 callback、一次 claim；
- `nested_lambda_claim_first_once` 和 `block_local_cross_tu` 的独立设备 oracle
  已通过；下一步仍须通过 external state + callback once + finish 的组合
  submit oracle，两个独立 PASS 不能代替组合验收；
- 检查 orchestration entry 尺寸，避免再次出现约 5 倍膨胀；
- A5sim 只跑一次流程验证，不做性能分析；
- 无 PMU production ELF 在真实 A5 上至少三轮严格 A/B，AIC/AIV lane 均
  不得稳定回退；
- 任何 I-cache 解释都必须引用第 4.3～4.6 节同窗口、同 oracle 的 A/B/C
  `scalar_instr_busy/icache_req/icache_miss` 净计数，不能由 symbol size 推断；
- 没有正收益就立即回退，不扩散到 SF/PV/UP。

A/B/C 完成后的决策顺序固定为：

| 实测方向 | 下一步 |
| --- | --- |
| `B-A` 稳定回退、`C-B` 稳定改善 | fusion 是明确负项而 lazy bundle 有动态收益；两个前置 probe 已通过，先补组合 submit oracle，再推进 §3.6 的 D |
| `C-B` 稳定回退 | lazy front/finish、claim 重排或条件求值的组合本身有问题；先做更小控制拆 bundle，不直接实现 D |
| `B-A` 与 `C-B` 都稳定回退 | fusion 和 lazy bundle 分别处理；不能期望只改回 split TU 就得到正收益 |
| 三轮方向混合或量级接近噪声 | 增加独立进程样本并检查设备重叠；不据中位数符号推进业务修改 |

这里的“稳定”要求三轮逐轮方向一致，不能只看三轮中位数的符号。
按 §4.7 的本次复测，`C-B` 已满足“方向一致地回退”，对应上表第二行：下一步
应先在 standalone 中构造更对称的小控制拆解 bundle，而不是直接实现 D。其
5.458% 是形态中位数差，不代表全局三轮都超过 5%，也不能直接触发 I-cache
唯一归因。

## 10. 证据路径

本节首先保留历史实验的原始定位。`/home/q00473782/atomic/sfd/gpt/` 路径当前
机器不存在；`/home/q00473782/atomic/private/3-gpt/` 下的冻结工件属于旧
worktree 或 ignored 输出，不属于当前 cleanroom 提交。它们只提供历史 provenance，
不能替代本分支重跑。

### 10.1 历史 Production 性能样本

严格 eager trace：

```text
/home/q00473782/atomic/sfd/gpt/eager_baseline_a3f5ecc2/outputs/
  TestPagedAttentionUnroll_Case1_20260717_220609/l2_swimlane_records.json
  TestPagedAttentionUnroll_Case1_20260717_220925/l2_swimlane_records.json
  TestPagedAttentionUnroll_Case1_20260717_221205/l2_swimlane_records.json
```

候选 trace：

```text
/home/q00473782/atomic/sfd/gpt/simpler/outputs/
  TestPagedAttentionUnroll_Case1_20260717_234917/l2_swimlane_records.json
  TestPagedAttentionUnroll_Case1_20260717_235222/l2_swimlane_records.json
  TestPagedAttentionUnroll_Case1_20260717_235441/l2_swimlane_records.json
```

A5sim 流程样本（不用于性能）：

```text
/home/q00473782/atomic/sfd/gpt/simpler/outputs/
  TestPagedAttentionUnroll_Case1_20260717_234457/l2_swimlane_records.json
```

ELF：

```text
eager:
/home/q00473782/atomic/sfd/gpt/eager_baseline_a3f5ecc2/build/cache/a5/onboard/
  fully_distributed_within_core/aicore-extra/b5b7de88d4776700/aicore/

candidate:
/home/q00473782/atomic/sfd/gpt/simpler/build/cache/a5/onboard/
  fully_distributed_within_core/aicore-extra/311f86f6e5c5dd1a/aicore/
```

### 10.2 历史 v8 局部 PMU 诊断物

三形态各有 STOP/EMPTY/QK 三个预编译 AICore ELF；测试时通过 override 注入，
不会在采样进程中重新编译。诊断 ELF 的 wall-time 不能替代 production 性能
验收；§4.7 只在三个冻结 STOP 形态内部比较方向。

| 形态/模式 | `aicore_kernel.o` | SHA256 |
| --- | --- | --- |
| A split eager / STOP | `eager_baseline_a3f5ecc2/build/lib/a5/onboard/fully_distributed_within_core/aicore-extra/pmu_diag_split_eager_stop_v8/` | `62eac678ab97be1227a7fc41d7851df376212eb93c3f84556ed75f31b084f9a5` |
| A split eager / EMPTY | `eager_baseline_a3f5ecc2/build/lib/a5/onboard/fully_distributed_within_core/aicore-extra/pmu_diag_split_eager_empty_v8/` | `7b2997519574ef6d79b80aabddb0bbd3cc03ceb7825912c600d36b2df4e8f988` |
| A split eager / QK | `eager_baseline_a3f5ecc2/build/lib/a5/onboard/fully_distributed_within_core/aicore-extra/pmu_diag_split_eager_qk_v8/` | `2babea42983acae47289869fbfaa5b7aea75de65c1e6b3552d599af520463920` |
| B fused eager / STOP | `fused_lazy_qk_candidate_a3f5ecc2/build/lib/a5/onboard/fully_distributed_within_core/aicore-extra/pmu_diag_fused_eager_stop_v8/` | `e4104f8aa0523e06e4640aa1a806f749b803ea99405d23fb03c10950b552674c` |
| B fused eager / EMPTY | `fused_lazy_qk_candidate_a3f5ecc2/build/lib/a5/onboard/fully_distributed_within_core/aicore-extra/pmu_diag_fused_eager_empty_v8/` | `1a024a7e5717181fb685908e179e36b351dad56f6a8d78686073d6f335940d1e` |
| B fused eager / QK | `fused_lazy_qk_candidate_a3f5ecc2/build/lib/a5/onboard/fully_distributed_within_core/aicore-extra/pmu_diag_fused_eager_qk_v8/` | `5878a7a3d4fabac1345dadd1f44526109a428d61de4c1240ce3afeed9d225bf0` |
| C fused lazy / STOP | `fused_lazy_qk_candidate_a3f5ecc2/build/lib/a5/onboard/fully_distributed_within_core/aicore-extra/pmu_diag_fused_lazy_stop_v8/` | `1726ddaebbe493050b62f00825255d06f979b15e36e5c384b6230ef6fc2b4928` |
| C fused lazy / EMPTY | `fused_lazy_qk_candidate_a3f5ecc2/build/lib/a5/onboard/fully_distributed_within_core/aicore-extra/pmu_diag_fused_lazy_empty_v8/` | `3b0598b3b6b051645355dbbf5fb67dbdf3d3150080e3467e3608c430131d23a9` |
| C fused lazy / QK | `fused_lazy_qk_candidate_a3f5ecc2/build/lib/a5/onboard/fully_distributed_within_core/aicore-extra/pmu_diag_fused_lazy_qk_v8/` | `5859c93a922a6433c9010609c884f8c17dc2f4f37653dba1c87927dfdc850af9` |

表中目录均以 `/home/q00473782/atomic/private/3-gpt/` 为根，文件名均为
`aicore_kernel.o`。三种模式的 flags 必须分别精确为
`STOP=0x504d00fc`、`EMPTY=0x504d01fc`、`QK=0x504d02fc`；shape 名本身不能
从 JSON 推导，所以采样入口还会把 `(shape, mode)` 与上表 SHA256 绑定。
2026-07-18 归档位于 `pmu_lazy_a5_results_v8/`，其 `manifest.tsv` 虽有 21 行，
但 QK total counter 全部为 0，当前严格解析器会以 `QK_TOTAL_ZERO` 拒绝，不能
据此发布 PMU 动态差分。2026-07-19 的三轮 STOP 泳道复测及自包含原始 JSON
位于：

```text
/home/q00473782/atomic/private/3-gpt/simpler/outputs/
  qk_lazy_abc_perf_20260719/trace_on_diagnostic_stop/
    SUMMARY.md
    manifest.tsv
    raw/{split_eager,fused_eager,fused_lazy}_stop_r{1,2,3}/
```

这些 STOP 数据只支持 §4.7 的同口径性能方向和内容 oracle；有效 EMPTY/QK
仍须在修正 collector 后按下列固定顺序重跑：

采样入口以 device-0 本任务锁避免本矩阵
重入，并拒绝当前机器上已知的 `msprof/xlog1py`、A5 pytest 与批量测试父进程；
它不是驱动级的全系统进程枚举，因此原始样本仍保留逐次门禁和 overlap 记录，
不能把该检查扩大成硬件层绝对独占证明：

```text
round 1: A empty/qk → B empty/qk → C empty/qk
round 2: C qk/empty → B qk/empty → A qk/empty
round 3: B empty/qk → C empty/qk → A empty/qk
```

每个正式 EMPTY/QK 链接均使用 `<shape>_<mode>_r<round>`，并在同目录
`manifest.tsv` 记录 artifact SHA256 与原始输出目录；最终解析还会拒绝两个
标签指向同一 JSON。失败、oracle 不通过或检测到重叠的样本不建立正式链接。

以下采样、严格解析、机器形态审计及诊断补丁曾固化于历史迁移提交
`142cf5ff`；当前 cleanroom 未纳入该目录：

```text
tests/atomic_probe/pa_lazy_qk_pmu/
  README.md
  run_lazy_pmu_sample.sh
  run_lazy_pmu_matrix.sh
  lazy_pmu_analyze.py
  analyze_lazy_pmu_matrix.sh
  audit_machine_shapes.py
  machine_shape_audit.tsv
  pa_diag_pmu.h
  eager_diagnostic_source.patch
  fused_lazy_diagnostic_source.patch
  fused_lazy_diagnostic_harness.patch
```

### 10.3 当前方法入口与历史资料

当前 cleanroom 内可访问的方法入口：

```text
tests/atomic_probe/pa_scheduler/PA调度器独立复现与泳道使用指南.md
tests/atomic_probe/test_case.md
tests/atomic_probe/AICore嵌套Lambda捕获与模板调用验证.md
src/a5/platform/include/common/pmu_profiling.h
```

以下方案原文、历史 PMU 包及 v7 selector 输出没有随附于当前 cleanroom：

```text
docs/make_replay_faster.md
docs/make_replay_faster_a5_performance_analysis.md
tests/atomic_probe/pa_lazy_qk_pmu/README.md
/home/q00473782/atomic/sfd/gpt/pmu_lazy_a5_config_probe_v7
```

本机 CANN I-cache/PMU 资料：

```text
/home/q00473782/cann/cann-9.1.0/x86_64-linux/data/platform_config/Ascend950PR_958b.ini
/home/q00473782/cann/cann-9.1.0/x86_64-linux/simulator/dav_3510/lib/Ascend950pr_9599_model.toml
/home/q00473782/cann/cann-9.1.0/x86_64-linux/simulator/dav_3510/lib/1982_cloud_config.toml
/home/q00473782/cann/cann-9.1.0/x86_64-linux/simulator/dav_3510/lib/cpro_light_config.toml
```

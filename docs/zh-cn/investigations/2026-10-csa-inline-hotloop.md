# CSA无记录调度循环：编译形态实验

日期：2026-10-29。结论：撤回强制内联和显式重申空profiling状态的拓展，
保留已独立评估的模板特化。两个撤回候选均未做真机性能采样，不能称为已测得负收益。
组协议记录体独立化的真机对照另列于末节，不与上述未采样实验混同。

## 假设与改动

原调度循环由有记录／无记录入口共用。先做的模板特化已使无记录循环从约303KB
缩至279–285KB，并取得独立Device对照。该特化不属于本次撤回范围。

进一步假设：只将无记录循环内联到`run_resident_scheduler<false>`，
编译器能利用本地profiling状态为空的事实消去更多可选记录代码。
实验给循环模板增加always_inline，有记录版本经单独noinline包装保留独立调用；
既有跨kernel调用的trace暂存helper仍是noinline。没有删除错误、依赖或三lane检查。

## 编译证据与取舍

A5与A5sim均编译成功。通过ELF符号表获取函数尺寸，不依赖尚不可用的A5指令解码：

| 代码范围 | 仅特化 | 再内联 | 变化 |
| -------- | ------ | ------ | ---- |
| AIC无记录入口＋调度循环 | 29040＋278960＝308000B | 333000B | 增大8.12% |
| AIV无记录入口＋调度循环 | 29024＋284868＝313892B | 332028B | 增大5.78% |
| 有记录调度循环 | 302656／303560B | 302656／303560B | 尺寸不变 |

原两个函数的尺寸相加后，与合并后的入口比较，不能只比较循环符号消失与否。
结果没有形成预期的代码体积收益，故在扩大测试前撤回该拓展。
函数变大不等同于已证实变慢；这里没有宣称运行时、寄存器溢出或指令缓存缺失的实测结论。
也没有将编译成功当作A5正确性通过。

## 归档与重开条件

工作区`reports/csa_poursoul_20261027.Osnaam/`保留：

- `idle-hotloop-production.patch`：独立模板特化候选。
- `idle-hotloop-inline-production.patch`：相对提交58e4a2c6c的强制内联候选完整补丁。
- `runtime-idle-hotloop-inline/`：强制内联原生二进制。
- 主仓`outputs/csa-hotloop-inline-runtime-build.log`：编译日志。

只有出现更具体的常量传播证据，或不依赖代码体积减少的可检验收益假设，才重开。
不要将本次“未采样、按编译结果停止”的取舍误写为一次负性能A/B实验。

## 显式重申空profiling状态：未形成预期收缩

基线为`57f5c4021`，生产实现与冷错误写出检查点相同。
无记录入口本来就构造空profiling状态，但独立的循环函数只将记录等级设为常量0。
候选在`run_ready_dispatch_loop<false>`入口增加一行`context->profiling = nullptr`，
尝试向循环内传播同一个既有事实；不改变L1以上或采样计时的入口选择。

| 无记录循环 | 基线 | 候选 | 缩小 |
| ---------- | ---- | ---- | ---- |
| AIC | 203436B | 203412B | 24B |
| AIV | 206944B | 206848B | 96B |

有记录循环仍为227300／231352B。没有出现预期的大幅代码收缩，故撤回该单行拓展，
未开展设备正确性或性能采样，不宣称运行时负收益，也不从尺寸推断具体残留指令。
入口C++目标在改动前后通过，覆盖三lane×四等级×采样开关；候选21个C++目标通过，
A5与A5sim构建成功。当前生产源码、原生二进制与仿真均恢复基线。

补丁和二进制在工作区归档根目录下的`idle-noprofile-state-production.patch`和
`runtime-idle-noprofile-state/`；主仓`outputs/csa-noprofile-state-{aic,aiv}-symbols.txt`
保存ELF尺寸，`outputs/csa-noprofile-state-runtime-build.log`保存构建日志。
下一次编译形态实验需有更直接的代码分离或常量传播机制，不重复仅重申此空指针事实。

## 组协议记录体独立化：收益证据不足，撤回

基线生产实现不变。候选只改`scheduler_protocol_block.h`中的六类记录函数：
领取、完成、组发布、descriptor、lane发布与ACK观察。函数体原样移至LOCAL绑定的noinline函数，
原入口保留inline开关判断，关闭记录时不进入新函数。没有修改调度协议、原子操作或缓存屏障。
无记录AIC／AIV循环为203436／206944→192880／197668B，约缩小5.19%／4.48%；
尺寸不是I-cache缺失或实际性能收益的直接证据。

新增三种关闭记录情形：空local、空profiling、layout offset为零。用非法block、空descriptor、
空ACK数组调用全部六类入口，比较整个共享状态保持不变。361项Mix用例在基线／候选通过，
候选21个C++目标与静态检查通过；两平台构建成功，原样CSA严格L4和七项golden通过。
这是等价保护，不伪称旧代码业务RED。

每批三轮七项golden、5预热＋100正式样本，54个kernel指纹一致，CLI竞争观察为空：

| 批次 | 完整Device中位数（us） |
| ---- | ---------------------- |
| `idle-full-service-before-c`：最近同二进制基线 | 2000.2325 |
| `idle-protocol-block-outline-after-a`：候选 | 1997.0750 |
| `idle-protocol-block-outline-before-a`：恢复基线 | 2010.6580 |

候选比前后基线分别低3.1575／13.583us，但两基线自身相差10.4255us。
没有扩大到每臂200样本的重复确认，不据此声称稳定Device收益，更不能称为无插桩负收益。
L4任务窗口2461.003→2557.442us、全核空窗798.642→840.052us均增大；
t23同cluster领取周期中位34.480→36.2485us。它们是独立捕获，不精确拆分level0成本。
结合记录路径的退化和有限的无插桩收益证据，撤回生产，保留三项关闭记录保护测试。
没有运行候选的全HBG设备回归或完整故障矩阵，当前生产源码、原生二进制及仿真恢复基线。

原始数据与验证限制见
[组记录独立化摘要](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-10-28-idle/protocol-block-outline-summary.json)。
工作区归档保留`idle-protocol-block-outline-{production,tests}.patch`、
`runtime-idle-protocol-block-outline/`与`artifacts/idle-protocol-block-outline-after-l4/`。

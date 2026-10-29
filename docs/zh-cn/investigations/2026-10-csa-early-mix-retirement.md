# CSA Mix提前服务：退休顺序与连续组周转

日期：2026-10-28；补充实验：2026-10-29。仅提前退休的方案已撤回。
保持连续领取的完整服务重开实验见文末，不将局部阶段改善当作完整Device收益。

## 问题与候选

8B close-seal、v5协议的CSA基线L4中，26个Mix任务最后贡献组的全三lane release
到C确认退休，中位间隔为6.267us。fill在确认当前已RELEASED的组之前，先发现新候选。
这会推迟TaskDone及依赖后继，值得针对全核计算空窗验证。

候选在已有pending发布之后、discovery之前，先复用frontier和ordinary槽检查，
再调用原退休与TASK_LAST完成路径；随后照常discovery。提前路径不close、claim或发布下一组。
同轮后段用局部布尔避免重复退休；三lane强ACK、epoch/error、唯一完成和两份candidate credit均保留。
无ABI或原生协议变化，不修改54个业务kernel、2308个执行身份或七项golden。

## 验证和结果

新增24项参数化测试，在旧顺序上16项按预期失败、8项保护通过。候选全部通过，
六个关联C++目标通过。四项旧noREADY时序断言曾调整为READY必须正常准入且身份正确，
原exactly-once验证未删除。严格原样CSA L4通过，保留133条依赖和46733个协议事件。
未将此范围冒充完整设备故障矩阵或全部HBG场景验收。

同卡按A/B/A串行采样，每批3轮×7项golden、5预热+100正式样本：

| 批次 | 完整Device中位数（us） |
| ---- | ---------------------- |
| `idle-prefix8-before-retire-a`：基线 | 2250.4470 |
| `idle-retire-first-after-a`：提前退休 | 2380.2695 |
| `idle-retire-first-before-a`：恢复基线 | 2251.3110 |

候选比恢复基线慢约5.73%。两个基线接近，结合下述阶段证据，停止扩样并撤回。
不因候选正确性通过或局部Resolve提前而保留负收益改动。

同布局L4对照：

| 指标 | 基线 | 候选 |
| ---- | ---- | ---- |
| 最后贡献组全release→C确认，中位 | 6.267us | 4.435us |
| 全task末kernel→DONE，中位 | 12.512us | 10.4325us |
| 同task退休→下一次claim，中位 | 1.657us | 5.433us |
| publish→全三lane arrive，中位 | 5.407us | 6.553us |
| C/V入口偏斜，中位 | 2.886us | 4.541us |
| t23同cluster相邻claim周期，中位 | 40.534us | 45.477us |
| 全局无kernel间隙 | 921.875us | 1011.268us |
| 其中同task间隙 | 104.706us | 175.637us |
| 首kernel到末kernel窗口 | 2711.793us | 2866.566us |

退休段确实缩短，但discovery被插到退休与同task下一次claim/publish之间，
使连续多block任务新增组周转停顿。同task间隙增加70.931us，占全局空窗增量的大部分。
这是阶段证据支持的解释，不把插桩差值直接当作无插桩Device的精确成本拆分。

## 撤回与重开条件

生产与测试补丁保存在工作区 `reports/csa_poursoul_20261027.Osnaam/` 的
`idle-retire-first-production.patch`、`idle-retire-first-tests.patch`。
主仓恢复原退休/发现顺序及旧策略断言，保留独立的Mix前缀早发和错误门禁测试。
原始结果位于该目录的 `artifacts/`；L4分别是
`idle-prefix8-before-retire-l4`、`idle-retire-first-after-l4`。
候选原生二进制保存在 `runtime-idle-retire-first/`，不覆盖基线归档。

只有能保持连续组的retire→next-claim/publish邻接，或有证据证明不再影响后续组周转时，
才重开这一排序方案。后续独立的空Ready结果复用已保留，见
[计算空窗优化记录](../../../tests/fdwic/pa_hbg/spmd/csa-idle-optimization-20261028.md)。
它不额外加一次判空，也不缓存跨轮许可，不能直接沿用本次局部收益。

## 保持连续领取的完整服务：双调用版本撤回

基线为`539e58b6d`。候选对已RELEASED的C先执行完整组服务，
保留retire→next-claim/publish邻接，再执行discovery；后段仍执行原组服务。
TaskLast完成产生Ready时清除本轮负探测提示，使依赖后继能在同一fill被发现。
提前关闭计入原两次义务预算，不增加候选数、槽数或在飞资源组。

新增七类×四个Mix mask共28项用例。修正协议记录下标后，在基线上16失败、12通过；
另五项既有测试验证非法配置拒绝及纯C后继必须正常准入，均在基线上通过。
候选386项Mix测试、21个C++目标及3527项Python测试通过。
非法scheduler计数／索引、队列owner边界、核类型必须沿原路径拒绝，不能先推进当前组。
旧的“后继这一轮不能READY”策略断言改为READY时必须已准入、身份／形状／payload正确；
原逐block唯一执行与完整结束检查保留。未运行本候选的全HBG设备回归或完整故障矩阵。

同卡A/B/A，每批三轮七项golden、5预热＋100正式样本，54个kernel指纹一致：

| 批次 | 完整Device中位数（us） |
| ---- | ---------------------- |
| `idle-full-service-before-a` | 2012.044 |
| `idle-full-service-after-a` | 2040.863 |
| `idle-full-service-before-b` | 2001.480 |

候选比恢复基线慢约1.97%，不保留该双调用版本。
严格L4通过，但相对`idle-cold-error-after-l4`，全局空窗798.642→780.055us的同时，
窗口2461.003→2485.944us、同task空窗79.454→95.845us、
C/V入口偏斜中位0.868→1.440us、t23领取周期34.480→35.7005us均变差。
这些是独立插桩捕获，不据此精确分摊无插桩成本。

代码审查发现空Ready路径仍可能在同一fill服务当前组两次；
AIC无记录热循环从203436B增至253252B。代码体积不是I-cache缺失的直接测量，
上述两个现象只作为下一实验依据，不将它们直接等同于全部退化原因。
补丁为工作区归档根目录下的`idle-full-service-{production,tests}.patch`，
二进制为`runtime-idle-full-service/`，L4为`artifacts/idle-full-service-after-l4/`。

单循环版本使用同一个组服务调用点，发现新任务至多一次，关闭义务仍至多两次；
已完整服务且无须discovery时直接返回，不再次执行同组服务。
21个C++目标、静态检查、原样CSA严格L4与七项golden通过；
无记录AIC循环降回207592B，但仍比203436B基线稍大。

使用上一批恢复基线作为前测，独立A/C/A对照如下，每批采样规则相同：

| 批次 | 完整Device中位数（us） |
| ---- | ---------------------- |
| `idle-full-service-before-b`：基线 | 2001.4800 |
| `idle-full-service-single-after-a`：单循环 | 2021.0440 |
| `idle-full-service-before-c`：恢复基线 | 2000.2325 |

候选仍比恢复基线慢约1.04%，不保留。严格L4全核空窗为777.012us，
窗口2455.317us；t23领取周期34.5075us、同task空窗95.836us。
这再次表明少量全核空窗下降不足以证明完整Device改善，不能只选正向指标。
单循环补丁为`idle-full-service-single-production.patch`，二进制为
`runtime-idle-full-service-single/`，L4为`artifacts/idle-full-service-single-after-l4/`。

两种候选的生产改动与28项策略测试均撤回，四项旧后继时序断言恢复。
独立的非法配置测试扩为六种输入，保留原拒绝／不贡献完成／不领取／不发布断言；
它在基线也通过，不把该扩展描述为已修复生产缺陷。
当前生产实现和原生二进制恢复至`539e58b6d`；1.5ms目标仍未达成。
完整批次身份、分位数、原始结果校验和、L4阶段及验证限制见
[完整服务实验摘要](../../../tests/fdwic/pa_hbg/spmd/perf_data/2026-10-28-idle/full-service-summary.json)。

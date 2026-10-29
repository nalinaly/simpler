# 普通多block重叠的确定性见证

日期：2026-10-29。范围：普通纯C／V及GM Mix测试，不修改生产调度规则或CSA业务kernel。

## 原问题

原`overlap`用例让尾块执行8000000次有限延时，再要求两任务的entry／return区间重叠。
有限延时只能扩大机会，不能保证另一任务已被调度。此前仿真整批中出现过两任务恰好串行的记录，
而同二进制单独及原顺序复跑通过；单次未重叠不能单独证明存在全任务完成屏障。

修复不删除重叠断言，也不增加延时。验收应同时证明“发生了重叠”和“强制见证路径确实执行”。

## 复用已有peer握手

普通ordered／GM Mix用例复用已有`sync_ordered_orchestration.cpp`及`sync_ordered_slot_*.cpp`，
传入全零`sync_flags`；名称复用不代表开启SYNC_START，普通多block仍按分波方式执行。
只有`scenario=overlap`传入`wait_for_peer=1`，其他场景不等待peer。
该编排原本支持任意非等待任务数，只有双向等待才限定为两个任务，故无需新增编排或kernel。

两任务互为peer，仍允许A或B先准入。每个活动lane只在指定尾块调用已有等待分支，
直到看到对方至少一个kernel已经进入；其他执行位置可完成本地义务并进入独立peer。
没有增加全核同时启动、整任务屏障或Scalar动态内存分配。

| guard列 | 验收条件 | 作用 |
| ------- | -------- | ---- |
| 4 | `N × active_lane_count` | 每个实际kernel都发布过peer可见的entry |
| 5 | 0 | 没有等待超时后伪装成功 |
| 6 | `active_lane_count` | 每个活动lane恰好走过一次尾块等待分支 |

五秒预算沿用既有kernel，用于有界失败退出，不作为成功证据。
若错误地等待整个A完成才让B进入，A尾块会超时，guard和原重叠断言都会拒绝验收。
执行次数、原N、block／slot身份、依赖、输出可见性及每轮fresh resident证明继续逐项检查。
原sync_ordered及ready_order用例的专用检查保留，未用普通场景覆盖它们。

## TDD及设备验证

- 新增48项对抗检查：保持事件区间已重叠，却去掉任意一方的peer entry或尾块等待计数。
  旧验证器全部错误接受，因此48项按预期失败；这不是缺少API或编译失败形成的RED。
- 新验证器全部拒绝上述伪见证，四个相关Python文件共1037项通过。
- 对原44个ordered及68个GM Mix场景逐一检查全零sync_flags和peer开关，防止普通请求
  被偷偷转换成SYNC_START，也保证非overlap场景不引入互等。
- A5与A5sim的ordered、GM Mix、sync_ordered、ready_order四个scene class全部匹配用例通过。
  两平台并行回归时仍通过；pytest显示每平台4通过，不将其误写为只有4个形状。

原失败日志保留在工作区`outputs/csa-hotloop-sim-regression.log`。
新RED／GREEN日志为`outputs/csa-overlap-witness-red.log`和`csa-overlap-witness-green-final.log`；
设备日志为`outputs/csa-peer-witness-a5.log`、`csa-peer-witness-sim.log`。
这是一项测试确定性修复，不计入CSA性能收益，也不代替最终完整故障矩阵验收。

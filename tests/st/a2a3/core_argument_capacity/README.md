# CORE 大参数 MIX 验证

本用例验证 Simpler 的 CORE 参数容量从 32 扩到 256，保留 16 个标量。
目标是整层融合后的大 InCore 任务；本用例不代表完整 CSA 编译或 CSA 性能验证。

## 覆盖范围

- 133 个 Tensor：覆盖当前 CSA 原型最大的共享签名。
- 256 个 Tensor + 16 个标量：覆盖传输容量上界，最后一个标量位于槽 271。
- 每项任务为 24 个 MIX 核组，每组 AIC + AIV0 + AIV1，检查逻辑核编号和 lane 身份。
- 每次入口连续提交两项任务，第二项依赖第一项，并启用 early resolve 与 sync_start。
- 每个核读取所有输入和全部标量，写入独立 cache line；完整比较输出和未写区域。
- program 模式与调用方持有设备 Tensor/stream 的 kernel 模式分别运行。
- 选择性 dump 验证第 63/64/128、最后一个 Tensor、最后一个标量；无截断或丢记录。

## 复现

先在独立 worktree 的 `.venv` 中安装本版本，按仓库环境文档加载 CANN。
所有真机命令通过 `task-submit`，核型号预检使用：

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3
python -m simpler_setup.tools.scene_test_compile tests/st/a2a3/core_argument_capacity \
    --platform a2a3 --compile-workers 4
```

选择一个 runtime，每次进程只加载一种 runtime；例如 program 模式：

```bash
task-submit --device auto --device-num 1 --timeout 600 --max-time 600 \
    --run 'python -m pytest tests/st/a2a3/core_argument_capacity --platform a2a3 \
    --runtime host_build_graph --level 2 --device "$TASK_DEVICE" --rounds 1 --dump-args 1 -s -q'
```

将 runtime 改为 `tensormap_and_ringbuffer` 可验证 TMR。多轮正确性用 `--rounds 3`；
框架会在多轮运行时禁用 dump，因此采集 dump 必须用 `--rounds 1`。
用 `verify_dump.py <args_dump.json> --count 133` 或 `--count 256` 验证实际生成的文件。

kernel 模式沿用仓库已有 Worker/ACL 测试工具，独立执行：

```bash
task-submit --device auto --device-num 1 --timeout 600 --max-time 600 \
    --run 'python -m tests.st.a2a3.core_argument_capacity.kernel_mode \
    --device "$TASK_DEVICE" --runtime host_build_graph --count 133 \
    --output outputs/core_capacity/kernel_hbg_133.json'
```

同样可切换 runtime 和参数数量。每个进程准备一次 callable 并执行三次 launch，
结果 JSON 只在输出和保护区全部比较通过后标记 `passed: true`。

## ABI 与构建要求

- `CoreCallable` 固定头部 1040 B，核二进制偏移 1088 B。
- CORE 参数区 272 个槽，上下文位于 272/273；派发结构 2304 B，保留首行控制区。
- dump 掩码为 5 个 64 位字，覆盖实际 272 位；记录结构保持 POD。
- HBG TaskPayload 为 256 B，ChipTaskStorage 为 384 B；Graph 边界 Tensor 容量为 256。
- TMR 的 Tensor 数组为 256 × 128 B，标量偏移从 Tensor 容量计算。
- Host、AICPU、AICore、Python binding 和 callable 必须成套重编译。旧二进制不兼容。
- callable 缓存已有序列化 ABI token，InCore 缓存已有头文件内容依赖；本次另补齐
  runtime 自定义 CCE 编译的 ABI 头文件依赖，防止增量构建复用旧派发偏移。

扩容会增加固定参数池的预留空间；HBG 的设备 live image 仍按实际用量压紧。
本用例未测试性能，也不改变 PyPTO/PTOAS/PTO-ISA/pypto-lib 源码。

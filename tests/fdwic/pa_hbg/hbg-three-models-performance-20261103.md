# A5 HBG：Qwen3-14B、V4 Pro CSA、DS4.1 SWA 的 AICore / AICPU 对照

日期：2026-11-03。三个用例全部使用 `host_build_graph`，每个后端各执行 **10 次**，0 次额外预热，保留首轮及全部样本；本文件的数据来自同版本、同卡的新一轮实测。

## 1. 性能与校验结果

时间单位 ms；主指标为完整 Device RunWall 的中位数。变化率为 `(AICore / AICPU - 1) × 100%`，正值表示 AICore 更慢。Host 与 Device 是各自窗口，不能相加，也不能将 Host 减 Device 当成纯调度耗时。

| 用例 | HBG/AICore Device | HBG/AICPU Device | AICore 时间变化 | HBG/AICore Host | HBG/AICPU Host | 数值校验 |
| ---- | ----------------: | ---------------: | --------------: | --------------: | -------------: | -------- |
| Qwen3-14B decoder | 50.0561795 | 29.361833 | +70.48% | 56.260045 | 30.858270 | 两端通过 |
| DeepSeek V4 Pro CSA | 2.4709005 | 2.093926 | +18.00% | 30.136610 | 27.935610 | 两端通过 |
| DeepSeek V4.1 Flash SWA | 0.872835 | 0.728078 | +19.88% | 9.303960 | 14.622550 | 两端通过 |

Device 包含设备启动、Graph 物化（存在时）、调度、kernel 执行及退出。Host 使用 `chip.run`；kernel 编译、CPU golden 生成与本次独立比较不计入这个窗口。Qwen 的一次性输入上传在计时循环外；CSA/SWA 采用原 Host Tensor 参数路径，每次调用包含相应输入搬运与输出回传，因此三个模型的 Host 数字不作为横向效率排名。

### 完整 Device 的全部 10 个样本

| 次数 | Qwen AICore | Qwen AICPU | CSA AICore | CSA AICPU | SWA AICore | SWA AICPU |
| ---: | ----------: | ---------: | ---------: | --------: | ---------: | --------: |
| 1 | 49.981256 | 29.598006 | 2.566867 | 2.403553 | 0.853153 | 1.059749 |
| 2 | 49.261243 | 29.333638 | 2.506157 | 2.085378 | 0.869257 | 0.728423 |
| 3 | 49.721273 | 29.376070 | 2.448591 | 2.090516 | 0.894816 | 0.753901 |
| 4 | 50.348681 | 29.305856 | 2.441285 | 2.122481 | 0.876967 | 0.727733 |
| 5 | 50.131103 | 29.347596 | 2.476866 | 2.060339 | 0.869182 | 0.750406 |
| 6 | 51.156481 | 29.450725 | 2.459950 | 1.971606 | 0.868111 | 0.691923 |
| 7 | 49.664396 | 29.523599 | 2.460823 | 2.105428 | 0.900396 | 0.784167 |
| 8 | 50.293692 | 29.238679 | 2.464935 | 2.095844 | 0.876413 | 0.719090 |
| 9 | 50.875411 | 29.338388 | 2.484318 | 2.093507 | 0.902103 | 0.715552 |
| 10 | 49.842083 | 29.388965 | 2.497618 | 2.094345 | 0.867617 | 0.724545 |

### Host 绑定成本与波动

| 用例 / 后端 | Host bind 中位数 / ms | Device 最小 / 最大 / ms | Host 首轮 / ms |
| ----------- | --------------------: | ----------------------: | -------------: |
| Qwen3-14B decoder / aicore | 1.625705 | 49.261243 / 51.156481 | 112.754590 |
| Qwen3-14B decoder / aicpu | 0.247910 | 29.238679 / 29.598006 | 79.605950 |
| DeepSeek V4 Pro CSA / aicore | 23.389485 | 2.441285 / 2.566867 | 56.154240 |
| DeepSeek V4 Pro CSA / aicpu | 22.589950 | 1.971606 / 2.403553 | 58.921450 |
| DeepSeek V4.1 Flash SWA / aicore | 7.270690 | 0.853153 / 0.902103 | 37.135800 |
| DeepSeek V4.1 Flash SWA / aicpu | 11.557125 | 0.691923 / 1.059749 | 45.378020 |

只运行一轮每端 10 次的顺序对照，没有交错重复或置信区间。设备队列锁不代表整机 CPU、内存和其他 NPU 负载独占；Host 首轮开销与共享主机波动均原样保留。本轮没有将差值进一步归因到某条原子指令或某个调度阶段。

## 2. 源码、环境与后端控制

- Simpler：`feat/fully_distributed_within_core` 的固定提交 `0f4b1e73d2b24240e49ea6a51869f6b33528ed2f`；在 detached worktree 中构建独立 venv 和 runtime。该提交包含 Graph 并行物化，并已回退 ordinary pool。
- 设备：A5 `Ascend950DT_9581`，32 AIC / 64 AIV；六组都使用设备 **1**，同一个 `task-submit` 任务持锁串行执行。任务 `task_20261103_105055_72596118523` 已 exit 0，锁已释放。
- 工具链：CANN 9.3.0、CCEC 15.0.5、GCC 12.3.1、Python 3.12.15、Torch 2.12.0+cpu；Linux aarch64。
- PTO-ISA：`217bb98952a8f9004b95fe938da97d8d5f68f702`；SWA 生成时 PTOAS 为 v0.70。本轮直接加载冻结 ChipCallable，没有重新生成计算 kernel。
- CPU：构建最多 2 worker；测量进程绑定 CPU 0、1，Torch intra-op 2 / inter-op 1。启用 `PYTHONNOUSERSITE=1`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，避免系统 Torch NPU 插件干扰 CPU golden。
- 所有调用关闭完整泳道和 PMU：`chip_swimlane_level=0`。启用既有 TIMING/STRACE 日志，解析 `chip.run.runner_run.device_wall`、`chip.run`、`chip.run.bind`；检查记录完整且无丢弃。

后端选择不是用户配置开关。本轮在同一提交构建 AICore 原版，再仅在隔离副本的 `create_scheduler_state()` 调用 `release_scheduler_state()` 后插入：

```cpp
select_legacy_scheduler(runtime, api, SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE);
return true;
```

这强制选择已有 HBG legacy AICPU 路径，记录 mode=4；该数字表示实验选路，不表示这三个负载真的存在不支持的形状。AICore 三组实际均为 mode=1，双方每次调用都要求调度模式发布计数递增。没有切换到 TMR，也没有把未验证的回退算作 AICore 成功。实验补丁只保存在归档，生产源码不含该强制返回。

两套 runtime 的 `libaicpu_kernel.so`、`aicore_kernel.o` 和 AICPU dispatcher 逐字节相同；Host `libhost_runtime.so` 不同。每个模型两端使用同一个 ChipCallable 文件，实际上传的 kernel 长度和 FNV1a64 指纹也逐项一致（Qwen 37、CSA 54、SWA 28 个）。

| runtime 文件 | AICore SHA256 | AICPU SHA256 |
| ------------ | ------------- | ------------ |
| `libhost_runtime.so` | `5e3c86b5d730346cbf301bef05e477a54389c20f242583c0534228937806256c` | `121398266bc0fd5b7559080348d2ddc873d0a72bbdd2fd2ed7f4548fc571978e` |
| `libaicpu_kernel.so` | `e372d313fc90d62f5338d8dee4ee23b52c62429c121b2358c2465aec33ef41ed` | `e372d313fc90d62f5338d8dee4ee23b52c62429c121b2358c2465aec33ef41ed` |
| `aicore_kernel.o` | `0cd389564d1f610c69842d2520552823f3e75ee2c87371d7d93e2e64574ad6de` | `0cd389564d1f610c69842d2520552823f3e75ee2c87371d7d93e2e64574ad6de` |

| 冻结 ChipCallable | 两个后端共同使用的 SHA256 |
| ----------------- | ------------------------- |
| Qwen3-14B | `39949fcf6aa82728312959e92640f106ab68cdbac92935fd79c838464cab7f58` |
| V4 Pro CSA | `8fea231c62f22fb1ae5fc18c8d0139daab8cb33466418505abd606c0dbbcbadb` |
| DS4.1 SWA | `d3234284f28000a1ea9c3eb8e7c96775b2201b8168e8e7ac6637b236aba3a9bd` |

## 3. 三种负载及数值检查

### Qwen3-14B：Simpler 独立 HBG 实现

- 入口：[HBG main.py](../../../examples/a5/host_build_graph/qwen3_14b_decode/main.py)。40 层 decoder、B=16、seq_len=3500、max_seq=5500、seed=1234。一份单层 Graph 定义被 40 个实例复用。
- HBG 编排复用 TMR 目录中的 37 个计算 kernel 和 fixture/golden；该源码组织不代表本轮执行了 TMR runtime。冻结 kernel 包含已修复的 RoPE inverse-RMS 缓存刷新，两端同样使用 `-fno-jump-tables` 编译得到的计算镜像。
- 使用原合成 fixture，各层权重由 layer 0 复制；不是加载真实训练 checkpoint。一次上传到持久设备缓冲区，10 次固定输入重放；KV 写入幂等，不连续推进 token。
- 每端 10 次结束后，比较完整输出 `out` 和全部 40 层 `k_cache`、`v_cache`，保持 rtol=0.05、atol=0.1。不是每轮分别执行 golden。
- 测量不含 LM head、采样或整模型连续生成吞吐。

### DeepSeek V4 Pro CSA：Simpler 冻结 HBG 实现

- 入口：[CSA 独立用例](../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/test_deepseek_v4_pro_csa.py)；B=4、S=2、seed=0，`position_ids=[8192,8193,0,1,2,3,3,4]`。
- 来源：PyPTO-lib `d1de019cb5e84bcb26b28d9c53756c8ac0fb78e8` 的 `models/deepseek_v4_pro/decode_attention_csa.py`，原生成版本 PyPTO `a0ff351c519a580e9523c9ba889b662459680f3f`。本轮加载已经在 CANN 9.3 环境冻结的 54 个 kernel。
- 52 个入口参数；冻结合同有 79 个计算任务、2 个 Dummy、2308 次物理执行。58 个数据文件由原 manifest 校验；manifest SHA256 为 `d1511befaae48a3b692a723320a076064e2ebb1bced193175866c692f9909b45`。
- 每次恢复相同 INOUT 初值。每次运行后均校验 `x_out`、`kv_cache`、`cmp_kv`、`compress_state`、`inner_compress_state`、`idx_kv_cache`、`idx_kv_scale`，沿用原比较器和容差。
- `hc_attn_scale` 保持 Host-only 的 `TensorTransfer.NONE`。本轮不是完整 decode_layer 或连续自回归测量。

### DeepSeek V4.1 Flash SWA：PyPTO-lib 生成的 HBG 实现

- 模型入口：相邻 `pypto-lib/models/deepseek_v4_1_flash/decode_swa.py`；PyPTO-lib `26ef3e865733747a89c3356ec145e567851ad985`、PyPTO `be23606f791fa2251d7fccabdb91c75582268fad`、PTOAS v0.70。
- 72 tokens、12 requests、seed=17、case=mixed；每个 request 的 query length 为 6、start position 为 8192。冻结 HBG 生成目录为 `_jit_l3_decode_swa_20261101_213409_swn1xrk6`。
- 28 个注册 kernel；原负载合同为 24 个计算任务、517 个 block、549 次物理执行。复用原 `data/in`、`data/out`、`distributed_meta.json` 和 golden 比较器，并逐文件验证已有 SHA256。
- 每次恢复相同 INOUT 初值，运行后校验 `window_cache`、`window_cache_scale`、`attention_output`、`attention_hidden`、`attention_pre_mix` 五项。
- 保留原比较器：hidden/output 的 atol=1e-2、rtol=1/128；pre_mix 的 atol/rtol 均为 1e-4；cache 的 atol=2^-9、rtol=2^-3、max_error_ratio=0.01，并检查页映射；cache scale 精确比较。
- 本轮两个后端都加载这同一份 HBG 编排；此前的 TMR 对照数据没有进入本文件性能表。SWA 组件通过不等于完整 DS4.1 decode_layer 已通过。

## 4. 执行与复核方法

运行顺序为 CSA AICore → CSA AICPU → SWA AICore → SWA AICPU → Qwen AICore → Qwen AICPU。每组在独立进程中初始化、注册 callable，执行 10 次后关闭 Worker；Qwen 在最后比较 golden，CSA/SWA 每次比较。CPU 比较在调用结束后执行，不计入 STRACE 的 `chip.run`。

本次的命令入口如下（从 Simpler 仓库根目录执行；实验目录内已有冻结包及独立环境）：

```bash
BENCH="$(cd ../logs/hbg-three-models-20261103-103229 && pwd)"
source "$BENCH/bench_env.sh"
.claude/skills/onboard-arch-precheck/check.sh a5
# run.sh 内固定每后端 10 次；首次执行创建 results，拒绝覆盖已有结果。
task-submit --device auto --device-num 1 --timeout 7200 --max-time 1800 --run "bash $BENCH/run.sh"
python "$BENCH/summarize.py"
```

该归档已完成，不能直接覆盖重跑。复跑应复制实验驱动到新结果目录并重新冻结目标提交，或给 `measure_attention.py` 使用新的 tag、给 `qwen_bench.py --output` 指定新的文件。所有重新计时仍设置 `--rounds 10 --warmup 0`（attention 驱动循环固定 10 次）。不要调用历史 `CSA run.py bench` 或旧 SWA measure 包装器：它们默认包含 100 次计时。

构建方式及复现边界：

1. 固定上述提交，建立独立 worktree 和 `--system-site-packages` venv；设置 `CMAKE_BUILD_PARALLEL_LEVEL=2`，安装当前 `_task_interface`，要求其 build commit 与源码一致。
2. `build_variants.py` 先构建并冻结原版 HBG 三件套，再应用本节所述测试选路补丁，重新构建并冻结 AICPU 版本，最后恢复隔离副本的源码。比对设备端二进制与计算 callable 摘要。
3. `prepare_inputs.py` 验证冻结 Qwen 来源、binding 和 callable；CSA 导入原冻结数据并由 fixture 校验；SWA 校验元数据、输入、golden 及比较器摘要。
4. 上板前运行 A5 架构检查；全部六组通过同一 `task-submit` 分配执行，日志单独设置 `ASCEND_PROCESS_LOG_PATH`。不与另一后端同时计时。
5. `summarize.py` 要求每组恰好 10 个完整计时窗口、新的后端模式发布、golden 通过、两端 kernel 指纹相同，并从全部样本计算中位数/均值/范围。

## 5. 原始证据与可携带范围

- [汇总及全部 Host/Device/bind 样本](../../../../logs/hbg-three-models-20261103-103229/summary.json)、[60 次调用 CSV](../../../../logs/hbg-three-models-20261103-103229/samples.csv)、[环境](../../../../logs/hbg-three-models-20261103-103229/environment.json)。
- [测试计划与版本](../../../../logs/hbg-three-models-20261103-103229/plan-and-identity.json)、[构建脚本](../../../../logs/hbg-three-models-20261103-103229/build_variants.py)、[六组执行脚本](../../../../logs/hbg-three-models-20261103-103229/run.sh)、[统计脚本](../../../../logs/hbg-three-models-20261103-103229/summarize.py)。
- [Qwen 测量驱动](../../../../logs/hbg-three-models-20261103-103229/qwen_bench.py)、[CSA/SWA 测量驱动](../../../../logs/hbg-three-models-20261103-103229/measure_attention.py)。
- [AICore 冻结清单](../../../../logs/hbg-three-models-20261103-103229/bundles/aicore/manifest.json)、[AICPU 冻结清单](../../../../logs/hbg-three-models-20261103-103229/bundles/aicpu/manifest.json)、[AICPU 选路补丁](../../../../logs/hbg-three-models-20261103-103229/bundles/aicpu/source.patch)。
- 原始日志：归档 `results/`（六组 stdout/运行日志）、`measurements/current-*/host/`（CSA/SWA STRACE）、各组 `ascend/` 和 `queue-run.log`。

上述归档链接指向当前工作区的相邻 logs，不随本 Markdown 自动进入 Git；在其他机器复跑需同时提供冻结 callable、runtime 包与输入/golden。本文保留全部 Device 样本和 runtime 摘要，以便在没有大文件的代码审查中核对统计。

本轮执行的是三种固定负载的双后端数值与性能检查，没有重跑全仓库回归，也没有采集新的 L1–L4 协议审计。功能结果、调度模式和 kernel 同一性已核对；不把这些检查替代完整协议验证。

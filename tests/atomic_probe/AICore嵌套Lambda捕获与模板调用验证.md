# AICore 嵌套 Lambda、caller 栈地址与 inline 调用边界验证

> 历史设备结果日期：2026-07-17
>
> 当前文档复核日期：2026-07-19
>
> 持久 fixture 源码首个完整提交：
> `a3f5ecc2186fb8a06cded6d26886a4ab802009c0`
>
> 测试目录：`tests/atomic_probe`
>
> 验证平台：A5 device 0、CANN 9.1.0

> cleanroom 迁移状态（2026-07-19）：本文由
> `wip/qk-lazy-pmu-migration-20260718@cb799c4d` 迁入当前
> `origin/fdwic-swimlane-deps@6caa269c`。当前分支保留旧 caller-capture
> fixture，但没有随附 `d11277bd` 新增的 `nested_lambda_claim_first_once`、
> `block_local_cross_tu` 源码、构建产物和设备日志；相关命令在重建前不能从
> 当前 HEAD 直接复跑。下文对应的 A5 PASS 是绑定当时源码/ELF SHA 的历史
> 结果，不表示 cleanroom 已重新执行。

> 文档定位：供后续 lazy-lambda/codegen submit 实现复用的 CCEC 调用边界、
> 生命周期和验收约束。PA 的 A/B/C/D 方案演进、PMU 归因与性能结论记录在
> `tests/atomic_probe/pa_scheduler/make_replay_faster_qk_single_callback_a5_analysis.md`，
> 不在本文重复性能数字。

> 2026-07-18 历史增量状态：`nested_lambda_claim_first_once` 与
> `block_local_cross_tu` 已完成本机 CCEC 编译、链接、ELF 符号检查及真实 A5
> oracle；前者 1/1 PASS，后者 AIC 5/5、AIV 1/1 PASS。源码提交与结果边界见
> 第 7 节；原始构建产物和逐轮日志未纳入当前 cleanroom，不能把历史 PASS 写成
> 当前 HEAD 已复跑。
>
> 增量构建所用 CCEC：
> `/home/q00473782/cann/cann-9.1.0/bin/ccec`，build
> `2026-06-10T11:29:46+08:00`，SHA256
> `6cfffcd312cafd676d4576c5aeb69438d78c84575b548a86988152d95ada0efc`；
> 解析后实体为
> `/home/q00473782/cann/cann-9.1.0/tools/bisheng_compiler/bin/bisheng`。
> 链接器 `/home/q00473782/cann/cann-9.1.0/bin/ld.lld` 的 SHA256 为
> `f20ccff7dd936a4da44bb03603f0524864ace2947a10fc1125085e8a2c8799bd`。

## 1. 文档目的和最终结论

本文验证或约束七件事：

1. CPU、AscendC 和纯 CCEC 是否支持嵌套 lambda、捕获和模板调用；
2. caller 栈上的 `Tensor` 地址经过未内联 submit 调用后，是否会触发
   AICore 异常；
3. 异常是否由“链接两个 `.o`”或 `ld.lld` 跨对象重定位直接导致；
4. runtime submit 全 inline，以及把 capture 写入既有 `L0TaskArgs`，
   是否可作为当前工具链的规避形态；
5. claim-first lazy submit 能否保持一次 callback，并且只把已物化
   `L0TaskArgs` 与 POD winner 状态交给一次跨 TU finish；
6. split runtime 若要把 drain/claim 留在 inline front，external
   `[[block_local]]` 状态需要满足哪些定义、分核型和同源约束。
7. 如何从 final HiIPU ELF 中提取实际运行的 entry 字节，并用 CANN 随包
   decoder 得到可校验的真实机器码反汇编。

历史受影响 CCEC 上得到的核心矩阵如下：

| AIC 构建形态 | submit 形态 | 语义变体 | 本轮结果 |
| ------------ | ----------- | -------- | -------- |
| 原始双 `.o` | 外部 submit | m0 | 5/5 507015 |
| 双 `.o`，第二个 `.text=0` | 全 inline | m0 | 5/5 PASS |
| 单 `.o` | 全 inline | m0 | 5/5 PASS |
| 单 `.o` | 仅 weak-context submit noinline | m0 | 5/5 507015 |
| 同一 noinline fixture | 仅 weak-context submit noinline | m1 | 3/3 PASS |
| 同一 noinline fixture | strong 路径保持 inline | strong | 3/3 PASS |

因此可以排除“只要链接两个 `.o` 就会失败”。单 `.o` 保留
`nested_probe_submit_weak_context` 未内联时仍然失败，而双 `.o`
在第二个对象没有代码、submit 已完全内联时通过。

在历史 `2026-07-07` CCEC 和当时代码布局上，已观察到的最小
fixture 触发边界是：

```text
weak-context-materialize-0
+ nested_probe_submit_weak_context 未内联
+ caller 栈地址跨该调用边界
-> 507015 AICore exception
```

这里的 `materialize-0` 只表示**额外写入 `L0TaskArgs` 的 caller 地址数量为
0**；三个 `Tensor *` 仍会写入独立 `CallerContext`，不能把 m0 理解成“完全
没有地址物化”。

全 inline 后 m0 通过；同一 noinline 产物增加一次无业务地址物化后也通过。
这仍然是对 HiIPU 最终 codegen 形态的收敛，不是对某个具体后端 pass 或
某条机器指令的根因定位。

需要严格保留以下结论边界：

- 不是 C++ lambda、捕获或模板语言语义不受支持；
- 没有证据表明 `ld.lld` 链接两个对象本身有错误；
- 不能把 weak、noinline 或“额外 Arg 地址物化数为 0”任一单因素写成充分根因；
- all-inline 是已验证规避，不是 CCEC 编译器修复；
- 当前 `args-runtime-read` 证明的是：在 inline 消费前提下，去掉独立
  caller context、改从既有 `L0TaskArgs` 读取的形态可行；它没有独立证明
  split/noinline runtime 直接读取这些 slot 也可行。

## 2. 最短复现路径

以下命令从仓库根目录执行。

### 2.1 环境

```bash
export REPO=/path/to/simpler
export CANN_ROOT=/path/to/cann-9.1.0

cd "$REPO"
source "$CANN_ROOT/set_env.sh"
export PTO_ISA_ROOT="$CANN_ROOT/x86_64-linux"
export ATOMIC_PROBE_DEVICE=0
```

确认工具和源码存在：

```bash
test -x "$ASCEND_HOME_PATH/bin/ccec"
test -x "$ASCEND_HOME_PATH/bin/ld.lld"
test -f "$PTO_ISA_ROOT/include/pto/common/kernel_meta.hpp"
test -f tests/atomic_probe/ccec/nested_lambda_cross_tu.cpp
test -f tests/atomic_probe/ccec/nested_lambda_cross_tu_api.h
test -f tests/atomic_probe/ccec/nested_lambda_only_weak_submit_noinline.cpp
```

### 2.2 CPU 语义和 inline runtime-read

```bash
tests/atomic_probe/run_nested_lambda.sh cpu
```

关键期望输出：

```text
[ASSERT] CPU nested capture/template semantics                PASS
[VALUES] rounds=64 mismatches=0 checksum=0x3e6cd1b792bff0e0 L0TaskArgs=1024B
[ASSERT] CPU L0TaskArgs args-runtime-read semantics PASS
[SUMMARY] semantic_failures=0
```

CPU 和 AIC 都从 `nested_lambda_cross_tu_api.h` 使用同一份 inline
runtime 实现。CPU 不再链接独立 runtime TU。

### 2.3 默认单对象、全 inline 正向组

构建：

```bash
tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu build
```

构建过程会检查：

```text
[ASSERT] CCEC caller-capture runtime symbol shape PASS
[VALUES] aic_input_objects=1 runtime_text=n/a submit_symbols=none
```

运行默认正向组：

```bash
tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
```

固定顺序和期望结果：

```text
args-runtime-read                 PASS
weak-context-materialize-0        PASS
run_failures=0
```

也可以单独运行 m0：

```bash
ATOMIC_PROBE_MODE=weak-context-materialize-0 \
  tests/atomic_probe/ccec/run_all.sh nested_lambda_cross_tu run
```

严格 oracle 为：

```text
rounds=64
mismatches=0
dispatches=128
materializations=0
checksum=0x3e6cd1b792bff0e0
L0TaskArgs=1024B
进程退出码=0
```

### 2.4 双对象数量控制

该 fixture 仍向 `ld.lld` 传入两个 AIC 对象，但第二个对象的
`.text` 大小必须为零，且所有 submit 已进入 caller 对象并被内联。

```bash
tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_inline_plus_empty_runtime build

tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_inline_plus_empty_runtime run
```

构建期望：

```text
[VALUES] aic_input_objects=2 runtime_text=000000 submit_symbols=none
```

运行期望：m0 PASS、退出码 0。该控制证明两个链接输入对象本身不足以触发
异常。

### 2.5 only-weak-submit-noinline 故障控制

该 fixture 仍只编译一个有代码的 AIC 对象，但仅保留
`nested_probe_submit_weak_context` 为 noinline。其他 runtime submit、
digest、consume 和 control 均保持 `always_inline`。

```bash
tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline build
```

构建期望：

```text
[VALUES] aic_input_objects=1 runtime_text=n/a
submit_symbols=*nested_probe_submit_weak_context*
```

“only-weak-submit-noinline”是相对 all-inline fixture 的差分命名：额外保留下来
的是 weak-context submit 边界。它不表示最终 ELF 的其余 consume/dispatcher
路径在所有 codegen 形态下都必然没有 out-of-line FUNC。

先运行通过控制：

```bash
ATOMIC_PROBE_MODE=strong-context \
  tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline run

ATOMIC_PROBE_MODE=weak-context-materialize-1 \
  tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline run
```

两条命令都必须退出 0。

最后运行故障控制：

```bash
set +e
ATOMIC_PROBE_MODE=weak-context-materialize-0 \
  tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline run
bad_rc=$?
set -e
echo "bad_rc=$bad_rc"
```

受影响 CCEC 上的期望结果：

```text
ACL error 507015 from aclrtSynchronizeStream(stream) ...
CCEC [...] failed runs: 1
bad_rc=1
```

这里的退出码 1 表示故障控制命中，不是构建失败。该组必须在独立 host
进程中运行，并放在所有通过控制之后。

### 2.6 复现 5/5、3/3 的进程次数

前几节每条 `run` 命令只启动一个独立 host 进程。若报告 `5/5` 或 `3/3`，
必须显式循环并保存完整 stdout，例如：

```bash
set -euo pipefail
log_dir=${NESTED_LAMBDA_LOG_DIR:-/tmp/nested_lambda_matrix_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$log_dir"
echo "logs=$log_dir"

run_passes() {
  target=$1
  mode=$2
  count=$3
  for i in $(seq 1 "$count"); do
    log="$log_dir/${target}_${mode}_${i}.log"
    env ATOMIC_PROBE_MODE="$mode" \
      tests/atomic_probe/ccec/run_all.sh "$target" run >"$log" 2>&1
    cat "$log"
  done
}

run_passes nested_lambda_cross_tu \
  weak-context-materialize-0 5
run_passes nested_lambda_inline_plus_empty_runtime \
  weak-context-materialize-0 5
run_passes nested_lambda_only_weak_submit_noinline \
  strong-context 3
run_passes nested_lambda_only_weak_submit_noinline \
  weak-context-materialize-1 3

for i in $(seq 1 5); do
  log="$log_dir/noinline_m0_${i}.log"
  set +e
  env ATOMIC_PROBE_MODE=weak-context-materialize-0 \
    tests/atomic_probe/ccec/run_all.sh \
    nested_lambda_only_weak_submit_noinline run >"$log" 2>&1
  rc=$?
  set -e
  cat "$log"
  test "$rc" -eq 1
  grep -Fq 'ACL error 507015 from aclrtSynchronizeStream(stream)' "$log"
  grep -Fq 'run_failures=1' "$log"
  # 进入下一轮前，按本机流程确认 device/driver 已恢复可运行状态。
done
```

故障循环必须最后运行；每次仍由 runner 创建独立 host 进程。
进程退出码为 1 不足以证明命中该问题，还必须在同轮日志中同时确认
`aclrtSynchronizeStream` 返回 507015 和 `run_failures=1`。没有逐轮日志时，
不得只凭一条命令把结果写成 5/5。

## 3. 被验证的问题

### 3.1 原始调用形态

业务中的 private-lazy 路径会在 orchestration 栈上构造多个 `Tensor`，
再把对象地址写入 caller context：

```text
orchestration caller
  -> caller 栈上的 Tensor first/second/third
  -> CallerContext {&first, &second, &third, salt}
  -> submit(context, args)
  -> dispatcher 解引用 context 中的 Tensor*
```

已有证据包括：

- 优化后的 LLVM IR 正确写入三个 `Tensor *`；
- orchestration CFA 为 `reg93 + 1952`，不是普通栈容量不足；
- 32 KiB 和 64 KiB 栈配置都失败；
- 无业务用途的地址写会改变 PASS/FAIL；
- 去掉 `-cce-aicore-addr-transform` 后业务仍失败；
- 业务原发日志包含无效 GM 地址访问。

这些是原业务排查期间的历史观察；对应 optimized LLVM IR、32/64 KiB 构建
产物、addr-transform 对照 stdout 和原发日志没有随本文持久化，不能作为当前
HEAD 可独立复跑的 fixture。本文的可复现结论以第 2、4、7 节列出的持久
源码、命令和明确标注状态的 probe 为准。

### 3.2 本轮新增的判别证据

本轮把“对象数量”和“函数调用边界”拆开：

```text
两个对象 + 空 runtime + submit 全 inline
  -> PASS

一个对象 + weak-context submit noinline
  -> 507015
```

因此多对象链接不是必要条件；在这组历史 fixture 中，未内联
submit 是 m0 触发故障时存在的必要构建差分。它仍不是单独的
充分解释，因为：

- 同一 noinline fixture 的 m1 通过；
- 同一 fixture 的 strong 路径通过；
- 全 inline 的 weak m0 通过。

最合理的范围仍是优化 LLVM IR 到 HiIPU 最终机器码之间的地址物化、
活跃区间、调度和寄存器分配组合。

当前安装的 `llvm-objdump` 能读取 `elf64-hiipu` 符号和 DWARF，但指令
反汇编仍显示 `<not available>`。第 3.3 节已经补充 CANN simulator 随包真实
decoder，可用于审计 final ELF 的机器指令；但历史 507015 日志没有可用 fault
PC，因此本文仍不声称已经把该异常定位到某一条具体机器指令。

### 3.3 HiIPU final ELF 的真实反汇编方法

#### 3.3.1 解码器来源与接口

CANN 9.1 simulator 安装目录包含真正的 HiIPU decoder，而不是只识别 ELF
容器、对指令输出 `<not available>` 的通用 `llvm-objdump`。本机已验证文件为：

```text
/home/q00473782/Ascend/cann-9.1.0-weekly-20260708/cann-9.1.0/
x86_64-linux/simulator/dav_3510/lib/libpem_davinci.so
```

其 SHA256 为：

```text
29835d2439d6dd464d34a212ad4bbd5c29af6a38465da09a6c273401d9a96dcb
```

该文件与 CANN toolkit 安装包内层 simulator 包中的
`lib64/Ascend950pr_9599/lib/libpem_davinci.so` 字节相同。动态符号表导出：

```text
000000000175c5d0 T pem_turbo_objdump
```

当前没有找到该符号的公开 header 或官方稳定 ABI 声明。仅对上述 SHA256 对应
的 `.so`，经 ABI 反查和构造机器字样本双重验证，接口等价于：

```cpp
extern "C" char *pem_turbo_objdump(
    const uint32_t *words,
    size_t word_count,
    const uint64_t (*rvec_ranges)[2],
    size_t range_count);
```

返回值是 decoder 分配的 C 字符串，调用者必须 `free()`。最小 Python 调用形态
如下；`restype` 必须使用 `c_void_p`，不能让 `ctypes` 先把返回值转换成 Python
字符串后再误释放：

```python
import ctypes

pem_so_path = "/path/to/simulator/dav_3510/lib/libpem_davinci.so"
pem = ctypes.CDLL(pem_so_path)
dump = pem.pem_turbo_objdump
dump.argtypes = (
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
)
dump.restype = ctypes.c_void_p

words = (ctypes.c_uint32 * (len(body) // 4)).from_buffer_copy(body)
result = dump(words, len(words), None, 0)
if not result:
    raise RuntimeError("pem_turbo_objdump returned null")
try:
    disassembly = ctypes.string_at(result).decode("utf-8")
finally:
    libc = ctypes.CDLL(None)
    libc.free.argtypes = (ctypes.c_void_p,)
    libc.free(result)
```

升级 CANN 或换用不同 SHA 的 decoder 时，必须重新验证符号、签名、返回值所有权
和构造样本，不能把这里的反查结果当作跨版本官方接口。

`rvec_ranges` 的每项是相对传入 `words[0]` 的 byte offset 半开区间
`[start, end)`；本次 `words[0]` 恰好是 entry body 起点，所以表现为
body-relative。本次六份 STOP orchestration 经范围样本和完整解码确认均为
scalar control code，因此传 `NULL, 0`；不能只因为对象属于 AIV lane 就把整个
entry 强制送入 RVec decoder，也不能把该结论泛化到所有 orchestration。只有
已经由符号或 section 边界确认的真实 RVec 片段才应填写 range。

#### 3.3.2 必须解 final-linked entry，而不是中间对象

分析运行代码时应从最终 `aicore_kernel.o` 的 `.text` 截取 entry，不能把
LLVM IR、backend MachineInstr 或 relocation 尚未应用的 combined object
字节称作最终汇编。固定步骤为：

1. 记录并校验 final ELF SHA256；
2. 从 ELF64 section table 取得 `.text` 的 `sh_addr/sh_offset/sh_size`；
3. 从 lane combined object 的符号得到 entry `st_value/st_size`；
4. 还原 final entry PC，并按下式从 final ELF 截取机器码：

   ```text
   relative = entry_final_pc - final_text_sh_addr
   body = elf[final_text_sh_offset + relative :
              final_text_sh_offset + relative + entry_size]
   ```

本次六份冻结链接产物只保留一个弱 AIC `aicpu_orchestration_entry` 符号。
AIV entry 不能臆测为 AIC entry，也不能只在 final ELF 中按字节特征搜索。
这六份已经分别核验 final `.text` 满足
`AIC combined .text + pad-to-0x100 + AIV combined .text`，因此本次按下式还原：

```text
aiv_entry_text_offset = align_up(aic_combined_text_size, 0x100)
                        + aiv_combined_entry_st_value
aiv_entry_final_pc = final_text_sh_addr + aiv_entry_text_offset
```

AIC final PC 则使用 final ELF 符号值。本次冻结 final ELF 的 `.text.sh_addr=0`，
所以 AIV text offset 与 final PC 数值相同；不能把这一巧合写进通用公式。
entry 大小仍取对应 lane combined object 的 `st_size`。上述布局关系不是通用
链接规则；链接顺序、linker script、alignment、section GC 或输入对象改变时，
都必须从新产物自己的 link map、section/symbol 和最终尺寸关系重新证明，
不能沿用旧 PC。

#### 3.3.3 完整性门禁与地址标注

每份反汇编至少同时满足以下条件后才能作为机器码证据：

- `entry_size % 4 == 0`；
- decoder 正文行数精确等于 `entry_size / 4`，逐个覆盖所有机器字；
- 输出中没有 `unknown/illegal/invalid/error/not available`；
- entry body SHA256、final ELF SHA256、final PC 和 size 一并记录；
- 函数尾边界与符号尺寸一致；当前六份 STOP entry 的最后一条均为 `RET`。

`pem_turbo_objdump` 原始地址列是相对输入 body 的 function-relative offset。
用于 final ELF 分析时，应同时保留该 offset，并标注：

```text
instruction_final_pc = entry_final_pc + decoder_offset
```

2026-07-19 用该方法解码冻结的 A/B/C STOP orchestration，AIC/AIV 分别得到：

| 形态 | AIC | AIV |
| --- | ---: | ---: |
| A split eager | 3,252 B / 813 条 | 3,228 B / 807 条 |
| B fused eager | 3,252 B / 813 条 | 3,228 B / 807 条 |
| C fused lazy | 16,688 B / 4,172 条 | 16,636 B / 4,159 条 |

六份均通过上述完整性门禁。A/B 的 combined raw entry 逐字节相同，但放入各自
final ELF 后分别有 21 个 AIC、19 个 AIV 指令字因链接布局/重定位值而不同；
这正是必须保留 final-linked 版本的原因。C 的 AIC/AIV 各解出 19 条 `ATOM`，
A/B 为 0。这些只作为反汇编方法和静态 codegen 差异的验真示例：STOP 仍保留
stop-only 清理、selector/counter 读取和 synthetic row，是诊断工件；上述
静态差异不证明 production 性能、I-cache 归因或历史 507015 根因。

当前机器的生成物、manifest、原始 entry body 和复跑脚本位于：

```text
outputs/qk_lazy_abc_stop_orchestration_asm_20260719/
```

该目录属于本机生成输出且被 `.gitignore` 忽略；可复现口径以上述 decoder
来源、ELF/entry 定位和完整性门禁为准。其中 `*.final-machineinstr.txt` 只是
HiIPU Assembly Printer 编码前的 backend 交叉证据，不是本文所称的 final ELF
反汇编。

### 3.4 两种规避形态

全 inline 形态：

```text
caller 栈地址
  -> inline submit/helper
  -> 不跨 nested_probe_submit_weak_context 函数边界
  -> dispatcher/consume
```

数据驱动形态：

```text
caller 栈地址
  -> 写入既有 L0TaskArgs 固定 slot
  -> inline args-runtime-read 直接读取
  -> submit 返回前完成 add_input 和结果物化
```

两者都在历史受影响工具链上通过。当前 `2026-06-10` CCEC 只完成了
本文记录的增量 build/ELF 复核，未重跑这两组设备矩阵。生产实现仍必须
保证 submit 返回前完成复制、物化和同步消费，不能让异步阶段继续
保存 caller 栈裸地址。

### 3.5 面向 lazy submit 的推荐调用边界

`all-inline` 不等于必须把完整 runtime 与 orchestration 融为一个 TU。由上述
caller-stack 证据收敛出的目标边界是：closure、callback 和 nested thunk
不跨未内联边界；已经物化的参数容器可以由一个同步 finish 消费。

```text
一个对外 submit API
  → inline front：fatal/task-id/execute-first drain/claim
  → caller 内 callback 恰好一次
      winner：求值 Tier-1 + Tier-2 参数
      private loser：只求值必须保持每核状态一致的 Tier-1 参数
  → callback 生命周期结束
  → 一次 noinline cross-TU finish(POD ticket, const L0TaskArgs &)
  → finish 同步消费并在 submit 返回前结束
```

这里必须同时满足：

1. outer callback 每个 submit 恰好调用一次；不能用 all-core `Prepare` 加
   winner-only `WinnerBind` 把 winner 变成两次调用；
2. execute-first drain 仍在 claim 之前，参数条件求值在 claim 之后；
3. callback 类型、closure 地址、独立 `CallerContext *` 和 thunk 对象都不得
   作为 finish 参数，也不得形成 finish 侧可调用的 function pointer；
4. finish 只接收 POD ticket 与已建好的 `L0TaskArgs`；它可以同步解引用其中
   仍在 caller 栈生命周期内的 Tensor/TensorCreateInfo，但不得保存到 submit
   返回后的异步阶段；
5. 热路径只保留一次 out-of-line finish。若先调用独立 runtime begin/claim，
   返回后再调用 finish，即使对外 API 和 callback 都只有一次，也已经把一次
   runtime 边界变成两次，不能默认视为等价性能实现；
6. 不允许靠增加无业务地址物化、扩大栈或偶然改变代码布局作为正式规避。

`nested_lambda_claim_first_once` 是这一形态的最小 AIC probe：inline front
执行真实 `atomicMax` claim，callback 中包含捕获 caller 栈对象的 nested
thunk，最后只把 `L0TaskArgs * + won` 交给独立 TU 的 noinline finish。当前
构建已断言：caller/runtime 两个 AIC 对象、无 callback/context 的 out-of-line
符号、恰好一个 finish 符号、最终 ELF 无 relocation。2026-07-18 的真实 A5
设备 oracle 以 4 blocks × 64 rounds 运行 1 次，全部精确 PASS。

它的覆盖边界也必须写明：当前只有 AIC、4 blocks；所谓 execute-first drain
只是顺序 marker，不是真实 `EfDrain`；claim 是直接对 GM 做 `atomicMax`，不
使用 FDWIC worker state；finish 参数只是 `L0TaskArgs * + won`，尚未覆盖完整
POD ticket。因此它证明的是目标**编译/调用形态**，不是完整 production
runtime 已被等价验证。

### 3.6 split runtime 的 external `[[block_local]]` 前置条件

上述边界要在 split TU 中实现，inline front 必须访问与 runtime helper 同源的
worker state。若状态仍以 header 内 `static [[block_local]]` 定义，runtime TU
和 orchestration TU 会各自拥有一份对象，源码名字相同也不能据此认为状态
同源。

推荐的链接约束是：

```text
runtime-owning TU:
  [[block_local]] pto_fdwic_aic_<state>;  // AIC 唯一定义
  [[block_local]] pto_fdwic_aiv_<state>;  // AIV 唯一定义

orchestration/internal header:
  [[block_local]] extern pto_fdwic_aic_<state>;
  [[block_local]] extern pto_fdwic_aiv_<state>;
  #ifdef __DAV_VEC__ 选择 AIV，否则选择 AIC
```

不能用同名 weak 对象让 linker 猜核型。静态验收至少包括：

- 未链接 runtime object 对对应核型 state 恰好一个 OBJECT 定义；
- orchestration object 只有对应核型 state 的 UND 与 text relocation，不引用
  错核型 state；
- final ELF 中 AIC/AIV state 各恰好一个定义、无未决 relocation；
- orchestration object 没有 `begin/prepare` 的 UND/call site，只有一次对应
  finish call site；
- AIC/AIV finish 使用强、分核型符号，不能依赖 generic weak dispatcher。

`block_local_cross_tu` 已独立覆盖其中的 state 链接形态：AIC/AIV 分名，
runtime 定义、orchestration extern 导入，最终 ELF 两个 state 各唯一、相隔
64 B 且无 relocation。它的设备 oracle 会在 4 个 block 上执行 257 轮递推，
并比较两个 TU 观察到的地址、终值、cookie 和轮数。2026-07-18 的真实 A5
结果为 AIC 5/5、AIV 1/1 PASS；每次 `concurrent_block_arrivals == 4`。

该 probe 尚未证明真实 FDWIC 的全部 worker state、108-worker AIC/AIV 同次
replay 或 `DistCore/DistGlobal` 指针都正确。production D 不能只因这个小
对象能链接就跳过真实状态集合和 PA 结构 oracle。

### 3.7 两个 probe 必须组合看，不能互相替代

`nested_lambda_claim_first_once` 验证“callback/Arg/finish 边界”，但不使用
external `[[block_local]]`；`block_local_cross_tu` 验证“跨 TU state 同源”，
但不包含 callback、`L0TaskArgs` 或 finish。后续 split-lazy 实现必须同时满足
两组约束，再补一个组合后的真实 submit oracle；任何一个单独 PASS 都不足以
宣布 D 已经可用。

## 4. 测试结构和 oracle

### 4.1 语言语义层

CPU、AscendC AIV 和纯 CCEC AIV 共同覆盖：

- outer lambda 按引用捕获；
- nested lambda 按值、按引用和混合捕获；
- 自由函数模板 `Submit<BuildCallback>`；
- 成员函数模板 `AddInput/AddOutput/AddScalar<Thunk>`；
- AICore lambda 的 `__aicore__` 标注。

固定 `seed=0x120` 的期望值为：

| 字段 | 期望值 |
| ---- | -----: |
| outer/input/output/scalar 调用次数 | 各 1 |
| `reference_state` | 300 |
| `input.value` | 591 |
| `output.value` | 323 |
| `scalar` | 337 |
| `combined` | 1251 |

### 4.2 caller-capture 语义层

每个变体执行 64 轮，每轮执行 4 次 submit：

- 第一次为待测 caller-capture 或 args-runtime-read 路径；
- 后三次为固定 control 路径；
- 每轮重新创建三个 caller 栈 `Tensor`；
- 同一轮复用一个 `L0TaskArgs`。

精确 oracle：

| 字段 | 期望 |
| ---- | ---: |
| completed rounds | 64 |
| mismatches | 0 |
| submits | 256 |
| checksum | `0x3e6cd1b792bff0e0` |
| `sizeof(L0TaskArgs)` | 1024B |
| callback dispatcher 次数 | 128 |
| args-runtime-read dispatcher 次数 | 0 |

`L0TaskArgs` 的测试 slot：

| slot | 内容 |
| ---: | ---- |
| scalar 8 | `&first` |
| scalar 9 | `&second` |
| scalar 10 | `&third` |
| scalar 11 | `salt` |
| scalar 0 | 计算结果 |
| scalar 5 | dispatcher 次数 |
| scalar 6 | dispatcher 缺失标志 |

这些编号仅属于探针，不是生产 ABI。

### 4.3 七个语义变体

| entry | 运行参数 | 独立 context | 额外写入 Arg 的 caller 地址数 | dispatcher |
| ----: | -------- | ------------ | -------: | ---------- |
| 0 | `weak-context-materialize-0` | 有 | 0 | weak |
| 1 | `weak-context-materialize-1` | 有 | 1 | weak |
| 2 | `weak-context-materialize-2` | 有 | 2 | weak |
| 3 | `weak-context-materialize-3` | 有 | 3 | weak |
| 4 | `weak-args-storage` | 无 | 3 | weak |
| 5 | `strong-context` | 有 | 0 | strong |
| 6 | `args-runtime-read` | 无 | 3 | 无回调 |

变体本身不再绑定固定 PASS/FAIL。结果必须同时写明构建 fixture。尤其 m0
在 all-inline fixture 中通过，在 only-weak-submit-noinline fixture 中失败。

### 4.4 三个持久构建 fixture

| target | AIC 输入对象 | submit FUNC 符号 | 默认运行 |
| ------ | -----------: | ---------------- | -------- |
| `nested_lambda_cross_tu` | 1 | 0 | args-runtime-read、m0 |
| `nested_lambda_inline_plus_empty_runtime` | 2 | 0 | m0 |
| `nested_lambda_only_weak_submit_noinline` | 1 | 仅 weak-context submit | strong、m1、m0 |

第三个 target 的 m0 放在最后；预期 runner 退出 1。默认 pytest 不运行该
故障组。

### 4.5 claim-first / single-callback / split-finish 增量 probe

`nested_lambda_claim_first_once` 的静态和设备 oracle 分开：

| 层级 | 精确检查 | 当前状态 |
| --- | --- | --- |
| caller object | callback/builder/context 无 out-of-line FUNC；finish 恰好一个 UND 和 relocation | PASS |
| runtime object | noinline finish 恰好一个强定义 | PASS |
| final ELF | 两个 AIC 输入对象；finish 恰好一个定义；无 relocation | PASS |
| device | 4 blocks × 64 rounds；每轮 outer/output/inout 各一次，input/scalar 仅 winner 一次；每轮唯一 winner；digest 精确匹配 | 1/1 PASS |

它不把顺序 marker 冒充真实 execute-first drain，也不把 `L0TaskArgs * + won`
冒充完整 production ticket。

### 4.6 external `[[block_local]]` 增量 probe

`block_local_cross_tu` 分别编译 AIC runtime/orchestration 与 AIV
runtime/orchestration。四个对象合成的 combined ELF 只用于静态审计；设备运行
使用 AIC-only、AIV-only 两个独立 ELF，二者的入口 suffix 都是 `0`。静态
oracle 已通过：

- runtime object 只定义本核型 state；
- orchestration object 只导入本核型 state；
- AIC/AIV state 在 combined ELF 中各唯一，均为 64 B、位于
  `.bl_uninit`、按 64 B 对齐且地址相隔 64 B；
- combined、AIC-only、AIV-only 三个 ELF 均无 relocation；
- 两个 runnable ELF 都拒绝另一核型的 state、入口和 metadata。

设备 oracle 为 AIC 5 个独立进程、AIV 1 个独立进程，每次 4 blocks ×
257 轮。4 个 block 先通过 atomic arrival barrier 同时完成各自 state 初始化，
再开始递推；host 还会精确检查 `concurrent_block_arrivals == 4`。2026-07-18
真实 A5 结果为 AIC 5/5、AIV 1/1 PASS；每核地址、终值、cookie、轮数和 guard
均精确匹配。

## 5. 源码和测试入口

| 文件 | 作用 |
| ---- | ---- |
| `nested_lambda_probe.h` | 三端共享的语言语义和 oracle |
| `cpu/nested_lambda.cpp` | 标准 C++17 语言对照 |
| `cpu/nested_lambda_args_runtime_read.cpp` | CPU inline runtime-read 对照 |
| `ccec/nested_lambda_cross_tu.cpp` | AIC caller、dispatcher 和七个入口 |
| `ccec/nested_lambda_cross_tu_api.h` | inline runtime 实现和 caller context |
| `ccec/nested_lambda_cross_tu_runtime.cpp` | `.text=0` 的第二对象控制 |
| `ccec/nested_lambda_inline_plus_empty_runtime.cpp` | 双对象数量控制 wrapper |
| `ccec/nested_lambda_only_weak_submit_noinline.cpp` | only-weak-submit-noinline wrapper |
| `ccec/nested_lambda_cross_tu_layout.h` | 变体、字段和精确 oracle |
| `ccec/nested_lambda_cross_tu_host.cpp` | raw ELF launcher 和结果校验 |
| `ccec/nested_lambda_claim_first_once.cpp` | AIC inline claim-first + callback once caller |
| `ccec/nested_lambda_claim_first_once_runtime.cpp` | 独立 TU 的同步 noinline Arg finish |
| `ccec/nested_lambda_claim_first_once_api.h` | split finish ABI |
| `ccec/nested_lambda_claim_first_once_layout.h` | claim-first 精确 oracle |
| `ccec/nested_lambda_claim_first_once_host.cpp` | claim-first raw ELF launcher/oracle |
| `ccec/block_local_cross_tu_{runtime,orchestration}.cpp` | AIC/AIV external `[[block_local]]` 跨 TU 两端 |
| `ccec/block_local_cross_tu_layout.h` | state 递推布局与精确期望 |
| `ccec/block_local_cross_tu_host.cpp` | external state raw ELF launcher/oracle |
| `ccec/run_block_local_cross_tu.sh` | external `[[block_local]]` 构建、ELF 检查和运行 |
| `ccec/run_all.sh` | caller-capture 与 claim-first fixture 的构建、ELF 断言和运行 |
| `run_nested_lambda.sh` | CPU、AscendC、CCEC 统一入口 |
| `test_atomic_probe.py` | CPU 和默认 A5 正向 pytest |

文件名中的 `cross_tu` 是历史命名。当前默认 target 是单个有代码的 AIC
输入对象，不能再根据文件名推断构建形态。

## 6. 自动化测试

### 6.1 CPU pytest

```bash
export PYTHONPATH=python
.venv/bin/python -m pytest \
  tests/atomic_probe/test_atomic_probe.py::test_cpu_nested_lambda_compiler_probe \
  -q -s
```

### 6.2 A5 默认正向 pytest

```bash
export PYTHONPATH=python
.venv/bin/python -m pytest \
  tests/atomic_probe/test_atomic_probe.py::test_a5_ccec_nested_lambda_call_boundary_controls \
  --platform a5 --device 0 \
  -q -s
```

该 pytest 会：

1. build 单对象 all-inline target，并验证 submit FUNC 符号为零；
2. 先运行 args-runtime-read，要求 PASS；
3. 再运行 m0，要求 PASS。

它不会运行 only-weak-submit-noinline m0，避免默认 CI 故意制造 AICore
exception。双对象数量控制和 noinline 故障控制使用第 2.4、2.5 节的显式
命令。

### 6.3 完整语言入口

```bash
tests/atomic_probe/run_nested_lambda.sh all
```

`all` 依次运行 CPU、AscendC、纯 CCEC AIV 和默认 AIC caller-capture
正向组。

### 6.4 两个 lazy-split 前置 probe

只做 CCEC/链接检查，不占设备：

```bash
tests/atomic_probe/ccec/run_all.sh nested_lambda_claim_first_once build
tests/atomic_probe/ccec/run_block_local_cross_tu.sh build
```

已有构建产物上板：

```bash
tests/atomic_probe/ccec/run_all.sh nested_lambda_claim_first_once run
tests/atomic_probe/ccec/run_block_local_cross_tu.sh run
```

两条 run 都不是默认 pytest 的一部分。设备运行必须保留 stdout、CCEC SHA、
最终 ELF SHA 和进程次数；没有这些证据时只能报告 build/ELF 形态通过。

## 7. 本轮实测结果

软件环境：

```text
Historical run worktree base HEAD: 3a3de54db0a900e489a5a5496cbee1dc5b76be7a
First commit containing the complete durable fixtures: a3f5ecc2186fb8a06cded6d26886a4ab802009c0
CCEC: clang 15.0.5, build 2026-07-07T20:35:46+08:00
GCC: 13.3.0
AIC arch: dav-c310-cube
Device: A5 device 0
```

上述 5/5、3/3 是历史设备结果。当时是以 `3a3de54d` 为 HEAD 的 dirty
worktree；单独 checkout `3a3de54d` 不包含本文的两个关键 wrapper，不能按文中
命令复现。持久 fixture 源码从 `a3f5ecc2` 开始可获得，但当时 dirty patch、
`2026-07-07` CCEC 二进制路径/SHA 和逐轮 stdout 没有持久化，因此不能
宣称 `a3f5ecc2` checkout 与历史运行产物逐字节相同。这些结果也不能自动
外推到当前 `2026-06-10` CCEC。后续复跑必须按文首格式绑定实际
源码状态、编译器 SHA 和逐轮日志。

持久 fixture 结果：

| fixture / variant | 次数 | 结果 |
| ----------------- | ---: | ---- |
| 单对象 all-inline / m0 | 5 | 5/5 PASS |
| 双对象、空 runtime / m0 | 5 | 5/5 PASS |
| only-weak-submit-noinline / m1 | 3 | 3/3 PASS |
| only-weak-submit-noinline / strong | 3 | 3/3 PASS |
| only-weak-submit-noinline / m0 | 5 | 5/5 507015 |
| 单对象 all-inline / args-runtime-read | 1 | PASS |
| CPU 语言语义 | 1 | PASS |
| CPU inline args-runtime-read | 1 | PASS |

此外，当时曾临时恢复原始 external runtime 源码，重新生成双 `.o`
产物；m0 记录为 5/5 507015。该临时 patch、产物和逐轮日志未持久化，
只能作为历史观察，不是当前 runner 的可独立复现 target。

故障组在 `aclrtSynchronizeStream` 返回 507015，不能回读 rounds、
mismatch 或 checksum。表中不为故障组伪造设备结果。

2026-07-18 增量构建结果：

| probe | CCEC/链接/符号检查 | final ELF SHA256 | 设备结果 |
| --- | --- | --- | --- |
| `nested_lambda_claim_first_once` | PASS | `5c1ed0935f700b4d3b7d8a66a22be6b0d9402938e0c568ce9ce8cd15ed38b861` | 1/1 PASS |
| `block_local_cross_tu` combined 静态审计 ELF | PASS | `b6d618d2fb04fcc88dea0a142ade2cf0b1f3f4236877bd9fa9bf4538c54e4091` | 不运行（仅静态审计） |
| `block_local_cross_tu` runnable AIC ELF | PASS | `0632c0bcc1577d69fb976e3d3b0dc0ed949eea8c14e6437a7ab79422067ed489` | 5/5 PASS |
| `block_local_cross_tu` runnable AIV ELF | PASS | `e86523127e71e1fd99c8166f63ea3a00d437f096f2c9298071f82aced435abd0` | 1/1 PASS |

这些 SHA 绑定当时源码和构建产物；相关 probe 源码后来固化于 `d11277bd`，
但该提交及原始产物/日志未纳入当前 cleanroom。SHA 不能代替源码提交标识，
也不能作为当前 HEAD 已重建的证明。

## 8. 如何解释结果

| 观察 | 可以支持 | 不能推出 |
| ---- | -------- | -------- |
| 双对象空 runtime 通过 | 两个链接输入不足以触发 | `ld.lld` 已被全面证明无缺陷 |
| 单对象 noinline m0 失败 | 多对象链接不是必要条件 | 任意 noinline 调用都会失败 |
| all-inline m0 通过 | inline 可规避当前形态 | inline 修复了 CCEC 后端 |
| noinline m1 通过 | 地址物化会改变最终 codegen | 增加一次 store 是生产修复 |
| 同 fixture strong 通过 | strong 路径是有效控制 | weak 单独就是根因 |
| args-runtime-read 通过 | inline 消费时，可取消独立 context 并从既有 Arg slot 取数 | split/noinline runtime 直读同样可行，或真实 PA 已完成修复 |
| claim-first-once 静态形态通过 | 最终 ELF 可表达“callback/closure 留在 caller，只保留一个 split finish 边界” | finish 已在设备上正确消费，或完整 EfDrain、POD ticket、external state 已通过 |
| block-local 静态形态通过 | AIC/AIV external `[[block_local]]` 可形成预期定义/导入关系 | 两个 TU 在设备上已看到同一地址，或真实 FDWIC 全状态已正确 |
| 两个增量 probe 分别通过构建 | 两组约束可以独立检查 | external state + callback once + finish 的组合路径已被验证 |

507015 在 CANN 中表示 AICore exception。业务经过 AICPU 外层时可能报告
507018；错误层级与调用路径不同，不能只凭错误码断言 fault PC 相同。

## 9. 生产落地约束

### 9.1 lazy-lambda 实现硬约束

1. 每个业务 submit 只能调用一次 outer callback；winner/loser 差异在同一次
   callback 内决定 thunk 是否求值；
2. submit 返回前必须完成 caller 栈对象的复制、物化和同步消费，不得让异步
   阶段保存 caller 栈裸地址；
3. callback、closure、thunk 或独立 caller context 不得跨 noinline/cross-TU
   边界；允许跨界的是 POD ticket 和已经建立的 `L0TaskArgs`；
4. split finish 必须同步返回，且每 submit 只有一次 out-of-line finish；
5. 保持 `task-id → execute-first drain → claim → 条件参数求值 → finish`，不得
   为提前 claim 越过 execute-first drain；
6. split front 所用 `[[block_local]]` 状态必须由 runtime TU 唯一定义，AIC/AIV
   物理分名，orchestration 只做 extern 导入；
7. CCEC build 必须检查 caller object 不含 out-of-line callback/context，只有
   一个 finish call site；final ELF 不得有未决 relocation；
8. inline/fusion 都可能改变 text 体积、outlining 和布局，真实业务必须检查
   最终 ELF 与实机性能，不能由源码“少做了参数”直接推定收益；
9. CCEC codegen 高度敏感，每组结果必须记录源码提交/dirty patch、编译器绝对
   路径与 SHA、最终 ELF SHA、运行命令和逐进程 stdout；
10. 无业务地址写、额外 materialization 或单纯扩大栈只能用于定因，不能作为
    正式规避；
11. only-weak-submit-noinline m0 只属于显式诊断，不进入默认 CI；
12. 两个前置 probe 必须分别通过静态和设备 oracle，完整业务还要补真实
    worker state、task/claim/callback 次数及数值/结构 oracle。

### 9.2 与 PA A/B/C/D 的接口关系

本文只规定调用边界；以下标签的业务性能、PMU 和取舍以
`tests/atomic_probe/pa_scheduler/make_replay_faster_qk_single_callback_a5_analysis.md`
为准：

| 形态 | 本文关注的调用边界 | 本文能否单独判定其性能 |
| --- | --- | --- |
| A：split eager | caller eager 构造参数，一次进入 split `dist_submit_impl` | 不能；只作为既有边界基线 |
| B：fused eager control | eager 参数流程不变，构造与 C 相近的 fusion/codegen 控制 | 不能；它是归因控制，不是推荐实现 |
| C：fused lazy | inline drain/claim/callback once，一次 noinline finish；不跨 closure/context | 只能判断边界形态符合本节约束，不能由此宣布有收益 |
| D：split lazy | 动态顺序和调用次数与 C 相同，仅把 runtime/state 恢复为 split TU | 必须先过两个前置 probe 和组合 oracle，再做真实 A5 A/B |

D 若把 header `static [[block_local]]` 改成 external state，直接做 `D-A` 会
混入 state-export 改动。至少还要有 `A′=split eager + 同一 state-export
patch`，分别报告 `A′-A` 和 `D-A′`；不能把 linkage 变化算成 lazy 本身。

### 9.3 不接受为默认实现的形态

- `runtime begin/claim → callback → runtime finish`：callback 虽一次，但每
  submit 有两次 runtime 边界，属于额外热流程；
- all-core `Prepare` + winner-only `WinnerBind`：loser dispatcher 一次、winner
  两次，违反 single-callback 目标；
- `m0 caller context + noinline weak-context submit + weak dispatcher` 组合：
  已在历史受影响 CCEC 上命中 507015 故障边界。这不表示 weak 单因素
  是根因，但 lazy-lambda 候选不应重新引入这个已知高风险组合；
- 为了 all-inline 默认融合完整 runtime + orchestration：可作为规避/诊断
  control，但是否采用必须由最终 ELF 和真实 A5 性能决定；
- 增加无业务 store、修改栈大小或依赖偶然布局让 m0 PASS：只能缩小 codegen
  敏感范围，不是生产修复。

## 10. 常见问题

### 10.1 build 目录中混入旧产物

每个 fixture 使用独立 kernel、caller object 和 host 文件名。不要拿
`nested_lambda_cross_tu_host` 启动 noinline kernel，也不要只看 build
目录中是否残留历史 runtime object。

以 runner 的构建输出为准：

```text
aic_input_objects=...
runtime_text=...
submit_symbols=...
```

### 10.2 noinline 完整运行退出 1

以下命令默认按 strong、m1、m0 顺序运行：

```bash
tests/atomic_probe/ccec/run_all.sh \
  nested_lambda_only_weak_submit_noinline run
```

如果前两组 PASS，最后 m0 返回 507015，则最终退出 1 是预期诊断结果。

### 10.3 m0 意外通过或失败

先确认：

- 使用了正确 fixture 的 kernel；
- all-inline target 没有 submit FUNC 符号；
- 相对 all-inline，noinline target 额外保留的 submit FUNC 恰为 weak-context submit；
- CCEC arch 为 `dav-c310-cube`；
- caller orchestration 和七个 metadata entry 仍存在；
- 没有复用另一 fixture 的 host/kernel 组合。

若 noinline m0 仍通过，应记录“当前工具链未命中”，不能添加无关代码强迫
失败。

### 10.4 `tensor.h` unused warning

CCEC 当前会报告 `buffer_elems` 未使用。它不属于本 probe 的 semantic
failure，不应为了该测试修改无关生产头文件。

## 11. 最终检查清单

- [ ] CPU 嵌套 lambda 语义 PASS；
- [ ] CPU inline args-runtime-read PASS；
- [ ] 默认 AIC target 只有一个输入对象；
- [ ] 默认 ELF 没有 `nested_probe_submit_*` FUNC 符号；
- [ ] 若复核历史矩阵，all-inline m0 在同一源码/编译器绑定下为 5/5 PASS；
- [ ] 双对象控制的第二个对象 `.text=0`；
- [ ] 若复核历史矩阵，双对象控制 m0 在同一绑定下为 5/5 PASS；
- [ ] 相对 all-inline，noinline target 额外保留的 submit FUNC 恰为 weak-context submit；
- [ ] 若复核历史矩阵，noinline m1 为 3/3 PASS；
- [ ] 若复核历史矩阵，同 fixture strong 为 3/3 PASS；
- [ ] 若当前 CCEC 仍命中该布局，noinline m0 的每次失败均有明确
      `aclrtSynchronizeStream=507015` 日志，不只是退出码为 1；
- [ ] args-runtime-read checksum 精确匹配；
- [ ] `sizeof(L0TaskArgs) == 1024`；
- [ ] `claim_first_once` caller 无 out-of-line callback/context，只有一次 finish 边界；
- [ ] `claim_first_once` 静态 oracle 与设备 oracle 分开记录；
- [ ] external `[[block_local]]` 的 AIC/AIV state 分名、各唯一、无错核型引用；
- [ ] `block_local_cross_tu` 静态 oracle 与设备 oracle 分开记录；
- [ ] 没有把两个独立 probe 的 PASS 扩大成组合 production 路径已通过；
- [ ] 结果记录 CCEC 绝对路径/SHA、源码提交或 dirty patch、final ELF SHA 和逐轮 stdout；
- [ ] 若实现 D，使用相同 state-export patch 的 A′ 分离 linkage 变化；
- [ ] 没有把 `.o` 数量、`ld.lld`、weak 或某个 pass 写成已定位根因；
- [ ] 默认 pytest 不运行故障组。

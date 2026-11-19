# A100 PP2 正确性验证与 FlashInfer planner 缓冲区修复

已修复两类运行时问题：FlashInfer pinned host planner 缓冲区过早复用，以及 tokenizer consumer 合并响应时丢失增量 token ID。统一源码版本后的九个主配置与五个机制试验均完成；独立 HF 四组评分完成 **906/906 条候选轨迹检查、110016/110016 个生成 token 评分**，无缺失，最大 regret 为 **0.5**。这些是明确输入与配置范围内的结果，**不表示所有正确性检查通过**：严格序列不匹配及相对 HF 的非最大概率选择仍然存在，全部保留。历史发现与早期 ALP 证据见第四、五节，最终 v2 结果见第六节。

## 1. 实验边界与可复查输入

模型为本地 `Llama-3.1-8B-Instruct`，在 GPU 6、7 的两张 A100-SXM4-80GB 上运行 TP1/PP2，BF16，FlashInfer attention，PyTorch sampling。上下文上限 8192，最多 32 个运行请求，microbatch 与 CUDA graph batch 上限均为 16。ALP whole-prefill 的 chunk 为 0；最终 CPP 采用 pilot 选定的 2048-token chunk，相关消融由各自配置控制。完整启动参数、配置、环境变量及五个运行时文件的 SHA256 见 [最终 ALP launch](../run/a100/v2-main-alp-whole/launch.json)、[最终 CPP launch](../run/a100/v2-main-cpp/launch.json)，硬件见 [hardware.json](../run/a100/hardware.json)。

基线代码提交为 `638aa79412593f10d8008b117516e4f98e661e0e`，实验包含本地修改，不能只用提交号标识实际运行版本。依赖包括 PyTorch `2.8.0+cu128`、Transformers `4.56.1`、FlashInfer `0.4.0rc1`；见 [依赖快照](../run/a100/requirements-freeze.txt)、[运行时补丁](../run/a100/runtime-source.patch) 和各次 `launch.json` 的源码哈希。本文描述的是这些版本的实际行为，不外推到其他 FlashInfer 版本。

[experiment_client.py](../scripts/a100/experiment_client.py) 使用 tokenizer 一次性生成并保存固定 `input_ids`，之后各服务接收同一批 token，不重新编码文本。[fixtures.json](../run/a100/fixtures.json) 的 SHA256 为 `29e25458619c730fac62b63542778f0acac2da13a649902ae3cfd9dcc2eec33c`；其中保存 tokenizer/config 文件指纹、Transformers 版本、构造 seed=42、各输入哈希。服务 seed 为 20261114，客户端保存实际顺序、seed 和请求 ID。并发线程和调度时序仍可能变化，固定 seed 不等于确定性执行。

| 套件 | 请求数 | 覆盖内容 |
| --- | ---: | --- |
| serial | 24 | 自然指令；255/256/257、511/512/513、1023/1024/1025、2047/2048/2049 输入长度边界；4096/6144 长输入 |
| concurrent | 每轮 32 | 混合 64 至 6144 输入 token，32/64/128/256 输出 token；早期验证两轮，最终每个主配置一轮 |
| cache | 5 | 2048-token base、重复 base、共享 1792-token 前缀的 branch、重复 branch、再次 base |
| perf | 每轮 64 | 混合输入/输出长度性能负载；最终 HF 只评分下文指定六个来源的 perf-0 |
| warmup | 24 | 不同输入的串行预热；用于模型与在线预测器预热，不计作正式性能试验 |

这些是程序内构造并裁剪的工程负载，包含重复材料；不是公开质量评测数据集，也不评判回答的事实正确性、推理能力或安全性。本地模型目录名不能证明模型仓库的远端 revision；复现还需核对本地权重 `SHA256SUMS`。本快照不覆盖全部 8192 长度边界、其他模型/精度、随机采样、约束解码或所有并发到达模式。TinyLlama 预检仅用于启动与 API 连通性，未混入下面的 8B 数值结果。

## 2. 协议与客户端数据完整性

所有生成使用原生 `/generate` SSE，`temperature=0`、`ignore_eos=true`，要求生成指定数量的 token。客户端检查状态、结束标记、finish reason、累计 completion count、实际输出长度和有限时间指标；异常请求保留原始错误并使进程返回非零。`outcome.json` 或客户端 `passed=true` 表示请求及协议检查完成，**不代表与基线的 token 序列一致**。

初期预检发现原生流首帧 `output_ids` 包含用于增量 detokenization 的 prompt 尾部：首帧 6 个 ID 实际是 5 个 prompt lookback 加 1 个生成 token，因此 32-token 输出原始累计得到 37 个 ID。来源为 [Req.init_incremental_detokenize](../python/sglang/srt/managers/schedule_batch.py)、[scheduler 输出切片](../python/sglang/srt/managers/scheduler_output_processor_mixin.py) 与 [detokenizer 转发 decode_ids](../python/sglang/srt/managers/detokenizer_manager.py)。

客户端以 `meta_info.completion_tokens` 的累计差分确定本帧生成数量；**只有首帧多余 ID 与实际输入后缀完全一致时才移除**，后续帧 ID 数必须与差分相等。保存 `raw_output_ids`、移除的 lookback ID/数量和每帧时间/数量；比较只使用规范化的生成 `output_ids`。这不是静默截断，也不通过文字相似度修补输出。首帧 6→1、后续增量、错误前缀及漏 token 的回归覆盖见 [客户端测试](../scripts/a100/test_experiment_client.py)。

另一个客户端问题是跨次调用重复 `rid`：detokenizer 的有限容量状态表仍可能保留已结束请求，影响后续文本拼接。现在每次客户端调用生成一个独立 UUID namespace，同次调用所有请求共享前缀，后缀仍含 suite/repeat/index/fixture；重复缓存请求也各有独立 ID。报告保存 namespace 与客户端源码哈希，跨次比较按 fixture 及出现次数配对。**该修正不解释原先的大 regret**：原始输出 ID 的独立评分已经发现异常，文本状态与生成 token ID 路径分别审计，见 [hf-outlier-analysis.json](../run/a100/hf-outlier-analysis.json)。

TTFT 使用首个生成 token 事件；TPOT 为首末生成事件时间差除以实际输出数量减一。原始事件保留每帧 token 数，多个 token 同帧到达时不会把帧间隔称为逐 token 延迟。arrival 指标包含客户端排队；这些计时约定不参与 token 正确性判定。

### 后续发现：consumer 合并响应时丢失增量 ID

主试验 v1 的 [main-no-calibration/concurrent-0.json](../run/a100/main-no-calibration/concurrent-0.json) 中，`concurrent-03` 首个收到的原始 ID 只有 `[19791]`，而 `completion_tokens=2`。客户端以 `output_ids length 1 != completion delta 2` 明确拒绝该响应；另外 31 个请求完成，失败运行与 [outcome.json](../run/a100/main-no-calibration/outcome.json) 原样保留。这里不能通过去除 lookback 或放宽长度校验补救，缺失的是尚未交付的增量。

源码路径是 [TokenizerManager._handle_batch_output / _wait_one_response](../python/sglang/srt/managers/tokenizer_manager.py)：producer 每次收到输出就推进 `last_output_offset`，并将独立的 ID 增量追加到 `state.out_list`；consumer 则只选择 `out_list[-1]` 并清空整个队列。如果两个 producer 回调先于 consumer 恢复执行，首帧的 5 个 prompt lookback ID 和首个生成 token 会被丢弃，最新帧仍带累计 completion count。累计 text/logprob 的最新快照可保留全部内容，因此只检查文本无法发现这个问题。[HTTP SSE handler](../python/sglang/srt/entrypoints/http_server.py) 仅序列化 consumer 给出的字典，不会重建丢失 ID。

修复在 consumer 合并响应时拼接所有 pending 增量 ID，同时保留最后一帧的 text、meta 与 finish reason。条件仅覆盖原本输出增量 ID 的两种模式：文本流，以及 `stream_output=true` 的原始 token 流。累计原始 token 模式、非流式响应和不带 ID 的 abort 保留原有行为；只存在一帧时也保留原对象。客户端及其严格 count 校验没有修改。

[test_tokenizer_stream_coalescing.py](../test/srt/test_tokenizer_stream_coalescing.py) 从实际源码提取方法和 dataclass，在 CPU 上构造两个 producer 回调先于 consumer 的时序。修复前 4 项失败，包括预期 7 个 ID、实际仅 `[19791]` 的忠实复现；修复后 **8/8** 通过，覆盖首帧 lookback、中途与结束帧、每个增量只交付一次、累计/非流式模式及 abort。[修复前日志](../run/a100/stream-coalescing-red.log)、[修复后日志](../run/a100/stream-coalescing-green.log) 和 [源码 SHA256/修复来源](../run/a100/streaming-fix-provenance.json) 可复查。这些是 CPU 方法级回归，不能替代真实模型数值验证。

为统一 HTTP 运行时版本，九个主配置按 [main-protocol.json](../run/a100/main-protocol.json) 整组重跑为 `v2-main-*`，新 launch 额外记录 `tokenizer_manager.py` 的第五个运行时源码哈希。v1 的成功与失败原始文件均保留，旧协议存为 [main-protocol-v1.json](../run/a100/main-protocol-v1.json)；性能汇总按当前 `main_runs` 白名单选择，旧主试验标记 `superseded`，不混入最终性能表。这是整组版本重跑，不按测量结果挑选记录。所有十四个 GPU 实验完成并清理后，最终独立 HF 四组评分依次执行，结果见第六节。

## 3. 严格比较与两个数值 oracle

[compare_outputs.py](../scripts/a100/compare_outputs.py) 按 fixture 与出现次数比较完整生成 ID 序列，记录首个分歧位置、双方 token、缺失/额外请求。任何 token 不同都保留为严格不匹配。普通 SGLang 并发自身重复也可能不同，因此另外诊断每个候选 token 在其**实际已生成前缀**下的概率，而不是把后续分歧一律归因于首个不同 token。

设输入为 `x`，候选输出为 `y`。第 `t` 个 token 的诊断量为：

```text
regret_t = max_v log p(v | x, y[:t]) - log p(y[t] | x, y[:t])
```

这里使用自然对数；这是相对 oracle 的数值诊断量。**没有将 0.25 或任何其他阈值定义为“通过”，没有修改严格比较结果。**

- **独立 HF oracle**：[hf_check.py](../scripts/a100/hf_check.py) 使用 Transformers Llama、SDPA、BF16、`use_cache=False`，输入 `prompt + candidate_output`，取 `logits_to_keep=N+1` 后去掉最后一项，对齐为位置 `P-1 … P+N-2` 的预测。随机两层 Llama 的 CPU 对齐检查与逐个实际前缀单独 forward 完全相同，最大差 0；证据在 [对齐分析](../run/a100/hf-outlier-analysis.json)。这检查的是候选轨迹上每个 token，不是另一条自由生成轨迹的文本相似性。
- **普通 SGLang HTTP oracle**：[diagnose_differences.py](../scripts/a100/diagnose_differences.py) 在未启用 PDMux 的服务上提交相同 `prompt + candidate_output`，设置 `return_logprob=true`、`logprob_start_len=P-1`、`top_logprobs_num=5`、`max_new_tokens=1`、`stream=false`。逐项验证最初的 `None` anchor 对应 prompt 最后一个 token，以及每个后续 logprob ID 对应候选 token。它独立于 PDMux 调度路径，但仍使用 SGLang，不能替代独立 HF。当前 HTTP 诊断覆盖选中的严格不匹配请求，并非整套请求的 HF 全覆盖。

旧 HF 报告未写 scorer version，汇总时标为 `legacy-topk-v1`：greedy equality 使用 `topk` 第一项，精确并列最大值的顺序不确定。修复后报告使用 `hf-actual-prefix-v2`，保存 scorer 源码哈希；canonical greedy equality 使用 `torch.argmax`，另报 `hf_max_probability_matches` 表示候选分数与最大分数精确相等。并列值可以属于最大概率集合但不是 canonical argmax。该变更仅修正报告语义，未改变前缀对齐、regret 公式或旧 JSON；[CPU tie 测试](../scripts/a100/test_hf_check.py) 覆盖两者区别。HTTP top-5 第一项也有并列排序限制，其 equality 不与 v2 canonical 统计混合。

## 4. 修复前证据与根因

[hf-concurrent.json](../run/a100/hf-concurrent.json) 对下列 192 个请求的全部生成 token 完成独立 HF 评分，无遗漏检查。普通基线的两轮并发最大 regret 约为 0.25，而 PDMux 存在明显更大的偏离：

| 修复前来源 | 完整请求检查数 | 最大 HF regret |
| --- | ---: | ---: |
| 普通 SGLang baseline，两轮 | 64 | 0.2500001192 |
| static，一轮 | 32 | 11.5 |
| ALP，一轮 | 32 | 13.5 |
| CPP，两轮 | 64 | 22.0 |

例如 static 的 `concurrent-05` 在生成位置 2 选择 ID 1473，HF 首选 ID 315，regret 11.5；ALP `concurrent-04` 位置 5 为 13.5；CPP `concurrent-21` 位置 6 为 22.0。位置从 0 开始。192 个请求的首个生成 token regret 均为 0，异常可以在后续 decode 出现；这只描述首 token 观测，不证明整个 prefill 实现正确。

普通 HTTP oracle 复核了 baseline/static/ALP/CPP 各自全部 14/17/21/21 个严格不匹配请求，其最大 regret 分别约为 0.2500、11.3750、13.4375、21.8125；原始逐 token 概率及覆盖计数在 [diagnostic-oracle](../run/a100/diagnostic-oracle/)。它与 HF 一起排除了仅靠文本展示或 `topk` 并列排序解释这些大偏离的可能。

[hf-diagnostic.json](../run/a100/hf-diagnostic.json) 每种消融检查两轮、共 64 个并发请求：关闭 async decode 后最大 regret 为 0.25；仅关闭 prefetch 仍为 21.875；同时关闭两者为 0.125。这些观测指向异步 decode 路径，不能单独证明具体根因，也不作为性能对照结论。

随后检查已安装 FlashInfer 源码发现：wrapper 持有持久 pinned CPU integer workspace；planner 先 `std::copy` 写入该 host buffer，再在当前 CUDA stream 上排入 `cudaMemcpyAsync`，随后返回。若上一轮 H2D 尚未读取完该 host buffer，下一轮 CPU plan 已覆盖它，前一轮复制就可能读到后一轮计划。CUDA stream 上设备任务的先后关系不能约束 CPU 对源缓冲区的提前写入。第三方源码位置、行号和 SHA256 保存在 [flashinfer-race-source.json](../run/a100/flashinfer-race-source.json)，未修改第三方包源码。

[diagnostic-plan-reuse/server.log](../run/a100/diagnostic-plan-reuse/server.log) 在 PP1 直接记录到前一计划的 copy event 尚未完成、下一计划已变更，例如 `previous_indptr=(0, 258)` → `next_indptr=(0, 130)`。这提供了运行时复用条件与源码缓冲区生命周期问题相互对应的证据；仅增加观测日志没有修复问题。

## 5. Guard 修复与早期 ALP 验证

[flashinfer_backend.py](../python/sglang/srt/layers/attention/flashinfer_backend.py) 的 `_checked_decode_plan` 为 **PDMux 且 PP>1 的 captured decode wrapper** 增加 guard：每次 planner 返回后立即记录 CUDA event；下一次重用同一 wrapper 时，若前次 event 未完成，先等待，再允许 planner 写 host buffer。event 位于后续 graph/kernel launch 之前，等待边界是前一计划复制，不是后续整段 graph。普通 SGLang 和 eager 路径不启用这个 guard。诊断开关只决定是否记录 indptr signature，修复不依赖该开关。

[test_flashinfer_plan_reuse.py](../test/srt/test_flashinfer_plan_reuse.py) 用可控事件和 host buffer 模拟未完成复制，验证旧内容先被读取、再执行下一次 host 写入；已完成复制不额外等待，诊断开关开/关均覆盖。完整相关回归为 **121 passed、2 warnings**，命令与范围见 [postfix-regression-metadata.json](../run/a100/postfix-regression-metadata.json)。该文件是执行工具输出的摘要，非原始 stdout；范围是 CPU/mock/Gloo，未执行实际 GPU 测试。旧 [regression-tests.log](../run/a100/regression-tests.log) 的 115 passed 属于修复前记录，不能混作 121 项证据。

最初的修复后验证为 `fixed-alp-correctness`：**ALP whole-prefill，async decode 与 prefetch 均开启**。以下保留历史严格比较，不与后来的 v2 重跑混合：

| 比较 | 完整序列完全一致 | 严格结果 | 原始证据 |
| --- | ---: | --- | --- |
| 普通 baseline 串行重复 | 24/24 | 通过 | [serial-self-comparison.json](../run/a100/baseline-correctness/serial-self-comparison.json) |
| 普通 baseline 并发重复 | 18/32 | 不通过，14 个不匹配 | [concurrent-self-comparison.json](../run/a100/baseline-correctness/concurrent-self-comparison.json) |
| 修复后 ALP serial-0 对 baseline serial-0 | 24/24 | 通过 | [serial-vs-baseline.json](../run/a100/fixed-alp-correctness/serial-vs-baseline.json) |
| 修复后 ALP cache-0 对 baseline cache-0 | 5/5 | 通过 | [cache-vs-baseline.json](../run/a100/fixed-alp-correctness/cache-vs-baseline.json) |
| 修复后 ALP concurrent-0 对 baseline concurrent-0 | 16/32 | 不通过，16 个不匹配 | [concurrent-vs-baseline.json](../run/a100/fixed-alp-correctness/concurrent-vs-baseline.json) |

缓存套件的实际 `cached_tokens` 为 `[0, 2047, 1792, 2047, 2047]`。page size 为 1 时，重复输入要求至少复用 `input_tokens-1`，branch 要求至少复用 1792 的共同前缀，同时重复输出必须完全一致；见 [cache-0.json](../run/a100/fixed-alp-correctness/cache-0.json)。

[hf-fixed-alp.json](../run/a100/hf-fixed-alp.json) 对修复后两轮并发的 **64/64 请求、8448/8448 生成 token** 完成实际前缀评分，未缺失或失败，最大 regret 为 **0.25**。v2 canonical argmax 相同的 token 为 **8389/8448**，属于精确最大概率集合的 token 为 **8418/8448**。因此即使数值偏离已回到本次普通基线观测到的范围，仍存在相对 HF 的非最大概率选择，也仍存在严格序列不匹配；不能写成“HF 完全一致”或“正确性全部通过”。该 64 项 HF 全覆盖仅指两轮 concurrent，不自动扩展到 serial/cache、其他变体或无限请求空间。

## 6. 最终 v2 严格比较与独立 HF 结果

[最终实验状态](../run/a100/full-jobs.status.json) 记录十四个作业退出码均为 0；[CPU 数据审计](../run/a100/main-client-audit.json) 覆盖 87 次客户端调用、3669 次请求、494720 个实际生成 token。输出长度、原始 ID/规范化 ID、SSE 累计计数、时间与排队恒等式、唯一请求 ID，以及两阶段缓存清理日志均完成检查。九个主配置各自的 serial/cache/concurrent 共 **549 次请求**均具备完整记录；这些完整性检查不等于模型输出相同。

### 严格序列比较

以下每格是与 `v2-main-baseline-whole` 同套件的完整生成 ID 序列完全一致数。该基准本身不做恒等自比较，标为“基准”；“—”表示未进行该项比较。分子小于分母的项继续判为严格不匹配，未因后续数值诊断改成通过。

| 配置 | serial | cache（5 次请求） | concurrent | perf-0 |
| --- | ---: | ---: | ---: | ---: |
| [v2-main-baseline-whole](../run/a100/v2-main-baseline-whole/) | 基准 | 基准 | 基准 | 基准 |
| [v2-main-static](../run/a100/v2-main-static/) | 24/24 | 5/5 | 15/32 | — |
| [v2-main-no-prefetch](../run/a100/v2-main-no-prefetch/) | 23/24 | 5/5 | 12/32 | — |
| [v2-main-cpu-metadata](../run/a100/v2-main-cpu-metadata/) | 23/24 | 5/5 | 15/32 | 29/64 |
| [v2-main-no-calibration](../run/a100/v2-main-no-calibration/) | 23/24 | 5/5 | 12/32 | — |
| [v2-main-no-async](../run/a100/v2-main-no-async/) | 23/24 | 5/5 | 18/32 | — |
| [v2-main-cpp](../run/a100/v2-main-cpp/) | 23/24 | 5/5 | 15/32 | 25/64 |
| [v2-main-alp-whole](../run/a100/v2-main-alp-whole/) | 24/24 | 5/5 | 15/32 | — |
| [v2-main-baseline-chunk2048](../run/a100/v2-main-baseline-chunk2048/) | 23/24 | 5/5 | 14/32 | — |
| [mechanism-credit-refill0](../run/a100/mechanism-credit-refill0/) | — | — | — | 20/64 |
| [mechanism-credit-refill1024](../run/a100/mechanism-credit-refill1024/) | — | — | — | 23/64 |
| [mechanism-migration](../run/a100/mechanism-migration/) | — | — | — | 25/64 |

每个目录中的 `serial-vs-main-baseline.json`、`cache-vs-main-baseline.json`、`concurrent-vs-main-baseline.json` 或 `perf-vs-main-baseline.json` 保存具体分歧；汇总见 [exact-comparisons.csv](../run/a100/correctness-data/exact-comparisons.csv)。八个主候选合计 serial **186/192**、cache **40/40**、concurrent **116/256** 完全一致。六个 serial 23/24 的配置均在 `long-4096` 的位置 30 首次分歧；最终 HF 已覆盖这六条候选的全部续写，而非仅检查首个分歧点。

### 四组 HF 覆盖和数值结果

四组均为 `hf-actual-prefix-v2`，使用同一评分源码 SHA256；[final-hf-inputs.json](../run/a100/final-hf-inputs.json) 保存精确输入清单，[final-hf-status.json](../run/a100/final-hf-status.json) 保存四个评分子任务的退出码 0 和完成计数。下面的检查数是跨配置的候选轨迹观测数，同一 fixture 在不同配置中会重复出现，不代表 906 条独立语料。

执行记录补充：外层 driver 工具会话返回 143，原因未知；四个 HF 子进程均退出 0 且标记完成，906/110016 覆盖与源文件 SHA256 已独立核验。两层状态分别记录于 [final-hf-process-audit.json](../run/a100/final-hf-process-audit.json)，未重跑或改写原始评分。

| HF 组 | 轨迹检查 | 生成 token 评分 | 最大 regret | canonical argmax 相同 token | 精确最大概率集合 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| [main-concurrent](../run/a100/hf-final-main-concurrent.json) | 288 | 38016 | 0.250000119 | 37766/38016 | 37894/38016 |
| [mechanism-perf](../run/a100/hf-final-mechanism-perf.json) | 384 | 59904 | 0.5 | 59413/59904 | 59645/59904 |
| [main-serial](../run/a100/hf-final-main-serial.json) | 216 | 10944 | 0.12500006 | 10860/10944 | 10890/10944 |
| [main-cache](../run/a100/hf-final-main-cache.json) | 18 | 1152 | 0 | 1143/1152 | 1152/1152 |
| **合计** | **906/906** | **110016/110016** | **0.5** | **109182/110016** | **109581/110016** |

分配置的最大 regret 如下；每个非空格覆盖对应来源的完整候选轨迹。serial 每格 24 条、concurrent 每格 32 条、perf-0 每格 64 条；cache 每格是 2 个唯一 fixture。小数在表中有限位数显示，原始 JSON 保留完整精度。

| 配置 | serial max | cache max | concurrent max | perf-0 max |
| --- | ---: | ---: | ---: | ---: |
| [v2-main-baseline-whole](../run/a100/v2-main-baseline-whole/) | 0.12500006 | 0 | 0.250000119 | 0.25000006 |
| [v2-main-static](../run/a100/v2-main-static/) | 0.12500006 | 0 | 0.25 | — |
| [v2-main-no-prefetch](../run/a100/v2-main-no-prefetch/) | 0.12500006 | 0 | 0.25 | — |
| [v2-main-cpu-metadata](../run/a100/v2-main-cpu-metadata/) | 0.12500006 | 0 | 0.25 | 0.25 |
| [v2-main-no-calibration](../run/a100/v2-main-no-calibration/) | 0.12500006 | 0 | 0.25 | — |
| [v2-main-no-async](../run/a100/v2-main-no-async/) | 0.12500006 | 0 | 0.125 | — |
| [v2-main-cpp](../run/a100/v2-main-cpp/) | 0.12500006 | 0 | 0.125 | 0.25 |
| [v2-main-alp-whole](../run/a100/v2-main-alp-whole/) | 0.12500006 | 0 | 0.25 | — |
| [v2-main-baseline-chunk2048](../run/a100/v2-main-baseline-chunk2048/) | 0.12500006 | 0 | 0.125 | — |
| [mechanism-credit-refill0](../run/a100/mechanism-credit-refill0/) | — | — | — | 0.25 |
| [mechanism-credit-refill1024](../run/a100/mechanism-credit-refill1024/) | — | — | — | 0.5 |
| [mechanism-migration](../run/a100/mechanism-migration/) | — | — | — | 0.25 |

这些结果仍包含 **435 个相对 HF 非最大概率 token**，canonical argmax 不同的 token 共 **834 个**。cache 的 1152 个 token 全部属于 HF 的精确最大概率集合，但其中 9 个与 canonical argmax 不同，体现并列最大值的语义区别；不能将这两种 equality 混为一谈。

全局最大值 **0.5** 位于 `mechanism-credit-refill1024/perf-0.json` 的 `perf-35`（输入 256、输出 256），位置 213：候选 ID 14832，HF canonical argmax ID 34174；候选 logprob 为 `-2.162339210510254`，最大 logprob 为 `-1.662339210510254`，top-two margin 为 0.5。该请求平均 regret 为 `0.001953125`。它高于本轮普通基线的最大值约 0.25，必须单列；没有把它自动归为可接受误差，也没有据此改写严格比较。

追加的普通 PP2 HTTP oracle 已完成，服务正常退出并清理。它复核了该配置 **41 个严格不匹配样本中的前 32 个**，包含 `perf-35`，共评分 5568 个 token；另有 9 个未选，报告明确 `completed=true`、`full_mismatch_coverage=false`、无错误，不能称作全部不匹配覆盖。在同一实际前缀和位置 213，普通 oracle 仍以 ID 34174 为最优（logprob `-1.6201995611190796`），而候选 ID 14832 的 logprob 为 `-2.120199680328369`；regret 与 top-two margin 均为 **0.5000001192092896**。[原始追加报告](../run/a100/final-refill-oracle/oracle-refill1024-outlier.json) 保留逐 token 概率和选样范围。

两个 oracle 均证实该候选在此实际前缀上是非最大概率选择；不能仅以 HF 实现差异或精确并列解释，**原因尚未确定**。原始 HF 结果及严格失败保持不变，不通过阈值将其改判。追加的 32 项普通 oracle 诊断也不计入 906 项独立 HF 检查。

完整逐来源统计与引用见 [final-correctness-validation.json](../run/a100/final-correctness-validation.json) 和 [numerical-reports.csv](../run/a100/correctness-data/numerical-reports.csv)。最终回归记录为 [129 项运行时测试](../run/a100/final-runtime-tests.log) 与 [78 项工具测试、29 个 subtest](../run/a100/final-tool-tests.log)；这些 CPU/mock/Gloo 与工具回归和真实模型 HF 评分分别报告。

## 7. 覆盖边界与汇总方法

- **Cache 去重边界**：HF 每个主配置仅评分 `cache-base`、`cache-branch` 的首次出现，共 **18 个唯一 fixture/source 对**。源文件实际包含 **45 次请求**；其余重复出现由 CPU 缓存重复输出完全一致及复用下界断言覆盖。HF 汇总明确 `full_source_unique_fixture_coverage=true`、`full_source_request_occurrence_coverage=false`，不声称 45 次请求都单独接受 HF 评分。
- **Perf 与机制边界**：HF perf 组只包含普通 whole 基线、CPP、CPU metadata、credit refill 0、credit refill 1024、migration 六个来源的 `perf-0`。没有评分其余 perf 重复、open-loop 性能试验或所有主配置的 perf。`mechanism-slo10`、`mechanism-slo20` 未纳入 HF；它们的客户端完成状态不是数值正确性证据。
- **推断边界**：最终观测中未再出现早期 11.5–22 的大 regret，但较小的非最大概率选择与严格 token 不匹配仍在。没有预设或事后引入 regret 通过阈值，也不把运行完成、数值检查覆盖完整与输出完全相同混为一谈。

[summarize_correctness.py](../scripts/a100/summarize_correctness.py) 使用显式 [correctness-phase-map.json](../run/a100/correctness-phase-map.json)，将原始未修复、普通参考、无 guard 的观测/消融、guard 后早期正确性、pilot、trace、废弃 v1、最终 v2 与机制试验分开；[分类说明及源码来源](../run/a100/correctness-phase-map-metadata.json) 保存其含义。未声明的 run 保持 unclassified，历史 `legacy-topk-v1` 和最终 canonical scorer 分组，重复评分保留为重复观测。最终输出保留原始文件及哈希、检查范围、未完成/错误记录；数值结果不反向修改严格比较 verdict。

```bash
.venv/bin/python scripts/a100/summarize_correctness.py \
  --root run/a100 --output-dir run/a100/correctness-data \
  --phase-map run/a100/correctness-phase-map.json
```

修复引入的 planner host 等待及 SSE 帧合并对性能的影响，由独立性能试验报告评估。本文的正确性、消融、oracle 或诊断数据不支持性能提升结论；其他模型、精度、长上下文边界和未执行负载仍需另行验证。

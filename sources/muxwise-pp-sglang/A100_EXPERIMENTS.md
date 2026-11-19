# A100 双卡 PP2 校准与机制实验

日期：2026-11-14（北京时间；原始日志为 UTC）。已完成 A100 校准、九组主对照、五组定向机制实验及独立输出核对。**严格 token 一致性未全部通过**；已修复两个实际竞态/流式问题，仍保留并发不匹配和一个经独立复核的非贪心偏离，不能将本报告作为严格正确性验收通过证明。

## 环境与范围

- 仓库：`Irisuko/muxwise-pp-sglang`，起点 `638aa79412593f10d8008b117516e4f98e661e0e`。
- 模型：Llama-3.1-8B-Instruct，revision `0e9e39f249a16976918f6564b8830bc894c89659`；14 项 SHA256、4 个 safetensors 文件、291 个张量已验证。见 [权重验证](run/a100/model-verification.json)。
- 只使用物理 GPU 6、7：A100-SXM4-80GB，各 108 SM，NVLink；TP1、PP2、BF16。每次启动检查显存和计算进程；每次实验结束释放服务进程组。
- Python 3.10.22、Torch 2.8.0+cu128、FlashInfer 0.4.0rc1、SGLang 0.5.3rc0、sgl-kernel 0.3.12。使用独立 CUDA 12.8 工具链，不修改系统 CUDA。见 [工具链来源](scripts/a100/toolkit-provenance.json)、[硬件与实际 SM 分配](run/a100/hardware.json)。
- FlashInfer attention、PyTorch greedy sampling；context 8192、max-running-requests 32、decode microbatch / graph max batch 16、KV capacity 135168。KV 容量大于 `16 * 8192 + 1`，避免校准映射回绕。

本轮实验及补充 oracle 服务均已停止。收尾检查时 GPU 6、7 均无计算进程、显存占用 0 MiB，端口 30000/30100 未监听；见 [最终核验记录](run/a100/final-verification.json)。

## 校准

实测六个混合 SM 分区：`100/8、92/16、84/24、76/32、60/48、52/56`（prefill/decode），加两个端点 `108/0、0/108`。YAML 与启动参数均设置 8 组。

两阶段各测量 7 个 decode 分区 × batch `1/4/8/16` × context `128/512/1024/2048/4096/8192`。三次独立启动共 1008 个有效点；每个点取均匀与偏斜长度形状中较慢者，每种形状由 3 次 graph replay 计时。冻结选择依据见 [校准结果](run/a100/calibration-selection.json)、[完整 CSV](run/a100/calibration-selection.csv) 和 [安全判据](run/a100/calibration-selection-safety.csv)。

主实验目标固定为 TTFT 2 秒、每请求平均 TPOT 50 ms。按运行时代码的初始判据 `PP_size * (1.2 * decode_table_envelope + 1 ms)`，取三次测量及两个阶段的最坏值：

| decode SM | batch / context | 最坏初始预测 | 50 ms 判据 |
|---|---|---:|---|
| 8 | 1 / 8192 | 93.857 ms | 不满足 |
| 16 | 4 / 8192 | 44.807 ms | 满足 |
| 16 | 8 / 8192 | 51.733 ms | 不满足 |
| 24 | 16 / 8192 | 47.056 ms | 满足 |

[A100 配置](configs/pdmux_pp_a100.yml) 的静态回退阈值为 `[1,1,5,17,17,17]`：batch 1–4 使用 `92/16`，batch 5–16 使用 `84/24`。重复阈值用于跳过不适合本工况的静态分区；全部候选仍保留供 ALP 选择。这是 decode 校准判据，不能保证包含排队、通信和并发干扰的端到端 TPOT。

默认 ALP 的 `latency_partition=true` 在满足 decode 判据的分区中优先选择最多 prefill SM；在线 RLS 仍用于 admission。不能把这一模式解释为“按预测到达吞吐量选择分区”。

![A100 decode 校准](run/a100/report-data/plots/a100-calibration-frozen.png)

## 正确性与修复

详细过程见 [正确性报告](docs/a100-correctness.md)。验证包括自然提示、12 个 chunk 边界长度、4096/6144 长输入、重复/分支前缀缓存、32 并发和重复运行。原始输入 IDs、原始 SSE IDs、规范化生成 IDs、完整长度及结束状态均保留。

发现并修复了异步 decode 的真实竞态：FlashInfer planner 重复使用 pinned host 规划缓冲区时，上次 H2D copy 可能尚未完成。观测到 27 个复用窗口，全部位于 PP1，并伴随不同微批的规划数据。仅关闭 prefetch 不解决问题；关闭 async decode 后大概率偏差消失。独立 HF 和普通 PP2 的 teacher-forcing 检查相互印证。

修复仅在 PDMux PP>1 的 captured decode wrapper 中保护该 host buffer：下一次 planner 写入前等待上一次规划拷贝的 CUDA event，随后立即记录新的 copy event；不等待后续整个 decode graph。普通 PP2 路径保持原调用。证据见 [复用窗口](run/a100/plan-reuse-evidence.json)、[依赖源码与 SHA](run/a100/flashinfer-race-source.json)、[修复来源](run/a100/fix-provenance.json)、[回归记录](run/a100/postfix-regression-metadata.json)。

修复后 ALP whole 的两轮 32 并发共 64 个样本，HF actual-prefix 最大 regret 为 0.25，原始异步路径曾达到 22.125；普通 PP2 的观测最大值同为 0.25。这里的 regret 是 `最佳 token logprob - 实际 token logprob`。这些数值用于诊断，没有把事后选择的阈值当作通过标准。

严格 token 一致性仍单独报告：普通 PP2 串行自重复 24/24、并发自重复 18/32；修复后 ALP 与普通 PP2 的串行 24/24、缓存 5/5、首轮并发 16/32。并发严格一致性未通过，不能用“请求无错误”替代该结论。统一修复版本的 129 项运行时 CPU/mock/Gloo 回归通过，另有 78 项工具测试和 29 个子测试通过。原始记录见 [运行时回归](run/a100/final-runtime-tests.log)、[工具回归](run/a100/final-tool-tests.log)。

另发现并修复了 tokenizer manager 的流式合并问题：当消费者恢复前已有多帧输出排队，旧代码只返回最新一帧的增量 IDs，却保留累计 completion_tokens。修复会拼接全部待发送增量 IDs，保持最新累计 text/meta，不改变非流式和累计 raw-token 模式。8 个针对性用例先复现、后通过。主矩阵因此整体重跑为 `v2-main-*`；旧结果与失败记录保留，但不进入最终性能表。见 [SSE 修复证据](run/a100/stream-coalescing-fix.json) 和 [v2 协议](run/a100/main-protocol.json)。原 pilot 的 576 个请求、57,344 帧均完整，且四个模型/校准源码哈希不变，因此保留其选参结果。

最终统一版本的严格比较以 `v2-main-baseline-whole` 为参考：

| 候选 | 串行 exact | 缓存 exact | 32 并发 exact |
|---|---:|---:|---:|
| Static whole | 24/24 | 5/5 | 15/32 |
| ALP whole | 24/24 | 5/5 | 15/32 |
| CPP | 23/24 | 5/5 | 15/32 |
| CPP：关闭 prefetch | 23/24 | 5/5 | 12/32 |
| CPP：CPU metadata | 23/24 | 5/5 | 15/32 |
| CPP：关闭启动校准 | 23/24 | 5/5 | 12/32 |
| CPP：关闭 async | 23/24 | 5/5 | 18/32 |
| 普通 chunk2048 | 23/24 | 5/5 | 14/32 |

独立 HF SDPA BF16 的 actual-prefix 评分完成 **906/906 项、110,016/110,016 个生成 token**，无缺失。canonical argmax 相同为 109,182 个，属于精确最大概率集合为 109,581 个；这两个指标不同，均未达到逐 token 全一致。

| HF 覆盖 | 样本数 | token 数 | 最大 regret |
|---|---:|---:|---:|
| 九组主配置全部并发样本 | 288 | 38,016 | 约 0.25 |
| 六个指定来源的第一轮性能样本 | 384 | 59,904 | 0.5 |
| 九组主配置全部串行样本 | 216 | 10,944 | 约 0.125 |
| 九组缓存的两个唯一输入 | 18 | 1,152 | 0 |

缓存评分是每组两个唯一输入的首个 occurrence；五次缓存请求的输出重复性另由 CPU 核验，不能称为 45 次独立 HF 评分。六个性能来源是 ordinary whole、CPP、CPU metadata、credit refill0/refill1024 和 migration；不代表全部性能轮次或 SLO10/20 均接受 HF 评分。见 [最终覆盖审计](run/a100/final-correctness-validation.json)、[明确阶段划分](run/a100/correctness-phase-map.json)、[严格比较审计](run/a100/main-client-audit.json)。

残留最大偏离来自 `mechanism-credit-refill1024/perf-0.json` 的 `perf-35`：生成位置 213 选 ID 14832，而 HF 首选 ID 34174，regret 为 0.5。另启普通 PP2 teacher-forcing 后同一位置仍约为 0.5，首选也为 34174；因此不能仅归因于 HF 实现差异或并列排序。该复核覆盖同一文件前 32/41 个严格不匹配样本，包含此例，范围明确而非全覆盖。原因尚未确定，**不设事后容差将其改判通过**。原始证据见 [普通 PP2 复核](run/a100/final-refill-oracle/oracle-refill1024-outlier.json)。

## 参数扫描与性能协议

8 组候选覆盖 token chunk `256/1024/2048` 与层切片 token-layer budget `2048/8192/32768`。`chunk=256,budget=32768` 与 `budget=8192` 均一次执行全部 16 个本地层，因此预先跳过这组等价配置。每组 24 个固定混合请求、2 轮；按 2 秒 / 50 ms 同时达标率优先、吞吐量次优选择，保留全部结果，不宣称全局最优。

| chunk tokens | token-layer budget | 平均联合达标率 | 输出吞吐中位数 (token/s) |
|---:|---:|---:|---:|
| **2048** | **32768** | **45.83%** | **415.19** |
| 1024 | 8192 | 33.33% | 309.46 |
| 2048 | 8192 | 29.17% | 332.98 |
| 1024 | 32768 | 25.00% | 360.89 |
| 2048 | 2048 | 20.83% | 238.53 |
| 1024 | 2048 | 20.83% | 223.23 |
| 256 | 8192 | 4.17% | 167.72 |
| 256 | 2048 | 4.17% | 145.85 |

选定 `chunk=2048,budget=32768`，见 [冻结选择](run/a100/pilot-selection.json)、[所有 pilot 指标](run/a100/pilot-selection.csv)、[CPP 配置](configs/pdmux_pp_a100_cpp.yml)。这组 pilot 的实际最大并发为 24；后面的 64 请求主实验是另一份固定 workload，不能直接把两表吞吐率比较为加速比。

![Pilot 参数扫描](run/a100/report-data/plots/a100-pilot-selection.png)

主性能矩阵使用独立的新服务实例，每个配置先运行相同的 24 个预热请求，再测量 3 轮相同 64 请求 workload。每轮确认 HTTP 200 的 cache-reset acknowledgement；flush 保留在线模型，因此每个配置采用相同的测量前历史。正确性请求放在性能轮次之后。

饱和模式客户端并发 32；另设固定种子的 8 req/s Poisson 到达、客户端并发 64。开放到达模式使用包含客户端排队的 arrival TTFT/SLO，并检查实际 queue delay。吞吐量是这批请求从发送到排空的输出 token/s；TPOT 是每请求平均值，不等同于逐 token 尾延迟。3 次重复位于同一服务生命周期内，不是 3 次独立部署。

## 主性能结果

以下仅含同一修复版本的九组主实验。吞吐量显示三轮中位数及实际最小–最大值；TTFT/TPOT p95 显示三轮各自 p95 的中位数；达标率显示三轮均值。饱和表的 TTFT 和联合达标率从 HTTP 请求发起计时，不含客户端并发槽等待。

| 配置 | 输出 token/s：中位数 [范围] | TTFT p95 (s) | 平均 TPOT 的 p95 (ms) | 联合达标率 |
|---|---:|---:|---:|---:|
| 静态 PDMux whole | 687.48 [683.39–715.38] | 2.676 | 39.54 | 82.29% |
| 普通 PP2 whole | 654.81 [643.73–710.58] | 1.753 | 45.44 | 95.31% |
| CPP：关闭 prefetch | 611.44 [598.04–636.44] | 4.149 | 50.58 | 67.71% |
| CPP：CPU prefill metadata | 664.78 [661.96–671.83] | 3.302 | 45.08 | 76.56% |
| CPP：关闭启动校准 | 637.46 [614.17–642.15] | 3.627 | 50.46 | 67.19% |
| CPP：关闭 async decode | 553.09 [542.68–561.70] | 3.529 | 55.87 | 67.71% |
| CPP 2048 / 32768 | 677.67 [676.17–688.09] | 3.098 | 45.70 | 76.04% |
| ALP whole | 699.56 [695.09–717.68] | 2.724 | 38.59 | 79.69% |
| 普通 PP2 chunk2048 | 623.69 [606.62–646.70] | 1.748 | 51.81 | 91.15% |

在这份固定负载中，ALP whole 的吞吐中位数最高，比普通 whole PP2 高 6.83%，但联合达标率从 95.31% 降到 79.69%。CPP 相比普通 chunk2048 吞吐高 8.65%，联合达标率却低 15.10 个百分点；相比 ALP whole，CPP 吞吐低 3.13%。因此 pilot 选出的 CPP 参数只是所扫描 CPP 候选中的选择，不能称为所有策略中的最优配置。

相对完整 CPP，关闭 prefetch、关闭 async decode、关闭启动校准均降低了本次观测吞吐；启用 prefetch/async 相对各自关闭版本的中位数差分别为 +10.83% / +22.52%。这些是完整代码路径的单开关消融，不是某个 CUDA kernel 或通信阶段的独立收益。static 与 ALP 同时改变 partition/admission 策略；关闭启动校准的比较是在相同 warmup 后测得，不包含启动耗时。CPU metadata 路径的吞吐中位数低 1.90%，本次没有支持其普遍加速的证据。

![主矩阵吞吐与延迟](run/a100/report-data/final-performance/perf-summary.png)

开放到达测试为固定 64 请求、8 req/s 的有限批次，包含最终排空时间，不是稳态容量测试。下表的 TTFT/联合达标率均包含计划到达后的客户端排队：

| 配置 | 输出 token/s：中位数 [范围] | arrival TTFT p95 (s) | 平均 TPOT 的 p95 (ms) | arrival 联合达标率 | 客户端排队 p95 (ms) |
|---|---:|---:|---:|---:|---:|
| CPP：CPU prefill metadata | 625.99 [621.13–646.50] | 3.032 | 45.91 | 69.79% | 7.20 |
| CPP 2048 / 32768 | 631.01 [622.87–655.75] | 3.032 | 43.81 | 71.88% | 7.62 |
| 普通 PP2 chunk2048 | 585.29 [581.82–586.30] | 2.740 | 46.90 | 72.92% | 7.72 |

开放到达下，CPP 的吞吐中位数比普通 chunk2048 高 7.81%，联合达标率低 1.04 个百分点。三轮范围和有限样本不足以证明统计显著性。在本次饱和负载下，若优先 HTTP 请求开始后的 2 秒 / 50 ms 联合达标，可优先考虑普通 whole PP2；若优先已测配置中的吞吐中位数，可从已校准的 ALP whole 开始，并保留 CPP 对照。普通 whole PP2 和 ALP whole 未参与本次开放到达测试，因此这项选择建议不能直接推广到开放到达或其他负载。

![开放到达性能](run/a100/report-data/final-performance/perf-openloop-summary.png)

完整 p50/p95/p99、端到端延迟、client SSE token-gap、每轮值及输入/源码校验来源见 [主表 CSV](run/a100/report-data/final-performance/perf-summary.csv)、[开放到达 CSV](run/a100/report-data/final-performance/perf-openloop-summary.csv)、[全部表格及准入审计](run/a100/report-data/final-performance/final-performance.json)。SSE token-gap 是客户端观察值，仅在所有请求及生成事件完整、每个含生成 token 的帧恰有一个生成 token 时计算；忽略零 token 元数据帧，不对合批帧插值，也不等同于服务端 kernel 间隔。

## 关键机制验证

九组主对照与五组定向实验全部完成。以下计数默认覆盖整个服务生命周期，包含预热和正确性套件；不同运行的生命周期长度不同，不能将计数直接当成效率或收益。完整来源、日志行号及解释见 [机制提要](run/a100/mechanism-report-notes.md) 和 [机器可读报告](run/a100/mechanism-analysis.json)。

| 机制 | 实际观测 | 结论边界 |
|---|---|---|
| Token chunking | CPP PP0：530 final slices / 316 admitted 父批 | 已触发 token 拆分；差值 214 不是独立请求数 |
| ALP 分区/admission | CPP PP1 使用静态不会选的分区 4/5；已观测预算 6656–8192 tokens | 自适应决策实际发生；RLS 满 16 样本本身只代表模型 ready |
| Prefetch | CPP PP1 的 10 个诊断批次中 5 个提前向通信 worker 提交接收任务；关闭预取为 0/11 | CPU 侧提前提交已触发，不等同于 GPU copy/compute 重叠 |
| 更紧 TPOT 目标 | 10 ms 仅用混合 52/56；20 ms 使用 60/48、52/56 | 分区响应目标；实测 TPOT 达标率为 0% / 33.33%，联合达标均为 0% |
| Token credit / refill | cap8192/refill0：cap blocked/stops=281/295；refill1024：cap=224/236、refill=40/40 | 两类 gate 实际触发；stops 包含部分放行后停止 |
| 请求迁移 | 默认主实验为 0；unit1/interval4 定向实验累计 29 | 已触发迁移；同一请求可多次迁移，不能称为 29 个不同请求 |

主 CPP 的 `chunk=2048,budget=32768` 允许一次完成本地 16 层；它激活 token chunking，并没有产生更多中间层切片。Whole 两组存在固定 budget 的 layer slicing。所有主 Mux 的 decode-aware slice shortening 计数为 0；adaptive-inflight 未启用，均不宣称已验证其收益。10/20 ms 达标率按各自 `experiment_slo` 及原始请求重算，没有套用主表的固定 50 ms 标签。

预取证据来自已有独立 trace，按八个实际请求 ID 配对，其中四次提前提交接收任务发生在此前 logical head 尚未在 CPU 侧报告 `prefill_done` 时。见 [原始事件配对与 SHA](run/a100/prefetch-trace-evidence.json)。

## GPU 执行重叠证据

修复 profiler 的 PP rank 文件名碰撞后，分别采集 CPP 与关闭 prefetch 的两个阶段 trace。诊断时给实际执行线程上的 prefill、decode 等操作添加语义范围，沿 CUDA graph launch correlation 关联 GPU kernel；约 96.1% 的 kernel 可以无歧义归类，未归类 kernel 不进入语义重叠计数。

| 诊断配置 | PP 阶段 | prefill/decode 同时执行 (ms) | 占 prefill kernel 时间并集 |
|---|---:|---:|---:|
| CPP | 0 | 208.897 | 38.92% |
| CPP | 1 | 271.146 | 49.62% |
| 关闭 prefetch | 0 | 215.947 | 40.27% |
| 关闭 prefetch | 1 | 189.881 | 34.83% |

这证明 GPU 上确实发生 prefill/decode 执行重叠，也表明关闭 prefetch 不等于关闭所有重叠。它不测量 SM 占用率，不构成吞吐收益或统计显著性证据。每个诊断仅有 8 个请求，profiler 会引入额外开销；数据全部排除在主性能表之外。见 [逐阶段 CSV](run/a100/gpu-trace-summary.csv)、[CPP 原始关联校验](run/a100/trace-cpp/gpu-trace-validation.json)、[关闭预取的关联校验](run/a100/trace-no-prefetch/gpu-trace-validation.json)。

![GPU kernel 语义时间线](run/a100/gpu-trace-timeline.png)

## 复现与局限

以下命令从仓库根目录执行，使用已经配置好的 `.venv`、`$HOME/.local/cuda-12.8`、当前包含两项正确性修复的源码，以及 `../models/Llama-3.1-8B-Instruct` 本地权重。依赖版本见 [实验环境快照](run/a100/requirements-freeze.txt)；重建环境时还需要 [CUDA 12.8 安装脚本](scripts/a100/install_cuda.py) 和 [spatial 扩展构建脚本](scripts/build_spatial_local.py)。启动时的 `.aria2` 检查只检测未完成下载，完整权重验证记录见前文。

日常前台服务使用 [run_a100.sh](scripts/run_a100.sh)，默认是 **ALP whole、chunk=0**，配置为 `configs/pdmux_pp_a100.yml`。要运行 pilot 选出的 CPP 参数，显式设置 `PDMUX_CHUNK=2048`；目前 `pdmux_pp_a100_cpp.yml` 与该 YAML 的有效参数相同，chunk 大小由启动参数提供：

```bash
cd /data01/home/mhy/workspace/muxwise-pp-sglang
PDMUX_POLICY=alp PDMUX_CHUNK=2048 scripts/run_a100.sh
```

服务默认地址为 `http://127.0.0.1:30000`，API 模型名为 `llama-3.1-8b`；`MODEL_PATH`、`HOST`、`PORT`、`PDMUX_POLICY` 和 `PDMUX_CHUNK` 可通过环境变量指定。脚本固定使用物理 GPU 6、7 和上述 CUDA 12.8 路径，并在两卡已有计算进程时退出。它启用两槽 prefetch、async decode 和启动 decode 校准；首次 FlashInfer JIT 可能较慢。另一个终端可这样调用：

```bash
curl --noproxy '*' --fail-with-body http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.1-8b","messages":[{"role":"user","content":"Say hello in one sentence."}],"temperature":0,"max_tokens":32}'
```

前台服务按 Ctrl-C 停止。正式复现使用下面的实验 runner；它使用端口 `30100`、API 模型名 `a100-llama`、summary 间隔 2 秒（日常脚本为 5 秒），逐任务启动和回收服务进程组，并额外检查两卡各至少有 40000 MiB 空闲显存。等待当前矩阵结束、两卡空闲后再启动复跑。

全部启动命令、配置快照、5 个运行时源码 SHA、原始日志、逐请求 JSON 和关闭记录位于 `run/a100/<run-name>/`。完整 v2 队列包含 9 个主比较与 5 个机制实验，顺序和参数记录在 [full-jobs.json](run/a100/full-jobs.json)。复跑时加唯一前缀，保存到新的 `repro-v2-main-*` / `repro-mechanism-*` 目录；同一前缀已经使用过时应换一个新前缀：

```bash
.venv/bin/python scripts/a100/run_matrix.py \
  --jobs run/a100/full-jobs.json --name-prefix repro-
```

runner 使用现有 [fixtures.json](run/a100/fixtures.json) 中的固定 token IDs；该文件的生成 seed 为 42，逐轮客户端到达 seed 为 `20261114 + repetition`。主配置均运行相同预热、3 轮饱和测量，再运行正确性套件；仅 CPP、CPU-metadata 和普通 chunk2048 三组追加 8 req/s 开放到达测量。任一任务失败时队列停止，详情见对应 `*-driver.log`、`outcome.json` 和 `repro-full-jobs.status.json`。独立复测 8 组 pilot 的命令如下，主队列仍使用协议中已冻结的 `chunk=2048,budget=32768`：

```bash
.venv/bin/python scripts/a100/run_matrix.py \
  --jobs run/a100/pilot-jobs.json --name-prefix repro-
```

原始 v2 结果可直接导出；先检查 `report-data/issues.json` 和机制报告的 `status`，未完成或失败的运行会保留记录并从正式性能表排除：

```bash
.venv/bin/python scripts/a100/summarize_results.py \
  --root run/a100 --output-dir run/a100/report-data
.venv/bin/python scripts/a100/mechanism_report.py \
  --run-root run/a100 --output run/a100/mechanism-analysis.json
.venv/bin/python scripts/a100/final_performance_tables.py
```

HF 评分的四条精确命令及覆盖清单位于 [final-hf-inputs.json](run/a100/final-hf-inputs.json)。对 `repro-` 复跑结果评分时，应在全部矩阵结束、GPU 6/7 空闲后，将每条命令的 `--results` 源目录替换为对应 `repro-` 目录，并为 `--output` 使用新文件名；不要覆盖原始评分。`hf_check.py` 固定使用物理 GPU 6（也可显式 `--gpu 7`），独立逐 token 检查 actual prefix。完整性退出成功不代表 token 一致性通过。最后的单个 0.5 偏离使用额外普通 PP2 服务复核，命令与范围保存在 [oracle outcome](run/a100/final-refill-oracle/outcome.json)。

带 `repro-` 前缀的运行不会自动加入原 `main-protocol.json` 白名单。复跑结束后，建立独立分析目录，将协议里的名字同步加前缀并链接复跑产物；这样可以直接使用相同分析工具，保留原始 v2 协议和结果：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("run/a100").resolve()
prefix = "repro-"
view = root / "repro-view"
view.mkdir(exist_ok=False)
protocol = json.loads((root / "main-protocol.json").read_text())
protocol["main_runs"] = [prefix + name for name in protocol["main_runs"]]
protocol["controlled_comparisons"] = {
    key: [prefix + name for name in pair] if pair else pair
    for key, pair in protocol["controlled_comparisons"].items()
}
(view / "main-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
jobs = json.loads((root / "mechanism-jobs.json").read_text())
for job in jobs:
    index = job["args"].index("--name") + 1
    job["args"][index] = prefix + job["args"][index]
(view / "mechanism-jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
names = protocol["main_runs"] + [job["args"][job["args"].index("--name") + 1] for job in jobs]
for name in names:
    if (root / name).is_dir():
        (view / name).symlink_to(root / name, target_is_directory=True)
PY
.venv/bin/python scripts/a100/summarize_results.py \
  --root run/a100/repro-view --output-dir run/a100/repro-view/report-data
.venv/bin/python scripts/a100/mechanism_report.py \
  --run-root run/a100/repro-view --output run/a100/repro-view/mechanism-analysis.json
```

本实验只覆盖这台共享服务器上的两张 A100、该 8B 模型和合成混合请求集。不能据此宣称跨硬件加速、PP4 扩展性或生产 SLO 保证。仓库启动时提示 Torch >2.6 可能使 PDMux 性能下降；本轮未进行 Torch 版本消融，不能量化或归因这一影响。共享主机的 CPU/内存干扰、有限重复次数也限制结论。启动计数 `overlap_launches` 只说明调度器提交了重叠工作；真实 kernel 并发须由单独的 GPU trace 证明，trace 的时间不进入性能表。

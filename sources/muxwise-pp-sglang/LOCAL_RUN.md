# 当前 A100 服务器运行说明

本机有 8 张 NVIDIA A100-SXM4-80GB，每卡 108 个 SM、compute capability 8.0，GPU 之间通过 NVLink 连接。下面的默认服务只使用物理 GPU **6、7**，TP=1、PP=2；启动前确认两卡可用。

## 模型与环境

- 模型：`../models/Llama-3.1-8B-Instruct`，相对于仓库根目录；revision 为 `0e9e39f249a16976918f6564b8830bc894c89659`。已有 [权重验证记录](run/a100/model-verification.json)。
- 虚拟环境：仓库根目录 `.venv`，Python 3.10.22、PyTorch 2.8.0+cu128、FlashInfer 0.4.0rc1、SGLang 0.5.3rc0、sgl-kernel 0.3.12。完整版本见 [实验环境快照](run/a100/requirements-freeze.txt)。
- CUDA 开发工具链：`$HOME/.local/cuda-12.8`，FlashInfer 编译目标为 `8.0`。来源见 [工具链记录](scripts/a100/toolkit-provenance.json)；实测硬件与驱动见 [hardware.json](run/a100/hardware.json)。
- 默认服务地址：`http://127.0.0.1:30000`，API 模型名 `llama-3.1-8b`。

## 校准后的服务

从仓库根目录启动：

```bash
scripts/run_a100.sh
```

根目录 `start_pdmux.sh` 转发到同一个启动器，也可执行 `bash start_pdmux.sh`。默认使用 [pdmux_pp_a100.yml](configs/pdmux_pp_a100.yml)：ALP whole-prefill、BF16、FlashInfer attention、PyTorch sampling、context 8192、KV token 上限 135168、最大并发 32、decode microbatch / CUDA graph batch 上限 16。YAML 与启动参数均使用 8 个 SM 分组，开启两槽 prefetch、async decode 和启动 decode 校准。

运行实验选出的 2048-token CPP 分块配置：

```bash
PDMUX_CHUNK=2048 scripts/run_a100.sh
```

可通过 `MODEL_PATH`、`HOST`、`PORT`、`PDMUX_POLICY`、`PDMUX_CHUNK` 覆盖对应参数，也可在命令末尾追加服务参数。该启动器固定使用物理 GPU 6、7 和上述 CUDA 12.8 工具链；两卡已有计算进程时会退出。启动时检查模型目录和未完成下载标记，但该检查不替代权重校验。首次 FlashInfer JIT 可能较慢。

前台运行按 Ctrl-C 停止。后台启动时使用新的日志和 PID 文件，避免覆盖原始实验记录：

```bash
mkdir -p run
nohup scripts/run_a100.sh > run/a100-current-server.log 2>&1 &
echo $! > run/a100-current-server.pid

# 先核对 PID 对应的命令，再停止自己启动的服务。
ps -p "$(cat run/a100-current-server.pid)" -o pid,args
kill -TERM "$(cat run/a100-current-server.pid)"
```

PID 文件只在执行上述后台启动后生成，不代表目前已有服务运行。

## 调用与快速检查

在服务就绪后，从另一个终端调用：

```bash
curl --noproxy '*' --fail-with-body http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.1-8b","messages":[{"role":"user","content":"Say hello in one sentence."}],"temperature":0,"max_tokens":32}'

.venv/bin/python scripts/smoke_local.py \
  --model llama-3.1-8b --output run/a100-current-smoke.json
```

smoke 脚本检查健康状态、模型列表、聊天、SSE 流式输出、前缀缓存及 16 个混合并发请求。使用其他端口时增加 `--base-url http://127.0.0.1:端口`。服务和验证脚本均支持 loopback 直连，避免本机 HTTP 代理干扰。

快速合成负载测量：

```bash
bash bench_pdmux.sh
```

默认使用本地 Llama tokenizer 和无需下载数据集的 `random-ids` 负载：64 个请求，构造 512-token 输入后解码为文本发送，要求输出 128 token，最大并发 32，连接 `127.0.0.1:30000`。服务重新分词后的输入长度可能略有变化。可通过 `MODEL_PATH`、`NUM_PROMPTS`、`REQUEST_RATE`、`OUTPUT_FILE` 和追加的 benchmark 参数调整测量。该命令不复现完整实验矩阵；冻结协议及独立输出核对见 [A100_EXPERIMENTS.md](A100_EXPERIMENTS.md)。

## 轻量 Triton 功能检查

[scripts/run_local.sh](scripts/run_local.sh) 使用同一个本地 Llama-3.1-8B 模型，默认仍为 GPU 6、7、TP1/PP2，但采用 Triton attention 和 [pdmux_pp_a100_local.yml](configs/pdmux_pp_a100_local.yml) 的 `80/28` 分区。其 context 为 2048、KV token 上限 8192、静态显存比例 0.15、最大并发 16、prefill 分块 256；用于接口和调度功能检查，不作为校准性能配置。

确认所选 GPU 空闲后运行：

```bash
CUDA_VISIBLE_DEVICES=6,7 PORT=30001 scripts/run_local.sh
```

可用上面的 smoke 命令并增加 `--base-url http://127.0.0.1:30001` 检查该服务。两种启动方式使用相同的默认 GPU，应分别启动和停止。

## 环境重建

以下命令供需要重建环境时使用。准备 `uv`、C++ 编译器以及完整本地模型；使用托管 Python 获取编译扩展所需的 Python 头文件。

```bash
A100_UV="$HOME/.local/bin/uv"
"$A100_UV" python install 3.10
"$A100_UV" venv --python 3.10 --managed-python .venv
"$A100_UV" pip install --python .venv/bin/python -e python pytest \
  --build-constraints configs/local-build-constraints.txt
.venv/bin/python scripts/a100/install_cuda.py \
  --provenance run/a100-toolkit-rebuild.json
CUDA_HOME="$HOME/.local/cuda-12.8" \
  .venv/bin/python scripts/build_spatial_local.py --verbose
"$A100_UV" pip check --python .venv/bin/python
```

CUDA 安装脚本校验固定版本归档并安装到用户目录，不修改系统 CUDA。`build_spatial_local.py` 编译仓库内的 Green Context 驱动扩展并安装至当前虚拟环境的 `sgl_kernel`；它使用主机 C++ 编译器及 CUDA 驱动头文件。重新安装 `sgl-kernel` 后需重新构建此扩展。重建后应与 [实验环境快照](run/a100/requirements-freeze.txt) 核对依赖版本。

## 验证结论的边界

**现有 A100 实验的严格 token 一致性未全部通过。** 两项运行时问题已经修复，但并发不匹配和一个由独立 oracle 复核的非贪心偏离仍保留在 [正确性报告](docs/a100-correctness.md)。接口检查通过、请求完成或 benchmark 成功都不能替代严格数值验证。原始结果及失败记录保存在 `run/a100/`，新的运行应使用新的输出路径。

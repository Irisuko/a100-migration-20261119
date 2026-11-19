# 实验交付与复核入口

本目录是独立的已完成实验，未覆盖旧实验。16 个正式条件、624 请求、一次重复、零新增调参。输入是固定形状合成 token-ID；真实文本计划与实际输入的偏差详见 `environment/execution-rulings.json` 和 `CONCLUSIONS.md`。原始冻结记录不被回写为另一个历史版本。

## 交付文件

| 内容 | 路径 |
|---|---|
| 完整表格 | `完整结果表.xlsx`，含性能、GPU、KV、显存、调度、配置、KV审计、网络、路由观测 |
| 报告与结论 | `REPORT.md`、`CONCLUSIONS.md` |
| 冻结配置与实验矩阵 | `FROZEN_CONFIG.md`、`matrix.csv`、`plans/jobs.json`、`plans/*.json` |
| 实际生效配置 | `effective-configurations.json`、`effective-config-summary.csv`、各运行 `worker*/server-final.json` |
| 完整命令 | `commands.sh`、`complete-commands.sh`、`runs/*/command.json`、各运行 `run.json` |
| 正式输入 | `traces/*.jsonl`、`input-provenance.json`（其 real-trace 文案须结合上述来源更正解读） |
| 原始逐请求输出 | `measurements/mini-*/A1..D2/` 中 `trace_planned.jsonl`、`trace_requests.jsonl`、`trace_outcomes.jsonl`、`trace_summary.json` 及客户端日志 |
| 资源及调度原始证据 | `measurements/mini-*/server-*.log`、`gpu-monitor.csv`、各场景 `live-metrics.jsonl`、`resource-*.json` |
| TCP/NVLink 证据 | 各场景 `network-before/after.json`、NVLink原始计数；`calibration/tcp-*.json`、`calibration/pd-tcp10/` |
| 校验结果 | `validation-final.json`、`calibration/*review*.json`、测试日志 |
| 源码与模型身份 | `source/`、`frozen-pins.json`、`environment/model-weights-verified.json`、`environment/actual-source.diff` |
| 环境/绑定/释放证据 | `environment/actual-processes.jsonl`、`gpu-final.csv`、`compute-processes-final.csv` |
| 最终文件完整性 | `MANIFEST.sha256`；目录旁独立 `*-results.tar.gz` 与 `.sha256` |

## 本次执行入口

```bash
unshare --user --map-root-user --net \
  /data01/home/mhy/workspace/muxwise-pp-sglang/.venv/bin/python \
  /data01/home/mhy/workspace/experiment-reports/pd-boundary-mini-20261118T074218Z/tools/run_mini.py \
  > /data01/home/mhy/workspace/experiment-reports/pd-boundary-mini-20261118T074218Z/driver.log 2>&1
```

驱动器取得 GPU0–3 控制锁，串行启动每个部署，以 `numactl --physcpubind=0-31,64-95 --membind=0` 运行；只清理本次拥有的进程。TCP 在私有 user/network namespace 内配置，每 GPU 共享 TX/RX 限速；没有修改宿主机网络、BIOS、驱动或 NVLink fabric。GPU4–7 未用于本实验。重执行该入口会跳过已有成功 receipt，不应删 receipt 后在本目录重跑。若需新一轮，应新建独立目录、重定向快照中绝对路径、生成新清单并重新冻结；本轮归档只用于复核。

原生条件保留两个系统各自优化；TCP 条件使用同一套经过校准的传输 hook。相对原生快照，源码差异在 `environment/actual-source.diff`。双方统一启用 scheduler metrics，每2秒抓取；GPU每1秒采样，测量包含这些诊断开销。实际正式输入种子为 A/B/C/D 的 20261118/19/20/21；继承的功能预检并非原计划 seed20261119。客户端 warmup 是与正式前缀不重叠的固定16-token序列，每条件2请求。

## 不使用 GPU 的复核命令

```bash
/data01/home/mhy/workspace/muxwise-pp-sglang/.venv/bin/python \
  /data01/home/mhy/workspace/experiment-reports/pd-boundary-mini-20261118T074218Z/tools/validate_mini.py
```

该命令会重写 `validation-final.json`，因此对归档校验应先复制目录并适配其中绝对路径，或只阅读已保存校验结果。最终处理入口 `tools/finalize_mini.py` 还会重算报告、验证模型四个 safetensors 哈希、旧两份 Python 源码快照未变、工作簿结构和采样到的线程 CPU 绑定；它不会启动 GPU 服务。硬件、依赖和模型仍需本机对应路径，归档不含模型权重和 Python 虚拟环境。

在本目录内执行 `sha256sum -c MANIFEST.sha256` 可只读验证清单内文件。在目录旁执行 `sha256sum -c pd-boundary-mini-20261118T074218Z-results.tar.gz.sha256` 可验证归档。归档排除可再生 `cache`/`__pycache__`，保留全部正式输入、逐请求记录、日志、命令、配置和源码。

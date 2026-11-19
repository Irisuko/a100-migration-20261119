# A100 私有迁移备份

目标：保存 MuxWise/SGLang 源码与8B实验，搬迁后能够恢复代码、原始记录、配置和运行环境。按用户要求排除70B模型及其专属实验配置、参数和数据。

## 内容

- `sources/`：可浏览的当前代码与旧实验仓库副本；原Git历史与未提交现场见Release快照。
- `reports/`：最近四轮8B正式实验的报告与表格。
- `environment/`：依赖、CUDA/硬件信息、8B模型配置与SHA256、原Git状态。
- `manifests/`：迁移逐文件校验、模型排除/混合记录转换清单、Release附件校验。
- `RESTORE.md`：下载、验证、解包及环境重建步骤。

## 保存边界

模型权重、`.venv`、Python字节码和pytest缓存不上传。70B专属目录/运行被排除；混合结构化清单移除70B记录，混合叙述报告不作为原始完整报告保留。通用框架源码和历史中的多模型支持保留，以免破坏8B运行。旧任务checkpoint不恢复为可继续执行的现场。原服务器所有文件保持不变。

过滤后的旧汇总文件是迁移派生物，详见转换清单；原旧清单内的全体文件哈希不代表过滤后的目录完整性，应使用本仓库新的逐文件清单。最近四轮8B正式结果生成明确标注的迁移派生包，剔除嵌入的历史70B失败记录并重建内部校验清单；8B原始测量记录另行与原件逐字节核对。原压缩包仅保留在旧服务器。

模型：Llama-3.1-8B-Instruct，BF16。此次迁移未启动GPU服务。仅完成离线文件恢复检查，不冒充新服务器GPU性能/数值验收。

## 模型与环境的后续步骤

8B权重需单独复制或按记录revision下载；70B完全不需要迁移。虚拟环境需按冻结版本与构建说明重建；本机Green Context扩展额外保存为同平台恢复参考。路径中保留原来的`/data01/home/mhy/workspace`，换路径后须先验证原始快照，再在运行副本修改配置和软链接。

## 实验入口

- [原生四框架实验](reports/fair8b-20261117T142324Z/)
- [TCP带宽实验](reports/edge8b-20261117T161715Z/)
- [PCIe主机中转实验](reports/pcie-host8b-20261118T025723Z/)
- [PD边界精简实验](reports/pd-boundary-mini-20261118T074218Z/REPORT.md)

`tools/filter_snapshot.py`、`build_manifest.py`、`package_snapshot.py` 是本次迁移构建工具，依赖原迁移工作目录布局；新服务器恢复只需要 `migration_tools.py` 和 RESTORE.md 的命令。

## 迁移验证状态

2026-11-19（UTC）：上传、远端回读校验和离线恢复验证已完成。

| 检查 | 结果 |
| --- | --- |
| Release附件 | 124个，共16,199,860,599字节，名称、大小和SHA256全部匹配 |
| 实际下载校验 | 124/124通过 |
| 下载分卷拼接后的整包SHA256 | 与已完成离线恢复验证的原包一致 |
| 空目录恢复校验 | 236,463个文件、207个软链接通过；无缺失或多余条目 |
| 原Git历史 | 两个仓库的完整性检查通过 |
| 最近四轮8B原始测量材料 | 487个文件与迁移前原件逐字节一致 |

[下载私有Release](https://github.com/Irisuko/a100-migration-20261119/releases/tag/migration-20261119)，按 [RESTORE.md](RESTORE.md) 恢复。验证证据见 `manifests/end-to-end-restoration-verification.json`、`remote-roundtrip-verification.json` 和 `local-restoration-verification.json`。这些检查不包含新服务器GPU推理验收。

# 恢复步骤

仓库 `Irisuko/a100-migration-20261119` 必须保持 Private。Release 标签为 `migration-20261119`。这是一份排除70B专属材料后的迁移备份，不是整机镜像。

本次完整流程建议预留约130GB空闲磁盘（分卷、合并包、恢复目录及8B模型/环境），后续实验输出另算。需要 git、gh、GNU tar、zstd、Python3。

## 1. 下载与验证

在新服务器登录有该私有仓库权限的 GitHub 账号，再执行：

```bash
git clone https://github.com/Irisuko/a100-migration-20261119.git
cd a100-migration-20261119
mkdir .downloads
gh release download migration-20261119 --repo Irisuko/a100-migration-20261119 --dir .downloads
(cd .downloads && sha256sum -c SHA256SUMS)
```

先校验所有分卷，再合并。分卷按文件名数字顺序拼接，不要分别解压：

```bash
cat .downloads/workspace.tar.zst.part-* > .downloads/workspace.tar.zst
zstd -t .downloads/workspace.tar.zst
mkdir restore-test
tar --zstd --same-permissions -xf .downloads/workspace.tar.zst -C restore-test
zstd -d .downloads/workspace-manifest.jsonl.zst -o .downloads/workspace-manifest.jsonl
python3 tools/migration_tools.py restore-test .downloads/workspace-manifest.jsonl
```

最后一条必须返回 `PASS`，它逐文件验证 SHA256、大小、权限和软链接目标。目标目录必须是新建空目录；不要直接解包覆盖现有工作区。

## 2. 目录布局

解包后：

```text
restore-test/
  workspace/
    muxwise-pp-sglang/         # 含原.git历史与保留的工作区改动、历史8B运行
    muxwise-pp-experiments/    # 旧实验仓库
    muxwise-pp-diagnostics/   # 保留的诊断证据
    experiment-reports/       # 独立8B实验结果
  dependencies/
    sgl_kernel-local/         # 本机Green Context扩展，仅作同平台恢复参考
```

原运行根目录为 `/data01/home/mhy/workspace`。条件允许时，在新机器使用相同路径，许多实验命令、配置及软链接就不需要改写。如果换路径，先完成上述原始字节校验，再复制一份运行副本，修改其中绝对路径/软链接；不要修改用于校验的原始解包目录。`manifests/symlinks.json` 列出了软链接。相对链接随目录移动，绝对链接仍指向原路径，需要检查。

迁移前源代码仓库有未提交修改和新增文件，本备份保留它们；70B过滤会额外删除或改写部分实验材料。`environment/*/status.txt` 是迁移前状态，`manifests/transformed-files.json` 与 `excluded-paths.json` 是派生迁移范围。不要用 `git reset --hard` 或 `git clean` 清理恢复目录，否则会丢失未提交代码或实验文件。

旧运行的 PID、端口、network namespace ID 和 GPU UUID 是历史证据，不是新服务器的可用资源。恢复不应自动启动旧控制器或继续旧任务。

## 3. 模型

没有上传任何模型权重。只需单独迁移 `models/Llama-3.1-8B-Instruct/`；不需要70B模型。

`environment/models/files.json` 保存本机8B文件大小与 SHA256，`environment/models/metadata/` 保存模型配置、tokenizer、来源和版本记录。可以从旧服务器复制8B目录，也可以按记录的官方 revision 获取，再逐文件核对。模型来源和权限沿用用户原有访问方式。

## 4. Python/CUDA 环境

`.venv` 没有打包。依赖清单在 `environment/requirements-freeze.txt`，原 Python 版本及包信息在 `installed-packages.json`，CUDA/NVCC 和驱动记录在对应文本文件。原环境使用 uv，未安装 pip；`pip-freeze.txt` 和 `pip-inspect.txt` 保存尝试时的无pip诊断，正式依赖清单由 uv 与 importlib.metadata 导出。

重建流程：

1. 安装记录的 Python 3.10 环境并在恢复的源码仓库建立 `.venv`。
2. 按冻结依赖安装，与 `requirements-freeze.txt` 核对；其中 editable SGLang 路径要指向恢复的 `muxwise-pp-sglang/python`。
3. 按源码中 `scripts/a100/install_cuda.py` 和工具链来源记录安装 CUDA 12.8 开发工具链，或配置兼容的已有工具链。
4. 按 `scripts/build_spatial_local.py` 构建 Green Context 扩展。`dependencies/sgl_kernel-local` 中的旧二进制仅供兼容平台恢复参考，不保证换 Python/CUDA/硬件后可直接复用。
5. 先检查GPU资源和拓扑，再按对应实验冻结配置做功能预检。不要直接运行默认启动脚本占用其硬编码GPU。

`run/` 中保留的构建/校准材料、源码快照和工具链记录也可用于追踪依赖来源。工作区外的CUDA工具链、uv托管Python以及账户配置不在本次GitHub备份内。

## 5. 正式实验与历史资料

最近四轮8B实验的独立归档也作为Release附件，方便不下载整个历史快照时单独查看。这些迁移派生包移除了原归档夹带的历史70B失败记录，并重新生成内部校验清单；8B请求/输出/统计记录与旧服务器逐字节核对。原压缩包保持在旧服务器，不上传。既往混合8B/70B的结构化汇总已移除70B记录；有转换记录的文件不能当作未经修改的原始全模型结果。旧全目录校验清单可能引用已排除文件，以本次 `workspace-manifest.jsonl` 为迁移完整性依据。

本次验证是私有可见性、上传/下载文件一致性和离线恢复检查，尚未在新服务器验证模型推理、数值正确性或性能。

## 旧混合任务脚本的限制

旧的跨8B/70B实验生成器可能硬性要求两个模型的配置同时存在。本次按用户要求只保留非70B投影，部分共享参数改为8B专用键名，因此不能直接恢复这些旧混合任务控制器或保证其重新生成全部历史计划。继续8B实验时，应使用已冻结的8B单项配置/命令与各独立实验的执行工具；如确需旧混合生成器，应另行改造为单模型版本。保留通用框架源码不表示70B实验材料已备份。

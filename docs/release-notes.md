私有A100迁移备份，排除70B模型及其专属实验配置、参数与数据。

- 完整非70B工作区快照：workspace.tar.zst.part-000至117，按序拼接后解压。
- 逐文件清单：workspace-manifest.jsonl.zst；附件校验：SHA256SUMS。
- 四轮8B正式实验提供独立migration.tar.gz派生包，已剔除历史70B失败记录并重建内部校验；487个原始输入/请求/统计文件与原件相同。
- 模型权重和.venv不包含；保存了8B模型元数据/哈希、依赖、编译扩展与重建说明。
- 共用框架代码/历史保留多模型支持；旧混合任务生成器需适配，使用冻结8B单项命令恢复实验。

恢复步骤见主分支RESTORE.md。离线恢复与远端下载校验结果将记录在manifests/remote-roundtrip-verification.json。

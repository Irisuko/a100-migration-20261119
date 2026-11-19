私有A100迁移备份，排除70B模型及其专属实验配置、参数与数据。

- 按保存边界过滤的工作区快照：workspace.tar.zst.part-000至117，按序拼接后解压。
- 逐文件清单：workspace-manifest.jsonl.zst；附件校验：SHA256SUMS。
- 四轮8B正式实验提供独立migration.tar.gz派生包，已剔除历史70B失败记录并重建内部校验；487个原始输入/请求/统计文件与原件相同。
- 模型权重和.venv不包含；保存了8B模型元数据/哈希、依赖、编译扩展与重建说明。
- 共用框架代码/历史保留多模型支持；旧混合任务生成器需适配，使用冻结8B单项命令恢复实验。

共124个附件，16,199,860,599字节。全部附件已实际下载并核对SHA256；118个下载分卷拼接后的整包SHA256与本地离线恢复验证使用的原包一致。空目录恢复的236,463个文件和207个软链接校验通过，两个原Git仓库的完整性检查通过。

恢复步骤见本标签的RESTORE.md。证据见manifests/end-to-end-restoration-verification.json、remote-roundtrip-verification.json和local-restoration-verification.json。未执行新服务器GPU推理或性能验收。

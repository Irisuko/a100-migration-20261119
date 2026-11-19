# A100 私有迁移 Implementation Plan

> 执行方式：inline，执行已获授权的迁移方案。

**Goal:** 私有GitHub仓库与经过下载复核的实验备份。
**Architecture:** 原目录只读；独立快照、Git浏览副本、Release分卷；恢复后逐文件校验。
**Tech Stack:** git,gh,rsync,tar,zstd,Python。
**Spec:** ../specs/2026-11-19-migration-design.md

## Global Constraints
不运行GPU，不改原仓库/归档，不上传模型权重/venv/账户凭据；仓库必须private。

## Review Focus
未提交与被忽略实验记录保留；软链接和外部路径解释；分卷损坏可检出；模型/环境不冒充已备份；远端下载验证。

## Tasks
- [x] 清点、稳定快照、记录原Git状态和排除项，扫描上传内容。
- [x] 整理可浏览源码/报告与环境和模型清单，生成恢复校验工具。
- [x] 创建分卷、逐文件验证解包、保留旧Git历史/未提交状态。
- [x] 创建private仓库并推送，上传Release，重新下载校验，确认private与文件完整。

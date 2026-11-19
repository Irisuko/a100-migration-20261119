# 命令与产物说明

本目录为独立实验记录，原NVLink实验位于protocol.json所列baseline目录。当前结果状态以status.json及validation-final.json为准。

|内容|文件|
|---|---|
|完整性能、显存、KV池、通信校准表|完整结果表.xlsx、REPORT.md、summary-edge.csv|
|冻结公共与框架配置|protocol.json、frozen-configurations.json、frozen-effective-configurations.json、frozen-configurations.csv|
|初始顺序、用户恢复及生命周期补充|plans/jobs.json、plans/resume.json、plans/completion.json|
|实际worker/router/client完整命令|complete-commands.sh、runs/*/command.json、measurements/*/run.json|
|每rank实际网络环境|measurements/*/network-rank-*.json|
|链路实现及实际队列计数|tools/edge_network.py、measurements/*/*/network-before.json、network-after.json|
|逐请求计划、状态、token-ID与接收时间|measurements/*/*/trace_planned.jsonl、trace_outcomes.jsonl、trace_requests.jsonl|
|控制器、资源审计、故障与恢复凭据|receipts/、driver-*.log、runs/|
|用户暂停前产物|snapshots/before-resume/、STOPPED.md、resume-state.json|
|历史中断组，独立列示不按分数取舍|historical-interrupted-attempt.csv|
|软件、模型、主机与NUMA信息|environment/|

当前执行入口：

```bash
unshare -Urn /data01/home/mhy/workspace/muxwise-pp-sglang/.venv/bin/python -u /data01/home/mhy/workspace/experiment-reports/edge8b-20261117T161715Z/tools/complete_edge.py
```

这条入口负责私有网络命名空间、TBF限速、GPU0–3所有权、串行启动和释放。实际模型进程使用CPU0–31,64–95及NUMA0，完整numactl参数记录在commands.sh和runs/*/command.json。不要脱离该生命周期直接执行worker命令。

目录保留了历史版本启动器及失败记录。当前入口会跳过已有有效凭据，不会覆盖已测目录；它不是在原目录重复测量的按钮。若要开展另一轮实验，应新建独立根目录，更新绝对路径并重新冻结计划及哈希。模型权重不打入产物包，按environment/model-provenance.json中的位置和校验和提供。

同一GPU模拟一个节点：全部连接共享单向TX和RX额度，二者全双工。每节点edge0出口和桥到节点edgepN出口分别设置TBF，25gbit或10gbit，burst256kb，limit16777216。不限速对照移除TBF，但保持相同TCP传输。CUDA IPC/custom allreduce、NCCL P2P/SHM及Mooncake NVLink直连路径均避开；原生Chunked/PD overlap和Mux重叠策略保留。

截止后的回收不改变已冻结的请求计量。Chunked在取消/缓存复用路径暴露异常后，后续受限带宽窗口使用单独服务进程；存在未完成请求时通过拥有者控制器结束进程释放GPU，不声称其进程内静默回收审计通过。此生命周期差异与跨执行时段背景负载差异均在报告披露。

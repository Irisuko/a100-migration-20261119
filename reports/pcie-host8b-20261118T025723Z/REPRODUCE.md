# PCIe 主机中转实验复现与产物

本轮对应 protocol.json 的四种架构、16个独立条件，均使用原开发选择c0；未新增性能调参。

执行入口：

```bash
/data01/home/mhy/workspace/muxwise-pp-sglang/.venv/bin/python /data01/home/mhy/workspace/experiment-reports/pcie-host8b-20261118T025723Z/tools/run_host.py
```

入口通过GPU0–3全局互斥锁、token归属检查和独立生命周期执行；已有成功回执会跳过，失败保留并停下。新的重复实验应另建独立根目录，重新生成路径和冻结计划，不能在本目录覆盖。每个worker继承numactl CPU0–31,64–95/NUMA0约束。

实际客户端与模型命令：complete-commands.sh、runs/*/command.json、measurements/*/run.json。模型worker命令不能脱离entry_host.py环境与控制器单独执行，否则可能重新使用NVLink。entry_host.py明确设置NCCL P2P关闭、SHM开启、NVLS关闭，去掉Mooncake NVLink/TCP开关；PCIE_HOST_ROOT启用PD专用适配器。

PD适配器是本轮实验性传输实现，保留原PD调度与缓存；每个发送线程/接收实例连接使用16MiB pinned共享mmap。发送方仅D2H，接收方仅H2D；AF_UNIX仅承载地址、长度和完成通知。每批确认在目标CUDA拷贝同步完成后发送。原生HTTP/ZMQ控制消息仍可使用本机TCP，不包含本适配器的KV数据。不得把本轮称为PCIe P2P或原生Mooncake优化上限。

校准：calibration/host-final记录最终源码的真实四GPU allreduce、CUDA graph、PP邻接发送、两对并发PD传输和逐GPU NVLink计数。calibration/p2p-pxb记录P2P_LEVEL=PXB仍走NVLink的反例。environment/*topology*与*p2p*记录平台支持状态。纯PCIe P2P的跳过结论见calibration/path-verdict.json。

结果：完整结果表.xlsx、REPORT.md、summary-host.csv、comparison-native-tcp-host.csv。KV池分配与rank显存分别为kv-pools.csv、rank-memory.csv。原始逐请求文件在measurements/*/*/trace_*.jsonl。资源审计见receipts/*-resource.json；NVLink计量区间含缓存清空、两次客户端预热、正式回放和排空。

核验：tools/validate_host.py从原始请求重新计算所有指标、检查身份与配置、串行区间和传输证据；tools/host_report.py生成表格。frozen-pins.json冻结执行源码、计划、配置和trace。最终包不含模型权重或编译cache；权重SHA见environment/model-provenance.json。

原生NVLink与不限速TCP引用历史独立实验，未在本轮重跑；本轮每条件独立进程，也不同于历史共享服务窗口，报告须保留这一差异及跨时段共享主机干扰限制。

PD额外计量限制：适配器每批传输在两端同步写JSON审计文件并获取进程内锁，该开销计入服务时延，没有事后扣除。Session标识以Unix路径加进程编号后缀兼容原生队列分配公式；与原Mooncake RPC端口的队列映射可能不同。

纯PCIe P2P补充调查：安装驱动暴露NVreg_NvLinkDisable，当前值0。官方nv-reg.h说明该选项全局禁止NVLink初始化/训练、需要整机重启。它不是已验证的PCIe P2P启用保证；本轮未执行全局变更，保存官方源码和modinfo输出，按用户的不可用则跳过规则结束此分支。

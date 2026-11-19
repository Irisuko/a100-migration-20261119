# PCIe 主机内存中转：完整实验结果

独立目录：`/data01/home/mhy/workspace/experiment-reports/pcie-host8b-20261118T025723Z`。已计量 16/16 条件；每条件单次、无新增调参。

模型数据路径为 GPU→PCIe→主机内存→PCIe→GPU，不是 PCIe P2P。TP/PP 使用 NCCL SHM；PD 使用本轮实验性 pinned mmap 适配器，Unix socket 只传描述符/完成通知，KV与辅助数据经共享缓冲区搬运。PD调度、配对、缓存和原生overlap保留，但传输后端已替换，不能声称是原生Mooncake最优实现。适配器在每批共享缓冲区传输的两端同步更新JSON审计文件，文件写入与进程内锁的开销也包含在计量中；本轮未单独测量或扣除该开销。

纯PCIe P2P：跳过。当前平台PCIe P2P能力为NS；NVLink P2P为OK。实际NCCL_P2P_LEVEL=PXB探针仍有NVLink流量，不能标为PCIe直连。已找到官方NVreg_NvLinkDisable全局选项，本机值为0；官方说明其影响所有GPU且需整机重启，并不保证启用PCIe P2P。该整机改动不属于本轮四卡实验范围，因此未修改驱动、BIOS或Fabric Manager，未验证重启后的P2P可用性。

固定 Llama-3.1-8B-Instruct BF16；GPU0–3；CPU0–31,64–95、NUMA0；串行。公共running128、graph16、mem0.85。Chunked TP4/chunk2048；MuxWise TP4/MLP4096/SM B；MuxWise-PP PP4/MLP4096/PPchunk8192/groups8/tpot_coeff0.85/inflight65536/refill16384/budget16384；PD 2P+2D各TP1/chunk4096。原生Chunked/PD overlap和Mux重叠均保留。禁用custom allreduce、NCCL P2P和NVLS以防绕过主机路径，启用SHM，卡间张量/KV不经TCP；客户端HTTP请求/响应及控制消息仍可使用TCP。

全部16个条件统一独立服务进程，避免截止取消后的缓存复用异常。每条件沿用功能预检、缓存清空、2请求客户端预热；conversation65请求/toolagent135请求，原连续源窗口[3000,3020)，时间伸缩产生50/25秒窗口，最后发送后120秒截止。与历史实验的进程生命周期不同，不能把差值全部归因于通信介质。

## 延迟与联合SLO

TTFT秒、TPOT毫秒，依次p50/p95/p99。SLO=TTFT≤5秒且TPOT≤100毫秒，同一请求同时满足，分母为全部计划请求。

|系统|负载|窗口秒|完成/失败/未完成/未发送|TTFT发送|TTFT到达|TPOT|TPOT样本|SLO发送/到达 %|
|---|---|---|---|---|---|---|---|---|
|PD|conversation|50.00|65/0/0/0|1.15/5.26/6.75|1.16/5.26/6.75|13.20/15.54/16.04|65|93.85/93.85|
|Chunked|conversation|50.00|65/0/0/0|7.30/20.29/26.15|7.30/20.29/26.15|124.29/257.22/271.28|65|6.15/6.15|
|MuxWise|conversation|50.00|65/0/0/0|6.94/19.50/24.03|6.94/19.51/24.03|18.50/21.55/24.01|65|38.46/38.46|
|MuxWise-PP|conversation|50.00|65/0/0/0|1.63/3.78/4.48|1.64/3.78/4.49|27.70/50.55/54.19|65|100.00/100.00|
|PD|conversation|25.00|65/0/0/0|1.96/6.65/8.30|1.96/6.65/8.30|14.71/17.69/19.41|65|89.23/89.23|
|Chunked|conversation|25.00|65/0/0/0|18.14/38.03/44.28|18.15/38.03/44.29|156.05/263.01/276.33|65|0.00/0.00|
|MuxWise|conversation|25.00|65/0/0/0|16.18/36.44/42.00|16.19/36.44/42.01|18.97/22.23/26.50|65|18.46/18.46|
|MuxWise-PP|conversation|25.00|65/0/0/0|1.67/4.00/4.74|1.67/4.01/4.75|40.40/60.23/69.40|65|100.00/100.00|
|PD|toolagent|50.00|135/0/0/0|1.49/3.92/5.25|1.50/3.93/5.26|14.12/22.44/23.86|135|97.04/97.04|
|Chunked|toolagent|50.00|135/0/0/0|8.48/18.72/25.63|8.49/18.74/25.64|248.20/279.45/285.91|135|0.00/0.00|
|MuxWise|toolagent|50.00|135/0/0/0|8.16/17.18/23.17|8.17/17.19/23.18|19.35/26.49/34.60|135|30.37/30.37|
|MuxWise-PP|toolagent|50.00|135/0/0/0|1.95/3.83/4.51|1.95/3.83/4.53|50.49/98.68/112.98|135|94.81/94.81|
|PD|toolagent|25.00|135/0/0/0|1.73/4.52/8.47|1.74/4.53/8.48|16.96/27.11/29.73|135|95.56/95.56|
|Chunked|toolagent|25.00|135/0/0/0|16.23/37.26/44.20|16.24/37.27/44.21|249.12/285.05/290.68|135|0.00/0.00|
|MuxWise|toolagent|25.00|135/0/0/0|19.25/35.91/41.91|19.26/35.93/41.92|19.56/26.12/32.42|135|6.67/6.67|
|MuxWise-PP|toolagent|25.00|135/0/0/0|1.86/3.66/4.72|1.87/3.67/4.73|55.50/99.07/112.65|135|94.81/94.81|

## 吞吐、goodput与积压

每对数值为到达窗口/全排空；token指标单位token/s，goodput只计SLO通过且在对应边界前完成请求的输出token。

|系统|负载|窗口秒|输出吞吐 窗/排|输入吞吐 窗/排|总吞吐 窗/排|发送goodput 窗/排|到达goodput 窗/排|完成req/s 窗/排|发送goodput req/s 窗/排|到达goodput req/s 窗/排|
|---|---|---|---|---|---|---|---|---|---|---|
|PD|conversation|50.00|340.92/315.99|10099.96/8922.57|10440.88/9238.56|316.74/299.05|316.74/299.05|1.20/1.10|1.14/1.04|1.14/1.04|
|Chunked|conversation|50.00|85.64/219.98|7009.78/6211.70|7095.42/6431.68|5.64/22.45|5.64/22.45|0.28/0.77|0.04/0.05|0.04/0.05|
|MuxWise|conversation|50.00|256.00/250.21|7009.78/7065.32|7265.78/7315.54|155.56/104.63|155.56/104.63|0.86/0.87|0.50/0.34|0.50/0.34|
|MuxWise-PP|conversation|50.00|306.86/316.44|10504.82/8935.49|10811.68/9251.94|243.74/316.44|243.74/316.44|1.00/1.11|1.00/1.11|1.00/1.11|
|PD|conversation|25.00|534.32/479.65|15207.68/13543.94|15742.00/14023.59|398.20/413.89|398.20/413.89|1.72/1.68|1.64/1.50|1.64/1.50|
|Chunked|conversation|25.00|45.24/231.62|7305.24/6540.21|7350.48/6771.83|0.00/0.00|0.00/0.00|0.12/0.81|0.00/0.00|0.00/0.00|
|MuxWise|conversation|25.00|229.08/266.45|7196.68/7523.76|7425.76/7790.21|124.80/44.69|124.80/44.69|0.80/0.93|0.48/0.17|0.48/0.17|
|MuxWise-PP|conversation|25.00|347.68/423.30|18005.24/11952.76|18352.92/12376.06|141.64/423.30|141.64/423.30|0.96/1.48|0.96/1.48|0.96/1.48|
|PD|toolagent|50.00|362.82/347.48|17184.44/15772.87|17547.26/16120.35|323.12/326.10|323.12/326.10|2.52/2.37|2.48/2.30|2.48/2.30|
|Chunked|toolagent|50.00|85.84/229.81|13055.94/10431.47|13141.78/10661.28|0.00/0.00|0.00/0.00|1.26/1.57|0.00/0.00|0.00/0.00|
|MuxWise|toolagent|50.00|278.18/263.39|13055.94/11955.98|13334.12/12219.37|109.78/73.08|109.78/73.08|2.00/1.80|0.82/0.55|0.82/0.55|
|MuxWise-PP|toolagent|50.00|321.98/333.38|17555.08/15132.98|17877.06/15466.36|266.74/324.50|266.74/324.50|2.38/2.28|2.26/2.16|2.26/2.16|
|PD|toolagent|25.00|579.44/492.29|28595.52/22346.40|29174.96/22838.69|427.52/439.31|427.52/439.31|4.04/3.36|4.00/3.21|4.00/3.21|
|Chunked|toolagent|25.00|54.12/241.40|12867.84/10957.76|12921.96/11199.16|0.00/0.00|0.00/0.00|0.84/1.65|0.00/0.00|0.00/0.00|
|MuxWise|toolagent|25.00|226.60/277.18|11957.96/12581.71|12184.56/12858.89|59.00/20.67|59.00/20.67|1.68/1.89|0.36/0.13|0.36/0.13|
|MuxWise-PP|toolagent|25.00|351.64/429.54|32076.60/19497.77|32428.24/19927.31|109.28/418.10|109.28/418.10|3.24/2.93|3.00/2.78|3.00/2.78|

|系统|负载|窗口秒|排空秒|窗口末积压|峰值积压|平均积压|截止剩余|发送滞后p50/p95/p99秒|
|---|---|---|---|---|---|---|---|---|
|PD|conversation|50.00|58.87|5|18|6.80|0|0.00/0.01/0.01|
|Chunked|conversation|50.00|84.56|51|56|24.43|0|0.00/0.01/0.01|
|MuxWise|conversation|50.00|74.34|22|28|12.74|0|0.00/0.01/0.01|
|MuxWise-PP|conversation|50.00|58.78|15|22|11.05|0|0.00/0.01/0.01|
|PD|conversation|25.00|38.78|22|28|13.43|0|0.00/0.01/0.01|
|Chunked|conversation|25.00|80.31|62|62|28.03|0|0.00/0.01/0.04|
|MuxWise|conversation|25.00|69.81|45|48|22.36|0|0.00/0.01/0.01|
|MuxWise-PP|conversation|25.00|43.94|41|48|21.62|0|0.00/0.01/0.01|
|PD|toolagent|50.00|56.93|9|32|9.68|0|0.01/0.01/0.02|
|Chunked|toolagent|50.00|86.09|72|82|38.75|0|0.01/0.03/0.04|
|MuxWise|toolagent|50.00|75.11|35|52|22.76|0|0.01/0.02/0.02|
|MuxWise-PP|toolagent|50.00|59.34|16|38|15.53|0|0.01/0.02/0.02|
|PD|toolagent|25.00|40.19|34|43|19.44|0|0.01/0.01/0.02|
|Chunked|toolagent|25.00|81.95|114|115|52.13|0|0.01/0.01/0.02|
|MuxWise|toolagent|25.00|71.37|93|97|44.45|0|0.01/0.02/0.04|
|MuxWise-PP|toolagent|25.00|46.06|54|64|30.98|0|0.01/0.01/0.04|

## 原生互联、TCP与主机中转对照

历史对照未重跑；以下分开列实际输出吞吐和到达口径goodput，均为全排空输出token/s。

|系统|负载|窗口秒|吞吐 原生/TCP/主机|goodput 原生/TCP/主机|SLO% 原生/TCP/主机|
|---|---|---|---|---|---|
|PD|conversation|50.00|321.41/309.47/315.99|309.02/275.05/299.05|95.38/89.23/93.85|
|Chunked|conversation|50.00|356.42/138.41/219.98|356.42/0.00/22.45|100.00/0.00/6.15|
|MuxWise|conversation|50.00|353.66/188.60/250.21|353.66/27.67/104.63|100.00/20.00/38.46|
|MuxWise-PP|conversation|50.00|317.35/312.10/316.44|317.35/303.56/316.44|100.00/98.46/100.00|
|PD|conversation|25.00|480.00/500.10/479.65|418.76/446.06/413.89|86.15/89.23/89.23|
|Chunked|conversation|25.00|534.08/144.75/231.62|534.08/0.00/0.00|100.00/0.00/0.00|
|MuxWise|conversation|25.00|613.90/195.63/266.45|613.90/14.88/44.69|100.00/7.69/18.46|
|MuxWise-PP|conversation|25.00|412.52/398.16/423.30|401.23/387.26/423.30|98.46/98.46/100.00|
|PD|toolagent|50.00|340.65/318.47/347.48|328.56/218.81/326.10|98.52/74.07/97.04|
|Chunked|toolagent|50.00|376.16/147.41/229.81|376.16/0.00/0.00|100.00/0.00/0.00|
|MuxWise|toolagent|50.00|375.00/194.41/263.39|375.00/14.50/73.08|100.00/6.67/30.37|
|MuxWise-PP|toolagent|50.00|331.37/330.07/333.38|330.92/321.33/324.50|93.33/95.56/94.81|
|PD|toolagent|25.00|539.42/393.42/492.29|478.28/100.22/439.31|96.30/30.37/95.56|
|Chunked|toolagent|25.00|550.79/151.66/241.40|536.62/0.00/0.00|99.26/0.00/0.00|
|MuxWise|toolagent|25.00|640.62/201.90/277.18|640.62/15.05/20.67|100.00/6.67/6.67|
|MuxWise-PP|toolagent|25.00|432.03/418.28/429.54|420.08/400.56/418.10|92.59/91.85/94.81|

## 路径校准

真实四GPU载荷校验均通过；主机中转的两种传输NVLink增量为零，PXB探针仍使用NVLink。通信耗时含发起和同步开销，不是PCIe链路峰值带宽。

|探针|NVLink TX KiB|NVLink RX KiB|结论|
|---|---|---|---|
|host-final/nccl-link-counters.json|0|0|SHM payload validated; zero NVLink|
|host-final/pd-link-counters.json|0|0|PD host staging payload validated; zero NVLink|
|p2p-pxb/nccl-link-counters.json|2962388|2962384|Still NVLink; not pure PCIe P2P|

## 本轮观测结论

|负载|窗口秒|最高观测排空吞吐系统|token/s|最高观测到达goodput系统|token/s|
|---|---|---|---|---|---|
|conversation|50.00|MuxWise-PP|316.44|MuxWise-PP|316.44|
|conversation|25.00|PD|479.65|MuxWise-PP|423.30|
|toolagent|50.00|PD|347.48|PD|326.10|
|toolagent|25.00|PD|492.29|PD|439.31|

这些排序只描述本次冻结配置和短窗口观测；吞吐与goodput分开排序，不外推稳态容量。

## 各rank显存与KV池

显存峰值包含启动、功能预检、预热及排空；nvidia-smi每秒采样可能漏掉瞬时峰值。PyTorch峰值取资源审计快照，未完成请求的进程退出场景可能没有计量结束后的快照。TP/PP分片池容量不可跨rank简单相加。

|运行|系统|GPU|角色|PP/TP|KV token容量|nvidia-smi峰值MiB|PyTorch峰值分配GiB|资源审计|
|---|---|---|---|---|---|---|---|---|
|host-pd-conversation-x2.5|PD|0|prefill|0/0|425303|70487.00|67.90|passed|
|host-pd-conversation-x2.5|PD|1|prefill|0/0|425303|70417.00|67.90|passed|
|host-pd-conversation-x2.5|PD|2|decode|0/0|425303|69825.00|67.49|passed|
|host-pd-conversation-x2.5|PD|3|decode|0/0|425303|69821.00|67.49|passed|
|host-chunked-conversation-x2.5|Chunked|0|model|0/0|2065011|70131.00|67.53|passed|
|host-chunked-conversation-x2.5|Chunked|1|model|0/1|2065011|70131.00|67.53|passed|
|host-chunked-conversation-x2.5|Chunked|2|model|0/2|2065011|70131.00|67.53|passed|
|host-chunked-conversation-x2.5|Chunked|3|model|0/3|2065011|70131.00|67.53|passed|
|host-muxwise-conversation-x2.5|MuxWise|0|model|0/0|2063814|75829.00|69.70|passed|
|host-muxwise-conversation-x2.5|MuxWise|1|model|0/1|2063814|75829.00|69.70|passed|
|host-muxwise-conversation-x2.5|MuxWise|2|model|0/2|2063814|75829.00|69.70|passed|
|host-muxwise-conversation-x2.5|MuxWise|3|model|0/3|2063814|75829.00|69.70|passed|
|host-muxpp-conversation-x2.5|MuxWise-PP|0|model|0/0|2047612|72971.00|68.67|passed|
|host-muxpp-conversation-x2.5|MuxWise-PP|1|model|1/0|2047612|71459.00|67.88|passed|
|host-muxpp-conversation-x2.5|MuxWise-PP|2|model|2/0|2047612|71603.00|67.88|passed|
|host-muxpp-conversation-x2.5|MuxWise-PP|3|model|3/0|2047612|72385.00|68.85|passed|
|host-pd-conversation-x1.25|PD|0|prefill|0/0|425303|70489.00|67.90|passed|
|host-pd-conversation-x1.25|PD|1|prefill|0/0|425302|70495.00|67.90|passed|
|host-pd-conversation-x1.25|PD|2|decode|0/0|425302|69821.00|67.49|passed|
|host-pd-conversation-x1.25|PD|3|decode|0/0|425303|69825.00|67.49|passed|
|host-chunked-conversation-x1.25|Chunked|0|model|0/0|2065011|70131.00|67.55|passed|
|host-chunked-conversation-x1.25|Chunked|1|model|0/1|2065011|70131.00|67.55|passed|
|host-chunked-conversation-x1.25|Chunked|2|model|0/2|2065011|70131.00|67.55|passed|
|host-chunked-conversation-x1.25|Chunked|3|model|0/3|2065011|70131.00|67.55|passed|
|host-muxwise-conversation-x1.25|MuxWise|0|model|0/0|2063814|75787.00|69.70|passed|
|host-muxwise-conversation-x1.25|MuxWise|1|model|0/1|2063814|75787.00|69.70|passed|
|host-muxwise-conversation-x1.25|MuxWise|2|model|0/2|2063814|75787.00|69.70|passed|
|host-muxwise-conversation-x1.25|MuxWise|3|model|0/3|2063814|75787.00|69.70|passed|
|host-muxpp-conversation-x1.25|MuxWise-PP|0|model|0/0|2047612|73667.00|68.67|passed|
|host-muxpp-conversation-x1.25|MuxWise-PP|1|model|1/0|2047612|72547.00|67.88|passed|
|host-muxpp-conversation-x1.25|MuxWise-PP|2|model|2/0|2047612|71787.00|67.88|passed|
|host-muxpp-conversation-x1.25|MuxWise-PP|3|model|3/0|2047612|73049.00|68.87|passed|
|host-pd-toolagent-x2.5|PD|0|prefill|0/0|425303|70373.00|67.90|passed|
|host-pd-toolagent-x2.5|PD|1|prefill|0/0|425302|70563.00|67.90|passed|
|host-pd-toolagent-x2.5|PD|2|decode|0/0|425303|69825.00|67.49|passed|
|host-pd-toolagent-x2.5|PD|3|decode|0/0|425303|69825.00|67.50|passed|
|host-chunked-toolagent-x2.5|Chunked|0|model|0/0|2065011|70153.00|67.56|passed|
|host-chunked-toolagent-x2.5|Chunked|1|model|0/1|2065011|70153.00|67.56|passed|
|host-chunked-toolagent-x2.5|Chunked|2|model|0/2|2065011|70153.00|67.56|passed|
|host-chunked-toolagent-x2.5|Chunked|3|model|0/3|2065011|70153.00|67.56|passed|
|host-muxwise-toolagent-x2.5|MuxWise|0|model|0/0|2063814|75877.00|69.63|passed|
|host-muxwise-toolagent-x2.5|MuxWise|1|model|0/1|2063814|75877.00|69.63|passed|
|host-muxwise-toolagent-x2.5|MuxWise|2|model|0/2|2063814|75877.00|69.63|passed|
|host-muxwise-toolagent-x2.5|MuxWise|3|model|0/3|2063814|75877.00|69.63|passed|
|host-muxpp-toolagent-x2.5|MuxWise-PP|0|model|0/0|2047612|72979.00|68.70|passed|
|host-muxpp-toolagent-x2.5|MuxWise-PP|1|model|1/0|2047612|71251.00|67.91|passed|
|host-muxpp-toolagent-x2.5|MuxWise-PP|2|model|2/0|2047612|71251.00|67.91|passed|
|host-muxpp-toolagent-x2.5|MuxWise-PP|3|model|3/0|2047612|72407.00|68.89|passed|
|host-pd-toolagent-x1.25|PD|0|prefill|0/0|425302|70359.00|67.90|passed|
|host-pd-toolagent-x1.25|PD|1|prefill|0/0|425303|70485.00|67.90|passed|
|host-pd-toolagent-x1.25|PD|2|decode|0/0|425302|69825.00|67.50|passed|
|host-pd-toolagent-x1.25|PD|3|decode|0/0|425302|69825.00|67.50|passed|
|host-chunked-toolagent-x1.25|Chunked|0|model|0/0|2065011|70151.00|67.56|passed|
|host-chunked-toolagent-x1.25|Chunked|1|model|0/1|2065011|70153.00|67.56|passed|
|host-chunked-toolagent-x1.25|Chunked|2|model|0/2|2065011|70151.00|67.56|passed|
|host-chunked-toolagent-x1.25|Chunked|3|model|0/3|2065011|70151.00|67.56|passed|
|host-muxwise-toolagent-x1.25|MuxWise|0|model|0/0|2063814|75913.00|69.63|passed|
|host-muxwise-toolagent-x1.25|MuxWise|1|model|0/1|2063814|75913.00|69.63|passed|
|host-muxwise-toolagent-x1.25|MuxWise|2|model|0/2|2063814|75913.00|69.63|passed|
|host-muxwise-toolagent-x1.25|MuxWise|3|model|0/3|2063814|75913.00|69.63|passed|
|host-muxpp-toolagent-x1.25|MuxWise-PP|0|model|0/0|2047612|73901.00|68.70|passed|
|host-muxpp-toolagent-x1.25|MuxWise-PP|1|model|1/0|2047612|72649.00|67.91|passed|
|host-muxpp-toolagent-x1.25|MuxWise-PP|2|model|2/0|2047612|72073.00|67.91|passed|
|host-muxpp-toolagent-x1.25|MuxWise-PP|3|model|3/0|2047612|73121.00|68.89|passed|

|运行|GPU|角色|PP/TP|池编号|token容量|K GiB|V GiB|
|---|---|---|---|---|---|---|---|
|host-pd-conversation-x2.5|0|prefill|0/0|0|425303|25.96|25.96|
|host-pd-conversation-x2.5|1|prefill|0/0|0|425303|25.96|25.96|
|host-pd-conversation-x2.5|2|decode|0/0|0|425303|25.96|25.96|
|host-pd-conversation-x2.5|3|decode|0/0|0|425303|25.96|25.96|
|host-chunked-conversation-x2.5|2|model|0/2|0|2065011|31.51|31.51|
|host-chunked-conversation-x2.5|0|model|0/0|0|2065011|31.51|31.51|
|host-chunked-conversation-x2.5|3|model|0/3|0|2065011|31.51|31.51|
|host-chunked-conversation-x2.5|1|model|0/1|0|2065011|31.51|31.51|
|host-muxwise-conversation-x2.5|3|model|0/3|0|2063814|31.49|31.49|
|host-muxwise-conversation-x2.5|2|model|0/2|0|2063814|31.49|31.49|
|host-muxwise-conversation-x2.5|0|model|0/0|0|2063814|31.49|31.49|
|host-muxwise-conversation-x2.5|1|model|0/1|0|2063814|31.49|31.49|
|host-muxpp-conversation-x2.5|1|model|1/0|0|2047612|31.24|31.24|
|host-muxpp-conversation-x2.5|2|model|2/0|0|2047612|31.24|31.24|
|host-muxpp-conversation-x2.5|3|model|3/0|0|2047612|31.24|31.24|
|host-muxpp-conversation-x2.5|0|model|0/0|0|2047612|31.24|31.24|
|host-pd-conversation-x1.25|0|prefill|0/0|0|425303|25.96|25.96|
|host-pd-conversation-x1.25|1|prefill|0/0|0|425302|25.96|25.96|
|host-pd-conversation-x1.25|2|decode|0/0|0|425302|25.96|25.96|
|host-pd-conversation-x1.25|3|decode|0/0|0|425303|25.96|25.96|
|host-chunked-conversation-x1.25|3|model|0/3|0|2065011|31.51|31.51|
|host-chunked-conversation-x1.25|1|model|0/1|0|2065011|31.51|31.51|
|host-chunked-conversation-x1.25|0|model|0/0|0|2065011|31.51|31.51|
|host-chunked-conversation-x1.25|2|model|0/2|0|2065011|31.51|31.51|
|host-muxwise-conversation-x1.25|0|model|0/0|0|2063814|31.49|31.49|
|host-muxwise-conversation-x1.25|3|model|0/3|0|2063814|31.49|31.49|
|host-muxwise-conversation-x1.25|2|model|0/2|0|2063814|31.49|31.49|
|host-muxwise-conversation-x1.25|1|model|0/1|0|2063814|31.49|31.49|
|host-muxpp-conversation-x1.25|1|model|1/0|0|2047612|31.24|31.24|
|host-muxpp-conversation-x1.25|2|model|2/0|0|2047612|31.24|31.24|
|host-muxpp-conversation-x1.25|0|model|0/0|0|2047612|31.24|31.24|
|host-muxpp-conversation-x1.25|3|model|3/0|0|2047612|31.24|31.24|
|host-pd-toolagent-x2.5|0|prefill|0/0|0|425303|25.96|25.96|
|host-pd-toolagent-x2.5|1|prefill|0/0|0|425302|25.96|25.96|
|host-pd-toolagent-x2.5|2|decode|0/0|0|425303|25.96|25.96|
|host-pd-toolagent-x2.5|3|decode|0/0|0|425303|25.96|25.96|
|host-chunked-toolagent-x2.5|0|model|0/0|0|2065011|31.51|31.51|
|host-chunked-toolagent-x2.5|2|model|0/2|0|2065011|31.51|31.51|
|host-chunked-toolagent-x2.5|1|model|0/1|0|2065011|31.51|31.51|
|host-chunked-toolagent-x2.5|3|model|0/3|0|2065011|31.51|31.51|
|host-muxwise-toolagent-x2.5|1|model|0/1|0|2063814|31.49|31.49|
|host-muxwise-toolagent-x2.5|0|model|0/0|0|2063814|31.49|31.49|
|host-muxwise-toolagent-x2.5|3|model|0/3|0|2063814|31.49|31.49|
|host-muxwise-toolagent-x2.5|2|model|0/2|0|2063814|31.49|31.49|
|host-muxpp-toolagent-x2.5|3|model|3/0|0|2047612|31.24|31.24|
|host-muxpp-toolagent-x2.5|1|model|1/0|0|2047612|31.24|31.24|
|host-muxpp-toolagent-x2.5|2|model|2/0|0|2047612|31.24|31.24|
|host-muxpp-toolagent-x2.5|0|model|0/0|0|2047612|31.24|31.24|
|host-pd-toolagent-x1.25|0|prefill|0/0|0|425302|25.96|25.96|
|host-pd-toolagent-x1.25|1|prefill|0/0|0|425303|25.96|25.96|
|host-pd-toolagent-x1.25|2|decode|0/0|0|425302|25.96|25.96|
|host-pd-toolagent-x1.25|3|decode|0/0|0|425302|25.96|25.96|
|host-chunked-toolagent-x1.25|0|model|0/0|0|2065011|31.51|31.51|
|host-chunked-toolagent-x1.25|3|model|0/3|0|2065011|31.51|31.51|
|host-chunked-toolagent-x1.25|2|model|0/2|0|2065011|31.51|31.51|
|host-chunked-toolagent-x1.25|1|model|0/1|0|2065011|31.51|31.51|
|host-muxwise-toolagent-x1.25|2|model|0/2|0|2063814|31.49|31.49|
|host-muxwise-toolagent-x1.25|1|model|0/1|0|2063814|31.49|31.49|
|host-muxwise-toolagent-x1.25|3|model|0/3|0|2063814|31.49|31.49|
|host-muxwise-toolagent-x1.25|0|model|0/0|0|2063814|31.49|31.49|
|host-muxpp-toolagent-x1.25|2|model|2/0|0|2047612|31.24|31.24|
|host-muxpp-toolagent-x1.25|3|model|3/0|0|2047612|31.24|31.24|
|host-muxpp-toolagent-x1.25|0|model|0/0|0|2047612|31.24|31.24|
|host-muxpp-toolagent-x1.25|1|model|1/0|0|2047612|31.24|31.24|

## 计量定义与限制

TTFT缺失计+∞，显示>cutoff是缺失哨兵，不是确知真实延迟。TPOT=(响应完成finished_at−首token时刻)/(输出token数−1)，仅统计成功且输出至少2 token。窗口输出吞吐按实际已接收token；输入吞吐在首token计入输入长度，包含缓存命中，不代表实际prefill计算量。排空分母沿用最后终结/截止定义，可能短于名义到达窗口。积压为客户端已计划到达而未终结请求，不等于GPU内部队列。

单次短窗口不能证明稳态容量或统计显著性。A100计算和内存未变；共享主机和GPU4–7其他任务可能干扰；CPU绑定由每个作业的numactl命令实施，额外的进程/线程直接采样从第二个作业中途开始，首项PD没有直接采样证据，不声称全生命周期所有线程均被连续验证；历史对照运行时段不同。PD MiniLB随机配对未固定种子。没有普通PP4对照，不能将拓扑收益全归于Mux调度。严格跨系统数值等价未验证，原始token-ID完整保留。

当前共计划1600请求，完成1600、失败0、截止未完成0、未发送0。

完整配置见frozen-configurations.csv；完整命令见complete-commands.sh、runs/*/command.json、measurements/*/run.json。校准和P2P跳过依据见calibration/path-verdict.json；所有原始记录、资源审计和失败均保留。

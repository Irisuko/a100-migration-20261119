# 受限边缘互联：完整实验结果

独立目录：`/data01/home/mhy/workspace/experiment-reports/edge8b-20261117T161715Z`。当前已计量 48/48 个预声明条件；正式主表每条件保留一次；用户中断后的整组重跑预先指定，历史结果另列，不按性能选择。

本轮模拟每卡一个边缘节点，单卡所有连接共享全双工发送/接收上限。25 Gbit/s=3.125 GB/s，10 Gbit/s=1.25 GB/s，单位为十进制。GPU仍为A100，未模拟边缘计算、内存、附加传播延迟或丢包。使用私有网络命名空间和内核TBF队列；没有修改宿主机网卡或NVLink硬件。

TP/PP使用NCCL Socket，PD使用Mooncake TCP。关闭NVLink/PCIe直连、共享内存跨rank数据通道和custom allreduce，是为了保证卡间通信穿过模拟链路；计算、分块与重叠策略保留。对照包含相同TCP路径的不限速档。历史NVLink结果仅作另一次运行的参考，不能把它与本轮差值全部归因于带宽。

## 冻结配置

|系统|架构|公共参数|框架参数|
|---|---|---|---|
|Chunked|TP4|BF16；running128；graph16；mem0.85|prefill chunk2048；原生overlap|
|MuxWise|TP4|同上|MLP4096；SM B [(84,24,1),(76,32,8),(60,48,16)]；groups5；Mux重叠|
|MuxWise-PP|PP4|同上|MLP4096；PPchunk8192；groups8；tpot_coeff0.85；inflight65536；refill16384；budget16384；prefill inflight3；Mux重叠|
|PD|2P+2D，各TP1|同上（每实例）|prefill chunk4096；原生overlap；原有MiniLB配对；Decode按原框架默认关闭radix|

GPU0–3，CPU0–31,64–95、NUMA0，全程串行。四个系统均直接复用上一轮选定c0配置，新增调参次数均为0。正式trace字节与上轮一致：conversation65请求、toolagent135请求，源连续窗口[3000,3020)，统一预过滤input+output>32768；时间伸缩2.5/1.25对应50/25秒到达窗口。每条件预热2请求，最后发送后允许120秒排空。

## 带宽与传输校准

|档位|CPU单流 Gbit/s|CPU并发出口合计 Gbit/s|PD GPU KV两对传输 Gbit/s|
|---|---|---|---|
|unlimited|43.03|65.83|12.60/12.07|
|25g|23.69|23.18|10.77/12.07|
|10g|9.51|9.46|9.50/9.35|

CPU初始校准每次512 MiB，检查逐字节正确性；TP allreduce、PP send/recv、CUDA Graph和远程控制广播均检查数据。PD每发送卡传256 MiB并检查目标GPU数据。25G档PD通道可能受CPU中转限制，配置线速不等于应用有效吞吐。实际每条件四卡发送/接收字节、双向队列限速次数和丢包见network-traffic.csv；该网络快照区间包括客户端的2次预热，平均网络速率不等于正式计量区间的瞬时峰值。

## 延迟、联合SLO和结果记账

TTFT单位秒，TPOT单位毫秒，分位数按p50/p95/p99排列。联合SLO为同一请求TTFT≤5秒且TPOT≤100毫秒；分母是全部计划请求。

恢复前额外校准使用每条连接4 GiB负载；下表发送/接收速率分别为各节点中的最大值，双向速率分别受限，不将二者相加当作单向上限。

|场景|配置Gbit/s|连接数|秒|最大节点TX Gbit/s|最大节点RX Gbit/s|数据校验|
|---|---|---|---|---|---|---|
|25g-pair|25|1|1.50|22.88|22.88|True|
|25g-fanout|25|2|2.94|23.38|11.69|True|
|25g-fanin|25|2|2.94|11.70|23.40|True|
|25g-duplex|25|2|1.53|22.47|22.47|True|
|10g-pair|10|1|3.65|9.42|9.42|True|
|10g-fanout|10|2|7.25|9.48|4.74|True|
|10g-fanin|10|2|7.25|4.74|9.48|True|
|10g-duplex|10|2|3.66|9.38|9.38|True|

以下为正式服务测量：

|档位|系统|负载|窗口秒|完成/失败/未完成/未发送|TTFT发送 p50/p95/p99 s|TTFT到达 p50/p95/p99 s|TPOT p50/p95/p99 ms|TPOT样本|SLO发送/到达 %|
|---|---|---|---|---|---|---|---|---|---|
|unlimited|Chunked|conversation|50.00|65/0/0/0|19.04/46.00/56.36|19.05/46.00/56.38|240.28/386.73/437.95|65|0.00/0.00|
|unlimited|Chunked|conversation|25.00|65/0/0/0|26.27/64.36/74.20|26.28/64.37/74.23|237.94/394.00/436.14|65|0.00/0.00|
|unlimited|Chunked|toolagent|50.00|135/0/0/0|18.49/42.74/52.93|18.50/42.75/52.93|349.73/414.66/431.05|135|0.00/0.00|
|unlimited|Chunked|toolagent|25.00|135/0/0/0|29.87/63.61/74.08|29.88/63.63/74.09|349.23/434.14/457.08|135|0.00/0.00|
|unlimited|MuxWise|conversation|50.00|65/0/0/0|13.50/32.64/39.35|13.51/32.64/39.36|45.58/47.60/48.90|65|20.00/20.00|
|unlimited|MuxWise|conversation|25.00|65/0/0/0|23.61/51.08/58.21|23.61/51.08/58.21|45.98/47.57/48.87|65|7.69/7.69|
|unlimited|MuxWise|toolagent|50.00|135/0/0/0|17.80/32.85/40.46|17.81/32.86/40.47|45.46/51.08/52.83|135|6.67/6.67|
|unlimited|MuxWise|toolagent|25.00|135/0/0/0|27.16/51.60/59.37|27.17/51.61/59.37|45.77/49.87/50.85|135|6.67/6.67|
|unlimited|MuxWise-PP|conversation|50.00|65/0/0/0|1.87/4.14/4.98|1.87/4.15/4.98|29.09/53.08/58.32|65|98.46/98.46|
|unlimited|MuxWise-PP|conversation|25.00|65/0/0/0|1.97/4.91/5.21|1.97/4.91/5.22|44.08/64.44/67.01|65|98.46/98.46|
|unlimited|MuxWise-PP|toolagent|50.00|135/0/0/0|2.55/3.94/4.70|2.57/3.95/4.72|51.87/96.10/110.55|135|95.56/95.56|
|unlimited|MuxWise-PP|toolagent|25.00|135/0/0/0|2.34/4.23/5.43|2.35/4.23/5.44|59.05/100.30/119.72|135|91.85/91.85|
|unlimited|PD|conversation|50.00|65/0/0/0|1.60/5.64/7.26|1.60/5.64/7.27|13.39/15.12/17.08|65|89.23/89.23|
|unlimited|PD|conversation|25.00|65/0/0/0|2.23/5.93/8.22|2.23/5.93/8.22|15.08/18.76/22.24|65|89.23/89.23|
|unlimited|PD|toolagent|50.00|135/0/0/0|3.48/8.48/9.82|3.49/8.50/9.84|14.16/23.41/24.82|135|74.81/74.07|
|unlimited|PD|toolagent|25.00|134/1/0/0|6.62/20.75/22.68|6.63/20.76/22.69|14.49/24.88/27.67|134|30.37/30.37|
|25g|PD|conversation|50.00|65/0/0/0|1.87/6.11/7.66|1.87/6.12/7.66|13.35/15.04/16.90|65|86.15/86.15|
|25g|PD|conversation|25.00|65/0/0/0|2.47/6.72/8.73|2.48/6.72/8.73|15.33/17.37/19.54|65|87.69/87.69|
|25g|PD|toolagent|50.00|135/0/0/0|3.09/7.09/8.04|3.09/7.12/8.06|14.12/22.36/24.63|135|77.78/77.78|
|25g|PD|toolagent|25.00|135/0/0/0|6.25/17.00/19.08|6.26/17.01/19.08|14.73/23.66/25.88|135|35.56/35.56|
|25g|MuxWise-PP|conversation|50.00|65/0/0/0|1.94/4.37/5.16|1.94/4.37/5.17|31.12/52.75/59.10|65|96.92/96.92|
|25g|MuxWise-PP|conversation|25.00|65/0/0/0|2.38/5.17/5.82|2.38/5.17/5.82|45.33/62.61/73.52|65|92.31/92.31|
|25g|MuxWise-PP|toolagent|50.00|135/0/0/0|2.66/4.12/4.97|2.67/4.14/4.99|53.30/108.53/130.92|135|91.11/91.11|
|25g|MuxWise-PP|toolagent|25.00|135/0/0/0|2.37/4.39/5.60|2.38/4.39/5.61|58.68/95.62/112.20|135|94.07/94.07|
|25g|MuxWise|conversation|50.00|48/0/17/0|44.78/102.36/>cutoff|44.78/102.36/>cutoff|168.73/174.49/178.19|48|0.00/0.00|
|25g|MuxWise|conversation|25.00|41/0/24/0|57.83/>cutoff/>cutoff|57.83/>cutoff/>cutoff|168.37/174.93/179.47|41|0.00/0.00|
|25g|MuxWise|toolagent|50.00|116/0/19/0|52.29/101.06/116.40|52.29/101.06/116.40|170.38/186.37/189.29|116|0.00/0.00|
|25g|MuxWise|toolagent|25.00|104/0/31/0|60.65/119.81/>cutoff|60.65/119.85/>cutoff|168.73/184.10/190.01|104|0.00/0.00|
|25g|Chunked|conversation|50.00|18/0/47/0|52.51/106.33/>cutoff|52.51/106.34/>cutoff|656.33/677.49/689.25|18|0.00/0.00|
|25g|Chunked|conversation|25.00|15/0/50/0|63.77/>cutoff/>cutoff|63.77/>cutoff/>cutoff|656.72/669.58/671.76|15|0.00/0.00|
|25g|Chunked|toolagent|50.00|86/0/49/0|48.80/105.02/>cutoff|48.80/105.04/>cutoff|658.97/693.92/711.77|86|0.00/0.00|
|25g|Chunked|toolagent|25.00|74/0/61/0|60.96/>cutoff/>cutoff|61.00/>cutoff/>cutoff|659.54/690.98/729.80|74|0.00/0.00|
|10g|MuxWise-PP|conversation|50.00|65/0/0/0|2.29/5.12/6.19|2.29/5.12/6.19|35.44/56.14/63.71|65|93.85/93.85|
|10g|MuxWise-PP|conversation|25.00|65/0/0/0|2.94/6.90/8.32|2.95/6.90/8.32|50.23/63.90/81.56|65|80.00/80.00|
|10g|MuxWise-PP|toolagent|50.00|135/0/0/0|2.63/4.92/5.87|2.63/4.94/5.88|50.24/111.64/131.03|135|88.15/88.15|
|10g|MuxWise-PP|toolagent|25.00|135/0/0/0|3.31/5.46/7.61|3.32/5.47/7.62|64.68/114.04/140.71|135|82.96/82.96|
|10g|Chunked|conversation|50.00|4/0/61/0|>cutoff/>cutoff/>cutoff|>cutoff/>cutoff/>cutoff|1447.55/1457.23/1457.50|4|0.00/0.00|
|10g|Chunked|conversation|25.00|4/0/61/0|>cutoff/>cutoff/>cutoff|>cutoff/>cutoff/>cutoff|1429.66/1452.07/1454.12|4|0.00/0.00|
|10g|Chunked|toolagent|50.00|37/0/98/0|>cutoff/>cutoff/>cutoff|>cutoff/>cutoff/>cutoff|1449.27/1469.50/1474.12|37|0.00/0.00|
|10g|Chunked|toolagent|25.00|25/0/110/0|>cutoff/>cutoff/>cutoff|>cutoff/>cutoff/>cutoff|1454.56/1465.83/1466.41|25|0.00/0.00|
|10g|PD|conversation|50.00|65/0/0/0|2.72/8.49/10.38|2.73/8.50/10.38|13.13/14.97/15.16|65|78.46/78.46|
|10g|PD|conversation|25.00|65/0/0/0|4.79/17.94/20.42|4.80/17.95/20.43|13.69/15.74/17.12|65|52.31/52.31|
|10g|PD|toolagent|50.00|135/0/0/0|6.09/19.20/22.71|6.10/19.21/22.72|13.40/22.32/23.26|135|40.74/40.00|
|10g|PD|toolagent|25.00|135/0/0/0|13.50/32.30/34.73|13.50/32.31/34.74|13.82/22.19/26.61|135|16.30/16.30|
|10g|MuxWise|conversation|50.00|12/0/53/0|>cutoff/>cutoff/>cutoff|>cutoff/>cutoff/>cutoff|457.99/481.66/491.57|12|0.00/0.00|
|10g|MuxWise|conversation|25.00|10/0/55/0|>cutoff/>cutoff/>cutoff|>cutoff/>cutoff/>cutoff|465.00/497.14/507.78|10|0.00/0.00|
|10g|MuxWise|toolagent|50.00|44/0/91/0|>cutoff/>cutoff/>cutoff|>cutoff/>cutoff/>cutoff|451.33/516.67/529.94|44|0.00/0.00|
|10g|MuxWise|toolagent|25.00|31/0/104/0|>cutoff/>cutoff/>cutoff|>cutoff/>cutoff/>cutoff|428.19/500.59/526.29|31|0.00/0.00|

## Goodput与输出吞吐

每对数值按窗口/排空排列；goodput只计SLO通过且在对应边界前完成的请求。

|档位|系统|负载|窗口秒|发送goodput tok/s 窗/排|到达goodput tok/s 窗/排|发送goodput req/s 窗/排|到达goodput req/s 窗/排|输出吞吐 tok/s 窗/排|排空总时长秒|
|---|---|---|---|---|---|---|---|---|---|
|unlimited|Chunked|conversation|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|40.88/138.41|134.39|
|unlimited|Chunked|conversation|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|24.80/144.75|128.50|
|unlimited|Chunked|toolagent|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|49.10/147.41|134.20|
|unlimited|Chunked|toolagent|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|31.68/151.66|130.45|
|unlimited|MuxWise|conversation|50.00|54.58/27.67|54.58/27.67|0.26/0.13|0.26/0.13|178.66/188.60|98.63|
|unlimited|MuxWise|conversation|25.00|16.72/14.88|16.72/14.88|0.12/0.05|0.12/0.05|135.84/195.63|95.08|
|unlimited|MuxWise|toolagent|50.00|29.50/14.50|29.50/14.50|0.18/0.09|0.18/0.09|171.16/194.41|101.76|
|unlimited|MuxWise|toolagent|25.00|36.92/15.05|36.92/15.05|0.32/0.09|0.32/0.09|134.96/201.90|97.98|
|unlimited|MuxWise-PP|conversation|50.00|221.06/303.56|221.06/303.56|0.94/1.07|0.94/1.07|295.56/312.10|59.60|
|unlimited|MuxWise-PP|conversation|25.00|86.84/387.26|86.84/387.26|0.80/1.37|0.80/1.37|310.44/398.16|46.72|
|unlimited|MuxWise-PP|toolagent|50.00|243.34/321.33|243.34/321.33|2.20/2.15|2.20/2.15|315.58/330.07|59.94|
|unlimited|MuxWise-PP|toolagent|25.00|96.84/400.56|96.84/400.56|2.80/2.62|2.80/2.62|327.32/418.28|47.30|
|unlimited|PD|conversation|50.00|296.16/275.05|296.16/275.05|1.06/0.96|1.06/0.96|335.84/309.47|60.11|
|unlimited|PD|conversation|25.00|363.12/446.06|363.12/446.06|1.60/1.56|1.60/1.56|540.92/500.10|37.19|
|unlimited|PD|toolagent|50.00|241.40/220.95|238.74/218.81|1.90/1.63|1.88/1.61|341.42/318.47|62.12|
|unlimited|PD|toolagent|25.00|199.40/100.22|199.40/100.22|1.56/0.82|1.56/0.82|364.48/393.42|50.25|
|25g|PD|conversation|50.00|268.42/279.73|268.42/279.73|0.96/0.97|0.96/0.97|336.56/323.42|57.51|
|25g|PD|conversation|25.00|408.08/433.11|408.08/433.11|1.68/1.53|1.68/1.53|530.88/498.34|37.33|
|25g|PD|toolagent|50.00|252.32/240.63|252.32/240.63|1.98/1.74|1.98/1.74|348.56/327.79|60.35|
|25g|PD|toolagent|25.00|187.72/109.58|187.72/109.58|1.84/1.06|1.84/1.06|374.24/437.23|45.25|
|25g|MuxWise-PP|conversation|50.00|220.48/300.83|220.48/300.83|0.90/1.05|0.90/1.05|293.68/309.55|60.09|
|25g|MuxWise-PP|conversation|25.00|69.84/368.92|69.84/368.92|0.72/1.27|0.72/1.27|306.72/392.62|47.38|
|25g|MuxWise-PP|toolagent|50.00|242.36/310.69|242.36/310.69|2.08/2.02|2.08/2.02|309.80/324.29|61.00|
|25g|MuxWise-PP|toolagent|25.00|97.08/382.52|97.08/382.52|2.88/2.61|2.88/2.61|322.96/406.80|48.63|
|25g|MuxWise|conversation|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|43.06/92.94|165.04|
|25g|MuxWise|conversation|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|23.60/94.07|142.52|
|25g|MuxWise|toolagent|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|46.02/98.24|165.01|
|25g|MuxWise|toolagent|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|28.04/98.08|142.54|
|25g|Chunked|conversation|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|13.86/40.68|165.04|
|25g|Chunked|conversation|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|8.24/37.71|142.53|
|25g|Chunked|toolagent|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|17.50/46.20|165.01|
|25g|Chunked|toolagent|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|10.12/42.96|142.55|
|10g|MuxWise-PP|conversation|50.00|193.20/292.89|193.20/292.89|0.80/1.00|0.80/1.00|275.32/304.87|61.01|
|10g|MuxWise-PP|conversation|25.00|58.56/297.89|58.56/297.89|0.68/1.05|0.68/1.05|268.04/375.49|49.54|
|10g|MuxWise-PP|toolagent|50.00|221.96/303.37|221.96/303.37|2.00/1.94|2.00/1.94|299.18/322.49|61.35|
|10g|MuxWise-PP|toolagent|25.00|74.00/337.41|74.00/337.41|2.44/2.26|2.44/2.26|277.84/399.25|49.55|
|10g|Chunked|conversation|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|3.22/10.53|165.03|
|10g|Chunked|conversation|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|2.04/9.79|142.53|
|10g|Chunked|toolagent|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|4.36/12.67|165.04|
|10g|Chunked|toolagent|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|2.12/11.04|142.54|
|10g|PD|conversation|50.00|256.84/230.98|256.84/230.98|0.92/0.81|0.92/0.81|331.18/296.96|62.64|
|10g|PD|conversation|25.00|311.56/209.71|311.56/209.71|1.20/0.72|1.20/0.72|420.48/393.50|47.27|
|10g|PD|toolagent|50.00|163.80/109.83|163.44/109.59|1.10/0.74|1.08/0.72|275.88/265.30|74.57|
|10g|PD|toolagent|25.00|105.72/43.46|105.72/43.46|0.88/0.36|0.88/0.36|272.00/325.33|60.81|
|10g|MuxWise|conversation|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|8.22/27.47|165.03|
|10g|MuxWise|conversation|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|5.00/25.57|142.53|
|10g|MuxWise|toolagent|50.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|10.48/26.84|165.06|
|10g|MuxWise|toolagent|25.00|0.00/0.00|0.00/0.00|0.00/0.00|0.00/0.00|6.24/26.30|142.53|

## 输入、总吞吐与积压

|档位|系统|负载|窗口秒|输入吞吐 tok/s 窗/排|总吞吐 tok/s 窗/排|完成请求 req/s 窗/排|窗口末积压|峰值积压|平均积压|排空末剩余|发送滞后 p50/p95/p99 s|
|---|---|---|---|---|---|---|---|---|---|---|---|
|unlimited|Chunked|conversation|50.00|4556.30/3908.39|4597.18/4046.81|0.10/0.48|60|61|27.27|0|0.00/0.01/0.04|
|unlimited|Chunked|conversation|25.00|5067.32/4087.47|5092.12/4232.22|0.08/0.51|63|64|28.58|0|0.00/0.03/0.03|
|unlimited|Chunked|toolagent|50.00|9026.04/6691.38|9075.14/6838.79|0.76/1.01|97|101|46.87|0|0.01/0.06/0.06|
|unlimited|Chunked|toolagent|25.00|7670.36/6884.04|7702.04/7035.70|0.64/1.03|119|120|55.10|0|0.01/0.01/0.02|
|unlimited|MuxWise|conversation|50.00|5643.12/5325.58|5821.78/5514.18|0.56/0.66|37|41|19.57|0|0.00/0.01/0.01|
|unlimited|MuxWise|conversation|25.00|5426.40/5524.17|5562.24/5719.80|0.28/0.68|58|58|26.09|0|0.00/0.00/0.01|
|unlimited|MuxWise|toolagent|50.00|9662.68/8824.76|9833.84/9019.17|1.36/1.33|67|78|36.77|0|0.00/0.01/0.05|
|unlimited|MuxWise|toolagent|25.00|8921.16/9164.89|9056.12/9366.79|0.96/1.38|111|113|50.59|0|0.01/0.05/0.05|
|unlimited|MuxWise-PP|conversation|50.00|10099.96/8812.92|10395.52/9125.02|0.94/1.09|18|24|11.77|0|0.00/0.01/0.01|
|unlimited|MuxWise-PP|conversation|25.00|17565.76/11242.82|17876.20/11640.98|0.80/1.39|45|49|22.40|0|0.00/0.01/0.01|
|unlimited|MuxWise-PP|toolagent|50.00|17555.08/14982.83|17870.66/15312.90|2.30/2.25|20|38|16.81|0|0.01/0.03/0.03|
|unlimited|MuxWise-PP|toolagent|25.00|30362.12/18986.69|30689.44/19404.97|3.04/2.85|59|65|32.23|0|0.01/0.01/0.02|
|unlimited|PD|conversation|50.00|9316.62/8738.54|9652.46/9048.01|1.14/1.08|8|18|7.25|0|0.00/0.01/0.01|
|unlimited|PD|conversation|25.00|15910.40/14121.35|16451.32/14621.45|1.64/1.75|24|30|14.14|0|0.00/0.01/0.01|
|unlimited|PD|toolagent|50.00|15968.22/14456.04|16309.64/14774.50|2.42/2.17|14|34|14.71|0|0.01/0.03/0.03|
|unlimited|PD|toolagent|25.00|18989.28/17502.11|19353.76/17895.53|2.88/2.67|63|69|32.82|0|0.01/0.01/0.03|
|25g|PD|conversation|50.00|9729.32/9132.47|10065.88/9455.89|1.10/1.13|10|21|7.65|0|0.00/0.01/0.01|
|25g|PD|conversation|25.00|15289.92/14071.88|15820.80/14570.22|1.76/1.74|21|32|14.07|0|0.00/0.01/0.01|
|25g|PD|toolagent|50.00|16295.56/14878.97|16644.12/15206.75|2.42/2.24|14|33|13.79|0|0.01/0.03/0.03|
|25g|PD|toolagent|25.00|19693.72/19846.73|20067.96/20283.96|3.00/2.98|60|65|31.48|0|0.01/0.01/0.01|
|25g|MuxWise-PP|conversation|50.00|10099.96/8740.84|10393.64/9050.39|0.92/1.08|19|24|12.01|0|0.00/0.01/0.01|
|25g|MuxWise-PP|conversation|25.00|16866.44/11086.60|17173.16/11479.23|0.76/1.37|46|49|22.77|0|0.00/0.00/0.01|
|25g|MuxWise-PP|toolagent|50.00|17555.08/14720.32|17864.88/15044.61|2.28/2.21|21|38|17.56|0|0.01/0.03/0.03|
|25g|MuxWise-PP|toolagent|25.00|29493.52/18465.74|29816.48/18872.54|3.04/2.78|59|65|32.44|0|0.01/0.01/0.02|
|25g|MuxWise|conversation|50.00|2713.20/3059.84|2756.26/3152.78|0.08/0.29|61|62|27.17|17|0.00/0.01/0.04|
|25g|MuxWise|conversation|25.00|2961.04/3236.15|2984.64/3330.22|0.08/0.29|63|63|28.00|24|0.00/0.02/0.02|
|25g|MuxWise|toolagent|50.00|4460.58/5319.48|4506.60/5417.73|0.38/0.70|116|116|51.64|19|0.01/0.04/0.04|
|25g|MuxWise|toolagent|25.00|4709.64/5772.88|4737.68/5870.96|0.44/0.73|124|125|56.78|31|0.00/0.01/0.04|
|25g|Chunked|conversation|50.00|2713.20/3059.85|2727.06/3100.53|0.04/0.11|63|64|28.47|47|0.01/0.04/0.04|
|25g|Chunked|conversation|25.00|2740.12/3158.13|2748.36/3195.84|0.00/0.11|65|65|28.85|50|0.01/0.03/0.03|
|25g|Chunked|toolagent|50.00|4460.58/5206.95|4478.08/5253.15|0.32/0.52|119|120|54.52|49|0.01/0.01/0.01|
|25g|Chunked|toolagent|25.00|3952.96/5363.65|3963.08/5406.61|0.20/0.52|130|131|58.53|61|0.01/0.04/0.05|
|10g|MuxWise-PP|conversation|50.00|9729.32/8608.75|10004.64/8913.62|0.82/1.07|24|28|13.40|0|0.00/0.01/0.01|
|10g|MuxWise-PP|conversation|25.00|15207.68/10602.68|15475.72/10978.16|0.72/1.31|47|52|23.49|0|0.00/0.01/0.01|
|10g|MuxWise-PP|toolagent|50.00|17184.44/14638.45|17483.62/14960.94|2.24/2.20|23|40|19.05|0|0.01/0.02/0.04|
|10g|MuxWise-PP|toolagent|25.00|27776.52/18122.68|28054.36/18521.93|2.92/2.72|62|69|35.23|0|0.01/0.02/0.04|
|10g|Chunked|conversation|50.00|1199.00/1319.94|1202.22/1330.47|0.00/0.02|65|65|28.85|61|0.01/0.03/0.03|
|10g|Chunked|conversation|25.00|1238.44/1428.88|1240.48/1438.67|0.00/0.03|65|65|28.85|61|0.00/0.03/0.03|
|10g|Chunked|toolagent|50.00|1976.48/2471.22|1980.84/2483.89|0.08/0.22|131|131|58.76|98|0.01/0.04/0.04|
|10g|Chunked|toolagent|25.00|1925.00/2404.45|1927.12/2415.49|0.04/0.18|134|134|60.14|110|0.01/0.04/0.04|
|10g|PD|conversation|50.00|9002.62/8385.33|9333.80/8682.29|1.12/1.04|9|22|8.55|0|0.00/0.01/0.01|
|10g|PD|conversation|25.00|10524.16/11111.32|10944.64/11504.82|1.44/1.38|29|35|16.20|0|0.00/0.01/0.01|
|10g|PD|toolagent|50.00|13077.82/12042.60|13353.70/12307.90|1.90/1.81|40|49|21.11|0|0.01/0.01/0.02|
|10g|PD|toolagent|25.00|14284.60/14767.51|14556.60/15092.84|2.40/2.22|75|85|39.44|0|0.01/0.05/0.05|
|10g|MuxWise|conversation|50.00|1108.96/1319.93|1117.18/1347.40|0.04/0.07|63|63|28.30|53|0.00/0.03/0.03|
|10g|MuxWise|conversation|25.00|833.12/1428.87|838.12/1454.45|0.00/0.07|65|65|28.85|55|0.00/0.03/0.03|
|10g|MuxWise|toolagent|50.00|1976.48/2431.92|1986.96/2458.76|0.18/0.27|126|127|57.87|91|0.01/0.03/0.06|
|10g|MuxWise|toolagent|25.00|1925.00/2278.28|1931.24/2304.58|0.12/0.22|132|135|60.36|104|0.01/0.03/0.05|

## 各rank显存与KV容量

nvidia-smi峰值覆盖启动、预热、计量及收尾，为1秒采样，可能漏掉瞬时峰值。PyTorch数字取现有闲时审计中的最大值；若截止后直接销毁进程，可能仅为真实峰值的下界，CSV用torch_peak_covers_measurement_end和torch_peak_scope明确覆盖范围。

|运行标签|档位|系统|GPU|角色|PP/TP rank|调度KV容量 tokens|请求槽|接收槽|nvidia-smi峰值MiB|PyTorch审计最大分配GiB|PyTorch审计最大保留GiB|审计|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|edge-unlimited-chunked-netfix1|unlimited|Chunked|0|model|0/0|2064249|128|0|70087.00|67.59|67.76|passed|
|edge-unlimited-chunked-netfix1|unlimited|Chunked|1|model|0/1|2064249|128|0|70087.00|67.59|67.76|passed|
|edge-unlimited-chunked-netfix1|unlimited|Chunked|2|model|0/2|2064249|128|0|70087.00|67.59|67.76|passed|
|edge-unlimited-chunked-netfix1|unlimited|Chunked|3|model|0/3|2064249|128|0|70087.00|67.59|67.76|passed|
|edge-unlimited-muxwise|unlimited|MuxWise|0|model|0/0|2063052|128|0|78135.00|69.72|75.52|passed|
|edge-unlimited-muxwise|unlimited|MuxWise|1|model|0/1|2063052|128|0|78135.00|69.72|75.52|passed|
|edge-unlimited-muxwise|unlimited|MuxWise|2|model|0/2|2063052|128|0|78135.00|69.72|75.52|passed|
|edge-unlimited-muxwise|unlimited|MuxWise|3|model|0/3|2063052|128|0|78135.00|69.72|75.52|passed|
|edge-unlimited-muxpp|unlimited|MuxWise-PP|0|model|0/0|2046851|128|0|73775.00|68.68|71.25|passed|
|edge-unlimited-muxpp|unlimited|MuxWise-PP|1|model|1/0|2046851|128|2|72493.00|67.90|70.00|passed|
|edge-unlimited-muxpp|unlimited|MuxWise-PP|2|model|2/0|2046851|128|2|72271.00|67.90|69.78|passed|
|edge-unlimited-muxpp|unlimited|MuxWise-PP|3|model|3/0|2046851|128|2|73197.00|68.88|70.69|passed|
|edge-unlimited-pd|unlimited|PD|0|prefill|0/0|425302|128|0|70507.00|67.92|68.32|failed|
|edge-unlimited-pd|unlimited|PD|1|prefill|0/0|425303|128|0|70603.00|67.91|68.41|failed|
|edge-unlimited-pd|unlimited|PD|2|decode|0/0|425303|128|0|69825.00|67.50|67.64|failed|
|edge-unlimited-pd|unlimited|PD|3|decode|0/0|425303|128|0|69825.00|67.49|67.64|failed|
|edge-25g-pd|25g|PD|0|prefill|0/0|425303|128|0|70573.00|67.92|68.38|passed|
|edge-25g-pd|25g|PD|1|prefill|0/0|425302|128|0|70537.00|67.92|68.35|passed|
|edge-25g-pd|25g|PD|2|decode|0/0|425303|128|0|69825.00|67.50|67.64|passed|
|edge-25g-pd|25g|PD|3|decode|0/0|425303|128|0|69825.00|67.50|67.64|passed|
|edge-25g-muxpp|25g|MuxWise-PP|0|model|0/0|2046851|128|0|73779.00|68.70|71.26|passed|
|edge-25g-muxpp|25g|MuxWise-PP|1|model|1/0|2046851|128|2|72585.00|67.92|70.09|passed|
|edge-25g-muxpp|25g|MuxWise-PP|2|model|2/0|2046851|128|2|72147.00|67.92|69.66|passed|
|edge-25g-muxpp|25g|MuxWise-PP|3|model|3/0|2046851|128|2|73203.00|68.89|70.69|passed|
|edge-25g-muxwise-resume1|25g|MuxWise|0|model|0/0|2063052|128|0|78713.00|69.62|76.08|failed|
|edge-25g-muxwise-resume1|25g|MuxWise|1|model|0/1|2063052|128|0|78713.00|69.62|76.08|failed|
|edge-25g-muxwise-resume1|25g|MuxWise|2|model|0/2|2063052|128|0|78713.00|69.62|76.08|failed|
|edge-25g-muxwise-resume1|25g|MuxWise|3|model|0/3|2063052|128|0|78713.00|69.62|76.08|failed|
|edge-25g-chunked|25g|Chunked|0|model|0/0|2064249|128|0|70029.00|67.49|67.63|failed|
|edge-25g-chunked|25g|Chunked|1|model|0/1|2064249|128|0|70029.00|67.49|67.63|failed|
|edge-25g-chunked|25g|Chunked|2|model|0/2|2064249|128|0|70029.00|67.49|67.63|failed|
|edge-25g-chunked|25g|Chunked|3|model|0/3|2064249|128|0|70029.00|67.49|67.63|failed|
|edge-25g-chunked-isolated-conversation-x1.25|25g|Chunked|0|model|0/0|2064249|128|0|70027.00|67.53|67.72|failed|
|edge-25g-chunked-isolated-conversation-x1.25|25g|Chunked|1|model|0/1|2064249|128|0|70025.00|67.53|67.72|failed|
|edge-25g-chunked-isolated-conversation-x1.25|25g|Chunked|2|model|0/2|2064249|128|0|70025.00|67.53|67.72|failed|
|edge-25g-chunked-isolated-conversation-x1.25|25g|Chunked|3|model|0/3|2064249|128|0|70025.00|67.53|67.72|failed|
|edge-25g-chunked-isolated-toolagent-x2.5|25g|Chunked|0|model|0/0|2064249|128|0|70049.00|67.54|67.73|failed|
|edge-25g-chunked-isolated-toolagent-x2.5|25g|Chunked|1|model|0/1|2064249|128|0|70049.00|67.54|67.73|failed|
|edge-25g-chunked-isolated-toolagent-x2.5|25g|Chunked|2|model|0/2|2064249|128|0|70049.00|67.54|67.73|failed|
|edge-25g-chunked-isolated-toolagent-x2.5|25g|Chunked|3|model|0/3|2064249|128|0|70049.00|67.54|67.73|failed|
|edge-25g-chunked-isolated-toolagent-x1.25|25g|Chunked|0|model|0/0|2064249|128|0|70047.00|67.54|67.72|failed|
|edge-25g-chunked-isolated-toolagent-x1.25|25g|Chunked|1|model|0/1|2064249|128|0|70047.00|67.54|67.72|failed|
|edge-25g-chunked-isolated-toolagent-x1.25|25g|Chunked|2|model|0/2|2064249|128|0|70047.00|67.54|67.72|failed|
|edge-25g-chunked-isolated-toolagent-x1.25|25g|Chunked|3|model|0/3|2064249|128|0|70047.00|67.54|67.72|failed|
|edge-10g-muxpp|10g|MuxWise-PP|0|model|0/0|2046851|128|0|73721.00|68.70|71.20|passed|
|edge-10g-muxpp|10g|MuxWise-PP|1|model|1/0|2046851|128|2|72631.00|67.92|70.13|passed|
|edge-10g-muxpp|10g|MuxWise-PP|2|model|2/0|2046851|128|2|71939.00|67.92|69.46|passed|
|edge-10g-muxpp|10g|MuxWise-PP|3|model|3/0|2046851|128|2|73305.00|68.89|70.79|passed|
|edge-10g-chunked-isolated-conversation-x2.5|10g|Chunked|0|model|0/0|2064249|128|0|69961.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-conversation-x2.5|10g|Chunked|1|model|0/1|2064249|128|0|69961.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-conversation-x2.5|10g|Chunked|2|model|0/2|2064249|128|0|69961.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-conversation-x2.5|10g|Chunked|3|model|0/3|2064249|128|0|69961.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-conversation-x1.25|10g|Chunked|0|model|0/0|2064249|128|0|69959.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-conversation-x1.25|10g|Chunked|1|model|0/1|2064249|128|0|69959.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-conversation-x1.25|10g|Chunked|2|model|0/2|2064249|128|0|69959.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-conversation-x1.25|10g|Chunked|3|model|0/3|2064249|128|0|69959.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-toolagent-x2.5|10g|Chunked|0|model|0/0|2064249|128|0|70007.00|67.52|67.68|failed|
|edge-10g-chunked-isolated-toolagent-x2.5|10g|Chunked|1|model|0/1|2064249|128|0|70007.00|67.52|67.68|failed|
|edge-10g-chunked-isolated-toolagent-x2.5|10g|Chunked|2|model|0/2|2064249|128|0|70007.00|67.52|67.68|failed|
|edge-10g-chunked-isolated-toolagent-x2.5|10g|Chunked|3|model|0/3|2064249|128|0|70007.00|67.52|67.68|failed|
|edge-10g-chunked-isolated-toolagent-x1.25|10g|Chunked|0|model|0/0|2064249|128|0|69977.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-toolagent-x1.25|10g|Chunked|1|model|0/1|2064249|128|0|69977.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-toolagent-x1.25|10g|Chunked|2|model|0/2|2064249|128|0|69977.00|67.51|67.65|failed|
|edge-10g-chunked-isolated-toolagent-x1.25|10g|Chunked|3|model|0/3|2064249|128|0|69977.00|67.51|67.65|failed|
|edge-10g-pd|10g|PD|0|prefill|0/0|425303|128|0|70611.00|67.92|68.42|passed|
|edge-10g-pd|10g|PD|1|prefill|0/0|425302|128|0|70459.00|67.92|68.27|passed|
|edge-10g-pd|10g|PD|2|decode|0/0|425302|128|0|69827.00|67.50|67.64|passed|
|edge-10g-pd|10g|PD|3|decode|0/0|425302|128|0|69825.00|67.50|67.64|passed|
|edge-10g-muxwise|10g|MuxWise|0|model|0/0|2063052|128|0|77815.00|69.62|75.21|failed|
|edge-10g-muxwise|10g|MuxWise|1|model|0/1|2063052|128|0|77815.00|69.62|75.21|failed|
|edge-10g-muxwise|10g|MuxWise|2|model|0/2|2063052|128|0|77815.00|69.62|75.21|failed|
|edge-10g-muxwise|10g|MuxWise|3|model|0/3|2063052|128|0|77815.00|69.62|75.21|failed|

## 每个实际分配的KV池

同一TP/PP实例的各rank容量是分片，不可相加当作请求token容量；PD的P/D池也分别解释。

|运行标签|档位|系统|GPU|角色|PP/TP rank|池序号|token容量|K GiB|V GiB|
|---|---|---|---|---|---|---|---|---|---|
|edge-unlimited-chunked-netfix1|unlimited|Chunked|1|model|0/1|0|2064249|31.50|31.50|
|edge-unlimited-chunked-netfix1|unlimited|Chunked|3|model|0/3|0|2064249|31.50|31.50|
|edge-unlimited-chunked-netfix1|unlimited|Chunked|2|model|0/2|0|2064249|31.50|31.50|
|edge-unlimited-chunked-netfix1|unlimited|Chunked|0|model|0/0|0|2064249|31.50|31.50|
|edge-unlimited-muxwise|unlimited|MuxWise|0|model|0/0|0|2063052|31.48|31.48|
|edge-unlimited-muxwise|unlimited|MuxWise|1|model|0/1|0|2063052|31.48|31.48|
|edge-unlimited-muxwise|unlimited|MuxWise|3|model|0/3|0|2063052|31.48|31.48|
|edge-unlimited-muxwise|unlimited|MuxWise|2|model|0/2|0|2063052|31.48|31.48|
|edge-unlimited-muxpp|unlimited|MuxWise-PP|1|model|1/0|0|2046851|31.23|31.23|
|edge-unlimited-muxpp|unlimited|MuxWise-PP|3|model|3/0|0|2046851|31.23|31.23|
|edge-unlimited-muxpp|unlimited|MuxWise-PP|2|model|2/0|0|2046851|31.23|31.23|
|edge-unlimited-muxpp|unlimited|MuxWise-PP|0|model|0/0|0|2046851|31.23|31.23|
|edge-unlimited-pd|unlimited|PD|0|prefill|0/0|0|425302|25.96|25.96|
|edge-unlimited-pd|unlimited|PD|1|prefill|0/0|0|425303|25.96|25.96|
|edge-unlimited-pd|unlimited|PD|2|decode|0/0|0|425303|25.96|25.96|
|edge-unlimited-pd|unlimited|PD|3|decode|0/0|0|425303|25.96|25.96|
|edge-25g-pd|25g|PD|0|prefill|0/0|0|425303|25.96|25.96|
|edge-25g-pd|25g|PD|1|prefill|0/0|0|425302|25.96|25.96|
|edge-25g-pd|25g|PD|2|decode|0/0|0|425303|25.96|25.96|
|edge-25g-pd|25g|PD|3|decode|0/0|0|425303|25.96|25.96|
|edge-25g-muxpp|25g|MuxWise-PP|1|model|1/0|0|2046851|31.23|31.23|
|edge-25g-muxpp|25g|MuxWise-PP|2|model|2/0|0|2046851|31.23|31.23|
|edge-25g-muxpp|25g|MuxWise-PP|0|model|0/0|0|2046851|31.23|31.23|
|edge-25g-muxpp|25g|MuxWise-PP|3|model|3/0|0|2046851|31.23|31.23|
|edge-25g-muxwise-resume1|25g|MuxWise|0|model|0/0|0|2063052|31.48|31.48|
|edge-25g-muxwise-resume1|25g|MuxWise|2|model|0/2|0|2063052|31.48|31.48|
|edge-25g-muxwise-resume1|25g|MuxWise|1|model|0/1|0|2063052|31.48|31.48|
|edge-25g-muxwise-resume1|25g|MuxWise|3|model|0/3|0|2063052|31.48|31.48|
|edge-25g-chunked|25g|Chunked|0|model|0/0|0|2064249|31.50|31.50|
|edge-25g-chunked|25g|Chunked|3|model|0/3|0|2064249|31.50|31.50|
|edge-25g-chunked|25g|Chunked|1|model|0/1|0|2064249|31.50|31.50|
|edge-25g-chunked|25g|Chunked|2|model|0/2|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-conversation-x1.25|25g|Chunked|0|model|0/0|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-conversation-x1.25|25g|Chunked|3|model|0/3|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-conversation-x1.25|25g|Chunked|2|model|0/2|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-conversation-x1.25|25g|Chunked|1|model|0/1|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-toolagent-x2.5|25g|Chunked|2|model|0/2|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-toolagent-x2.5|25g|Chunked|0|model|0/0|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-toolagent-x2.5|25g|Chunked|1|model|0/1|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-toolagent-x2.5|25g|Chunked|3|model|0/3|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-toolagent-x1.25|25g|Chunked|3|model|0/3|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-toolagent-x1.25|25g|Chunked|1|model|0/1|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-toolagent-x1.25|25g|Chunked|2|model|0/2|0|2064249|31.50|31.50|
|edge-25g-chunked-isolated-toolagent-x1.25|25g|Chunked|0|model|0/0|0|2064249|31.50|31.50|
|edge-10g-muxpp|10g|MuxWise-PP|2|model|2/0|0|2046851|31.23|31.23|
|edge-10g-muxpp|10g|MuxWise-PP|3|model|3/0|0|2046851|31.23|31.23|
|edge-10g-muxpp|10g|MuxWise-PP|0|model|0/0|0|2046851|31.23|31.23|
|edge-10g-muxpp|10g|MuxWise-PP|1|model|1/0|0|2046851|31.23|31.23|
|edge-10g-chunked-isolated-conversation-x2.5|10g|Chunked|3|model|0/3|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-conversation-x2.5|10g|Chunked|1|model|0/1|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-conversation-x2.5|10g|Chunked|0|model|0/0|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-conversation-x2.5|10g|Chunked|2|model|0/2|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-conversation-x1.25|10g|Chunked|1|model|0/1|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-conversation-x1.25|10g|Chunked|2|model|0/2|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-conversation-x1.25|10g|Chunked|3|model|0/3|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-conversation-x1.25|10g|Chunked|0|model|0/0|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-toolagent-x2.5|10g|Chunked|3|model|0/3|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-toolagent-x2.5|10g|Chunked|2|model|0/2|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-toolagent-x2.5|10g|Chunked|0|model|0/0|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-toolagent-x2.5|10g|Chunked|1|model|0/1|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-toolagent-x1.25|10g|Chunked|3|model|0/3|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-toolagent-x1.25|10g|Chunked|2|model|0/2|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-toolagent-x1.25|10g|Chunked|1|model|0/1|0|2064249|31.50|31.50|
|edge-10g-chunked-isolated-toolagent-x1.25|10g|Chunked|0|model|0/0|0|2064249|31.50|31.50|
|edge-10g-pd|10g|PD|0|prefill|0/0|0|425303|25.96|25.96|
|edge-10g-pd|10g|PD|1|prefill|0/0|0|425302|25.96|25.96|
|edge-10g-pd|10g|PD|2|decode|0/0|0|425302|25.96|25.96|
|edge-10g-pd|10g|PD|3|decode|0/0|0|425302|25.96|25.96|
|edge-10g-muxwise|10g|MuxWise|0|model|0/0|0|2063052|31.48|31.48|
|edge-10g-muxwise|10g|MuxWise|2|model|0/2|0|2063052|31.48|31.48|
|edge-10g-muxwise|10g|MuxWise|1|model|0/1|0|2063052|31.48|31.48|
|edge-10g-muxwise|10g|MuxWise|3|model|0/3|0|2063052|31.48|31.48|

## 计量边界与可复现性

TTFT缺失（包括失败或未发送）在冻结计量规则中记为+∞，分位数显示>cutoff；这表示缺失/未达标哨兵，不是已知延迟，也不一概证明真实TTFT超过排空截止。TPOT=(响应完成finished_at时刻−首token时刻)/(输出token数−1)，响应完成由流结束/DONE记时，仅统计成功完成且输出至少2 token的请求。窗口吞吐按边界前实际收到的token计数；输入吞吐在首token到达时记入输入长度，包含缓存命中，不等于新执行的prefill计算量。排空分母为最后终结或排空截止时间，沿用既有客户端定义，可能短于预定窗口。积压是客户端可见已计划到达但未终结的请求数，包含未发送，不等于GPU队列。

单次短窗口不证明稳态饱和性能或统计显著差异；保留重叠策略不证明不同框架数值输出严格等价。GPU4–7有其他任务，整机CPU/内存与互联不是独占。TCP主机中转、软件网络、框架通信实现均影响结果。PD保留原有MiniLB随机配对，路由种子未固定，单次结果包含路由分配差异。TPOT分位数仅基于完成请求，大比例未完成时不能把较低的条件分位数解释为整体服务改善。

原始测量见measurements/；所有失败启动见receipts/和driver-*.log；限速校准见calibration/。protocol.json、frozen-configurations.json、plans/jobs.json冻结方法及顺序；source/保存独立代码；完整命令见complete-commands.sh和runs/*/command.json；初始阶段入口为tools/run_edge.py；恢复阶段入口为tools/complete_edge.py，读取plans/completion.json（接续plans/resume.json），均必须在unshare -Urn内运行。旧实验未覆盖。初次启动的共享内存localhost不兼容已改用已有远程广播路径，原始尝试与修复记录保留，计时前完成。

## 运行异常（保留并计入结果）

|档位|系统|整体审计|KV/请求槽守恒|已观察原因|错误日志标记数|
|---|---|---|---|---|---|
|unlimited|PD|failed|通过|Mooncake TCP队列满，KV传输失败|4|
|25g|MuxWise|failed|未确认|计量后异步取消尚未回收即审计；随后KV/请求槽恢复且关闭剩余为0；保留原失败审计|6|
|25g|Chunked|failed|未确认|计量截止后取消/清缓存触发原生radix状态异常；下一窗口未计时，改为独立进程重启|6|
|25g|Chunked|failed|未确认|截止尚有未完成请求；按独立窗口协议退出进程释放GPU，未声称进程内静默回收|3|
|25g|Chunked|failed|未确认|截止尚有未完成请求；按独立窗口协议退出进程释放GPU，未声称进程内静默回收|3|
|25g|Chunked|failed|未确认|截止尚有未完成请求；按独立窗口协议退出进程释放GPU，未声称进程内静默回收|3|
|10g|Chunked|failed|未确认|截止尚有未完成请求；按独立窗口协议退出进程释放GPU，未声称进程内静默回收|3|
|10g|Chunked|failed|未确认|截止尚有未完成请求；按独立窗口协议退出进程释放GPU，未声称进程内静默回收|3|
|10g|Chunked|failed|未确认|截止尚有未完成请求；按独立窗口协议退出进程释放GPU，未声称进程内静默回收|3|
|10g|Chunked|failed|未确认|截止尚有未完成请求；按独立窗口协议退出进程释放GPU，未声称进程内静默回收|3|
|10g|MuxWise|failed|通过|截止时存在未完成请求的快照；最终资源回收通过|4|

已完成计量后的回收审计失败不改写性能数据。后续entry_v3仅在所有测量结束后等待tokenizer状态清零，再执行原严格审计；计量区间和截止不变。

错误日志标记数不等于失败请求数；同一请求可能在Prefill和Decode各产生多个堆栈。失败请求数以完整性能表为准。Mooncake TCP队列满是本轮传输实现的限制，不能据此推断PD算法本身的最佳性能；本轮不根据已测结果改变传输队列参数，也不删除失败条件。

## 本轮观测结论

|档位|负载|窗口秒|最高观测到达排空goodput系统（并列全列）|最高tok/s|MuxWise-PP tok/s|MuxWise-PP SLO %|
|---|---|---|---|---|---|---|
|unlimited|conversation|50.00|MuxWise-PP|303.56|303.56|98.46|
|unlimited|conversation|25.00|PD|446.06|387.26|98.46|
|unlimited|toolagent|50.00|MuxWise-PP|321.33|321.33|95.56|
|unlimited|toolagent|25.00|MuxWise-PP|400.56|400.56|91.85|
|25g|conversation|50.00|MuxWise-PP|300.83|300.83|96.92|
|25g|conversation|25.00|PD|433.11|368.92|92.31|
|25g|toolagent|50.00|MuxWise-PP|310.69|310.69|91.11|
|25g|toolagent|25.00|MuxWise-PP|382.52|382.52|94.07|
|10g|conversation|50.00|MuxWise-PP|292.89|292.89|93.85|
|10g|conversation|25.00|MuxWise-PP|297.89|297.89|80.00|
|10g|toolagent|50.00|MuxWise-PP|303.37|303.37|88.15|
|10g|toolagent|25.00|MuxWise-PP|337.41|337.41|82.96|

48个条件共计划4800请求；成功3868，失败1，排空截止未完成931，未发送0。所有负载均保留；排序只描述本次短窗口观测。

## 用户暂停与恢复

用户暂停后保留6个完整作业的24个条件。中断的MuxWise/25G作业整体在新目录重跑；旧作业的3个已计量条件完整单列，原始文件未覆盖。替换规则在新测量开始前冻结，不比较新旧结果择优。Chunked/25G首窗口截止后的取消触发原生缓存异常，下一窗口尚未计时；保留该首窗口，其他受限带宽Chunked窗口使用独立服务进程，在冻结计量结束后释放进程，不改原生overlap或负载。各窗口仍执行相同预热及缓存清理。主表48个条件跨两个执行时段，共享主机任务状态不同，存在跨时段环境变化的限制。

|历史档位|系统|负载|窗口秒|完成|未完成|到达SLO %|到达排空goodput tok/s|计量状态|
|---|---|---|---|---|---|---|---|---|
|25g|MuxWise|conversation|50.00|47|18|0.00|0.00|completed|
|25g|MuxWise|conversation|25.00|42|23|0.00|0.00|completed|
|25g|MuxWise|toolagent|50.00|117|18|0.00|0.00|completed|
|25g|MuxWise|toolagent|25.00|—|—|—|—|not_measured|

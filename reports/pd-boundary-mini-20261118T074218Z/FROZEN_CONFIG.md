# 冻结配置清单

执行前 `frozen-pins.json` 固定 1,142 个运行源码、配置、输入、作业计划及执行工具文件；最终逐项哈希核验通过。以下为便于阅读的清单，实际生效值以 `effective-configurations.json` 和每个 worker 的 `server-final.json` 为准。后处理报告不属于运行时冻结源码。

| 项目 | MuxWise-PP | PD |
|---|---|---|
| 模型/精度 | Llama-3.1-8B-Instruct / BF16 | 相同 |
| GPU 池 | GPU0–3，1 实例，PP=4、TP=1 | GPU0/1 为 P，GPU2/3 为 D，4 实例各 TP=1、PP=1 |
| CPU/NUMA | CPU0–31,64–95 / NUMA0 | 相同 |
| max_running_requests | 128 | 每实例128；每条件请求总数≤60，全局也不超过128 |
| cuda_graph_max_bs | 16 | 16 |
| mem_fraction_static | 0.85 | 0.85 |
| context_length | 33792 | 33792 |
| attention / sampling | flashinfer / pytorch | 相同 |
| KV dtype | auto，BF16 模型对应缓存 | 相同 |
| MLP 分块 | 4096 | 不适用 |
| PP Prefill 分块 | 8192 | 不适用 |
| sm_group_num | 8 | 不使用 Mux 调度 |
| tpot_coeff | 0.85 | 不适用 |
| max_inflight_tokens | 65536 | 不适用 |
| token_refill_size / max_budget | 16384 / 16384 | 不适用 |
| max_inflight_prefill | 3 | 不适用 |
| PD Prefill 分块 | 不适用 | P=4096；D 不执行 chunked Prefill |
| 重叠机制 | 保留 Mux 自身重叠；通用 overlap scheduler 关闭 | 原生 overlap scheduler 开启 |
| 传输 | A/B/C 原生；D1/D2 TCP | A/B/C 原生 Mooncake；D1/D2 强制 Mooncake TCP |
| 测量重复/新增调参 | 1 / 0 | 1 / 0 |

各条件串行、部署独立重启，先预检及清缓存，再相同固定短 warmup 和冻结输入。每系统使用同一条件的逐字节相同 trace；D1/D2 也使用同一 trace。忽略 EOS、greedy、输出长度固定。PD 保留原 MiniLB 独立随机 P/D 路由，未额外固定该路由的随机种子；本轮服务 random_seed=20261114 不等于固定路由。

| 条件 | 输入/输出 token | 请求数 | 到达窗口秒 | 到达率 req/s | 传输 | 最后实际发送后的排空上限秒 |
|---|---|---|---|---|---|---|
| A1 | 16384/32 | 45 | 30 | 1.5 | 原生 | 120 |
| A2 | 24576/32 | 45 | 30 | 1.5 | 原生 | 120 |
| B1 | 128/2048 | 60 | 30 | 2 | 原生 | 240 |
| B2 | 128/4096 | 60 | 30 | 2 | 原生 | 445 |
| C1 | 24576/4096 | 24 | 2 | 12 | 原生 | 445 |
| C2 | 24576/4096 | 48 | 2 | 24 | 原生 | 445 |
| D1 | 24576/128 | 15 | 30 | 0.5 | TCP 不限速 | 120 |
| D2 | 24576/128 | 15 | 30 | 0.5 | TCP 每 GPU 共享 TX 10 Gbit/s、RX 10 Gbit/s | 120 |

到达时间 j/rate，j 从0开始；不是随机到达。发送窗口结束后继续排空；本轮全部在截止前完成，未触发截断。各档依次交换系统执行顺序：A1 PD→PP，A2 PP→PD，B1 PD→PP，B2 PP→PD，C1 PD→PP，C2 PP→PD，D1 PD→PP，D2 PP→PD。

输入来源及种子更正见 `environment/execution-rulings.json`。与原始计划的真实文本设想不同，本轮实际是合成 token-ID 固定形状负载。所有结果保留，不基于结果更换工作负载。

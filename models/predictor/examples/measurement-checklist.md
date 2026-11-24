# 真实 H800 基线收集任务卡

> 这是**未来数据要求**，不是已采集结果。不要在公开仓库发布生产 prompt、模型权重、内部 IP 或未脱敏日志。

## 1. 固定比较域

- environment ID / 推理引擎 commit / 模型 ID + revision：
- H800 数量、节点数、TP/EP/DP、GPU 互联及 NUMA：
- 当前 weight、compute、KV/indexer、recurrent state、PLE dtype：
- workload ID 和 trace hash，采样、thinking、缓存冷热：
- 输入/实际输出长度分布；不要只记录 max_tokens：

## 2. 低负载时间账

- TTFT **mean** ms、TPOT **mean** ms（同一请求群）：
- Prefill **mean** ms、first_decode **mean** ms：
- 其余 TTFT（前处理、排队、网络等，以配对均值求差）：
- 每个 block：串行 / 计算 / 通信 ms、真实重叠比例、可验证的下界：
- 对齐 CPU/GPU timeline，避免 CPU launch 与 GPU kernel 计时重复计费。

## 3. 饱和吞吐账

- 同一 workload 的饱和输出 tok/s、并发/RPS 扫描与时间窗口：
- GPU compute、HBM、通信、CPU 的资源繁忙比例及计数器来源：
- 候选每输出 token 的资源需求因子；未知就记 **unknown**，不要把 1.0 假装成实测。
- 单实例 / 每卡 / 集群总吞吐分开记录；保持总 GPU 预算不变。

## 4. 单变量对照与预测误差

- Trick ID、代码 commit、唯一变量、正确性、质量和资源副作用：
- 预测的保守 / 中心 / 乐观假设：
- A/B 实测 TTFT / TPOT / 容量，重复次数和不同负载：
- 填写 scenario 的 observed，记录带符号误差；检查错在可消除比例、重叠、额外开销、资源因子还是调度。

预测器只给出**低负载均值**、**饱和输出容量**和**负载压力**。正式报告还必须遵循 [Benchmark 协议](../../../docs/benchmark-protocol.md) 报 TTFT / TPOT p50/p95/p99、显存、失败率和质量。

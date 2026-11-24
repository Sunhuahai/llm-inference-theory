# Notation

| 符号 | 含义 / 单位 |
| --- | --- |
| F | 浮点计算量，FLOPs |
| D | 指定存储层级数据流量，Bytes |
| C | 指定通信路径数据量，Bytes |
| M | 峰值状态与工作内存，Bytes |
| R | 串行执行或验证轮数 |
| P | 对应精度峰值计算速度，FLOPs/s |
| B | 对应存储层级带宽，Bytes/s |
| I | 算术强度 F/D，FLOPs/Byte |
| T | 时间，s；报告可换算 ms |
| b | 批规模，注明每卡或全局 |
| λ, μ | 请求到达率、服务率，requests/s |
| TTFT | 首 token 时延 |
| TPOT | 输出 token 平均间隔（注明统计口径） |
| Throughput | 单位时间有效输出或总 token 数；分别统计 |
| Goodput | 单位时间满足服务 SLO 的有效请求数 |

不同符号在章节内若有局部含义应明确定义。所有资源下界须指定精度、存储层级和并行路径。评价口径详见 [Benchmark 协议](benchmark-protocol.md)。

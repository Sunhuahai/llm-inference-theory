# 每晚性能反馈

日期按 Asia/Shanghai 填写。可以先填已知项；未知写 `unknown`。本模板不包含任何实测结果。

## 1. 本次试验

- 日期：
- trick ID：
- baseline run ID / candidate run ID：
- 证据来源：user_report / agent_measured
- 运行配置 commit / 代码改动链接：
- 环境 env_id / 负载 workload_id / 原始结果链接：
- 唯一改动及实际生效证据：

## 2. 环境与负载

- 完整模型 ID + revision、8-bit 格式：
- H800 总卡数、节点数、TP/EP/DP 实际布局：
- 引擎版本、benchmark 工具和版本：
- 文本 / 图片 / 视频 / 音频 track：
- 输入与实际输出长度分布（tokens）、thinking/EOS/stop：
- 闭环并发，或开环 offered / actual sent RPS：
- cold-prefix / warm-prefix、命中率、预热方式：
- 重复次数、超时/重试设置：

## 3. Baseline → Candidate

| 指标 | Baseline | Candidate |
|---|---:|---:|
| TTFT p50 / p95 / p99（ms，含 reasoning 首 token） | | |
| user_visible_ttft p95（ms，如适用） | | |
| TPOT p50 / p95 / p99（ms/token） | | |
| stream gap p99（ms；不能当作 MTP 的每 token 时延） | | |
| 输出吞吐量（output tokens/s） | | |
| 总 token 吞吐量（input + output tokens/s） | | |
| E2E p95（ms） | | |
| goodput（requests/s；填写预设 SLO 后才有效） | | |
| sent / success / error / timeout 数量 | | |
| 各卡峰值显存的最大值（GiB） | | |

- TTFT / E2E SLO（ms）和 TPOT SLO（ms/token），未知则不评价 goodput：
- 测量时长、实际输入/输出 token 总数：
- MTP 首 chunk 是否含多个 token，逐 chunk token 数是否可得：
- 质量/功能回归：方法、结果、是否通过：
- 是否存在 OOM、重试、fallback、排队上升或客户端发不满负载：

## 4. 现象与下一轮

- 最明显的现象：
- 重复波动、未解决疑问：
- 下一轮可用 GPU/时间预算：
- 暂定状态：researched / ready / running / validated / inconclusive / regressed / unsupported / blocked / baseline-covered / parked

“validated”需符合[测量协议](../docs/benchmark-protocol.md)中的重复性、质量和延迟约束。不要因关闭 thinking、缩短生成、降低精度或增加 GPU 得到收益就标为同预算优化。公开提交前去除密钥、内部地址及敏感 prompt；保留可复现实验所需的脱敏配置与结果。

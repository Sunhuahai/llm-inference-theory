# 性能测量协议

状态：实验协议，2026-11-20 状态更新。用户已反馈首轮全部尝试，逐项原始结果尚未归档；后续各轮使用当前最佳有效配置作比较基线，并保留它与首轮结果的关联。协议初版：2026-11-16。

目标是在固定模型、固定精度、固定 H800 总预算和相同任务下，量化 TTFT、TPOT 与吞吐量的变化。第一轮暂以文本请求为主；若实际业务含图片、视频或音频，另建 workload track，固定媒体输入及编码/预处理范围，不能与纯文本请求混合排名。

## 1. 一次实验固定什么

每次先验证一个 trick，单项通过后再组合。环境清单记录下列内容，并给出 `env_id`；未知项写 `unknown`，不猜测。

| 类别 | 必须固定或显式记录 |
|---|---|
| 模型 | 完整 checkpoint ID、revision、tokenizer 与 chat template、现有 8-bit 格式、实际计算与 KV/state dtype |
| 硬件 | H800 总卡数、显存、节点数、GPU/CPU/NIC 拓扑、功耗设置、是否独占 |
| 软件 | 引擎版本及 commit、容器 digest、驱动/CUDA/PyTorch/通信与 kernel 依赖版本、完整脱敏启动命令 |
| 任务 | 数据集/trace hash、输入和实际输出 token 分布、采样、thinking 模式、EOS/stop、工具调用约束 |
| 测量 | benchmark 工具/版本、客户端位置、流式协议、负载、缓存状态、预热、超时、SLO |

TP/EP/DP 可以是试验变量，但**总 GPU 预算保持相同**。记录真实进程组和副本资源，不用简单的 `TP × EP × DP` 推定卡数。不得改变权重/计算/KV 精度，不用关闭 thinking、缩短输出上限、截断 prompt、移除业务功能或增加卡数冒充同预算优化。

## 2. 指标与时间点

对请求 i：`s_i` 是客户端实际发送时刻；`f_i` 是首个有效**生成内容**到达时刻；`e_i` 是最后一个有效生成内容到达时刻。`I_i`、`O_i` 分别为输入和实际生成 token 数。统一使用客户端单调时钟；空 role delta、心跳和 DONE 不算生成内容。

**TTFT 包含 reasoning token。** 如果接口先输出 reasoning，再输出最终回答，首个 reasoning token 就结束 TTFT；另记 `user_visible_ttft`，表示首个用户可见回答内容的到达时间。若接口隐藏 reasoning，只能测到可见内容时间，应将真正 TTFT 标为不可观测，不把整段 reasoning 耗时悄悄算进不同口径的 TTFT。`O_i` 必须覆盖被计时的全部生成 token；优先使用可靠的服务端 usage，并记录 tokenizer 重计数差异。vLLM 官方文档也提醒隐藏 reasoning 会改变 TTFT 的含义。[来源](https://docs.vllm.ai/en/v0.32.0/benchmarking/cli/)

| 指标 | 本仓库定义 | 单位及报告 |
|---|---|---|
| TTFT | `f_i - s_i` | ms；p50/p95/p99 |
| user_visible_ttft | 首个用户可见回答内容时刻减 `s_i` | ms；与 TTFT 分开 |
| TPOT | `(e_i - f_i)/(O_i - 1)`，仅 `O_i > 1` | ms/token；先逐请求计算再求 p50/p95/p99 |
| stream gap | 相邻有效 stream chunk 到达时间差 | ms；p50/p95/p99，反映输出停顿 |
| E2E | `e_i - s_i` | ms；p50/p95/p99 |
| 输出吞吐量 | `Σ成功请求 O_i / Δt` | output tokens/s；主吞吐量指标 |
| 总 token 吞吐量 | `Σ成功请求(I_i + O_i) / Δt` | tokens/s；辅助指标 |
| 请求吞吐量 | `成功请求数 / Δt` | requests/s |
| SLO goodput | `成功且同时满足全部指定 SLO 的请求数 / Δt` | requests/s；附各 SLO 与合格比例 |

`O_i ≤ 1` 的 TPOT 为 N/A，不填零。TTFT 是端到端指标，包含排队、处理及传输，不等于纯 GPU prefill 时间。主报告的 `Δt` 从测量批次首个请求实际发出，直到最后一个测量请求完成、失败或超时，包含排空时间；预热请求不在其中。另报稳态窗口时，分子也必须只属于该窗口。

总 token 吞吐量会随输入长度及缓存命中变化，不能替代输出吞吐量，也不能直接代表 GPU 计算效率。相同工具的 TTFT/TPOT 定义可查 [vLLM Benchmark CLI](https://docs.vllm.ai/en/v0.32.0/benchmarking/cli/)；跨工具应按公式比较，而非仅按指标名。

### MTP / speculative decoding：不能把 chunk 当 token

一个 stream chunk 可以携带多个 token。vLLM 的 ITL 记录 chunk 间隔，而 TPOT 按实际生成 token 数归一；AIPerf 的 ITL 接近 token 归一平均值，并用 ICL 单列 chunk 间隔。必须保留工具版本和公式。[vLLM](https://docs.vllm.ai/en/v0.32.0/benchmarking/cli/) · [NVIDIA AIPerf 指标说明](https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference)

主表保留上面的标准 TPOT。若首个内容 chunk 含 `m_i > 1` 个 token，且 API 能可靠返回逐 chunk token 数，补充：

```text
post_first_chunk_ms_per_token = (e_i - f_i) / (O_i - m_i)
```

仅当 `O_i > m_i` 有定义，命名为“首 chunk 后 token 归一延迟”；不能与标准 TPOT 混列。若不能取得 `m_i`，标为无法校正。保持输出流 flush/聚合策略一致，并同时报告 chunk gap，防止通过积攒 token 改善表面数值。

## 3. 最小负载矩阵

以下是**建议筛选起点，不是最佳配置或用户业务分布**。每个 trick 先选 1–2 个主场景与一个回归场景；通过后再扩展，真实 trace 到位后优先使用固定业务子集。

| Track | 输入 / 输出 token | 主要观察 |
|---|---|---|
| short-chat | 512 / 128 | 低并发延迟、CPU/launch 开销 |
| long-prefill | 8192 / 128 | TTFT、预填充调度、前缀缓存 |
| long-decode | 1024 / 1024 | TPOT、MTP、MoE 与通信 |
| long-context | 32768 / 256；确认容量和支持后 | attention/state 管理 |
| mixed | 固定短/长请求配比或真实 trace | prefill 对 decode 的干扰、尾延迟 |

实际输入长度必须包含 chat template。强制固定输出长度只用于明确标识的合成性能测试；质量集合保留正常 EOS/stop，并记录实际输出长度。

### 负载模型

- **闭环**：固定并发，请求完成后补下一个；可从 C=1 开始，再按资源试 4/16/64。用于观察延迟与饱和吞吐权衡，不代表生产固定 RPS 能力。
- **开环**：固定请求到达序列或固定种子的 Poisson 到达；baseline pilot 确定低/中/近饱和 RPS，所有方案复用同一序列。记录 offered RPS、actual sent RPS、计划到达与实际发出的偏差、队列变化。
- 开环受客户端并发上限或客户端 CPU 限制时，实际 RPS 会下降；不得据此声称系统承受了名义负载。vLLM 的 `request-rate` 与 `max-concurrency` 同用就可能产生该限制。[参数说明](https://docs.vllm.ai/en/v0.28.0/cli/bench/serve/)

`offered_rps` 是设定到达率；`actual_sent_rps` 在实际注入窗口内统计，窗口定义随 workload manifest 保存，不能拿含排空时间的请求吞吐代替。计划发送时刻为 `a_i` 时，另报 `max(0, s_i-a_i)` 的 p95；TTFT 从实际发送起算，因此仅看 TTFT 会遗漏客户端发出前积压。

### 缓存与预热

编译/Graph/kernel 预热和 prefix cache 预热分开。`cold-prefix` 在引擎预热后清理相关缓存，或使用确认无复用的固定数据；`warm-prefix` 用相同预置请求建立缓存，记录共享前缀长度和命中率。每次重复都执行同一准备过程。vLLM 官方特别提示：相同 seed 重复压测会复用 prefix cache，可能夸大性能。[来源](https://docs.vllm.ai/en/v0.32.0/benchmarking/cli/)

## 4. 执行与接受规则

1. **先验支持检查**：核对实际模型/格式/引擎路径、warnings 和 fallback；用小批请求确认完整输出及启动参数生效。当前环境未知时只生成任务书，不宣称已运行。
2. **独立预热**：完成模型加载、编译与 kernel/Graph 预热，记录预热时长，再按上节准备缓存。正式指标运行不全程开启重 profiler；必要时单独采样解释瓶颈。
3. **重复与顺序**：每点至少 3 次，交错 baseline/candidate，避免单向温度或缓存漂移。首次筛选可从每次 ≥200 个成功请求开始；这并不保证 p99 稳定，样本不足时不作强尾部结论。
4. **失败完整计数**：`sent_n = success_n + error_n`；`error_n` 包含 timeout、OOM、取消及其他失败，`timeout_n` 是其中子集。所有测量请求都应有最终状态。成功延迟统计不能掩盖错误率；不能丢弃失败请求或缩短计时窗口来改善吞吐。固定超时与重试策略，推荐禁用隐式自动重试。
5. **质量和功能回归**：固定一组小型请求，检查完整性、终止行为、NaN、结构化输出及任务得分；不要要求随机采样逐字一致。MTP 的“无损”声明必须核实实现与采样保证，并做对应回归。
6. **接受条件**：收益超过 baseline 重复波动，关键尾延迟、错误率和质量约束通过，无隐藏资源增加，配置与结果可复现。否则标为 inconclusive、regressed、unsupported 或 blocked。

用户 SLO 尚未知时，goodput 为 `unknown`，先报告 TTFT/TPOT/吞吐的 Pareto 权衡，不擅自指定业务阈值。延迟改善定义为 `(B-C)/B`，吞吐改善为 `(C-B)/B`，均乘 100%。已有质量或延迟门槛必须预先记录，不能看完结果再调整。

## 5. 文件与证据

每个 `env_id`/`workload_id` 应能定位到固定 manifest；其中保存长命令、版本、数据/采样、SLO 和缓存设置。逐次结果使用 [CSV 表头](../archive/qwen-h800-tricks/templates/result.csv)，每晚提交 [反馈模板](../archive/qwen-h800-tricks/templates/feedback.md)。CSV 一行对应一次重复，缺失指标留空；解释写 `notes`，不使用 0 表示未知。

`evidence_origin` 区分 `user_report`、`agent_measured`、`literature`（上游资料）、`hypothesis`；只有前两类进入本项目性能对照，文献结果保留其原硬件与条件。`artifact_ref` 链接原始统计/脱敏日志，`config_git_sha` 对应实际运行配置。`gpu_budget_gpus` 为该服务总卡数，`peak_gpu_mem_gib` 为各卡显存峰值的最大值；各卡明细存附件。`quality_pass` 取 true/false/unknown。

CSV 的 `slo_tpot_ms_per_token` 单位为 ms/token；TTFT/E2E SLO 单位为 ms。CSV 的 `input_tokens`、`output_tokens` 是成功请求的累计数量，与吞吐分子一致；错误请求消耗的 token 如可得，另存原始结果。`date_local` 使用 Asia/Shanghai；每请求的精确时间戳保留在附件。

先选定一套主 benchmark 工具及版本并保持一致。NVIDIA 已宣布 GenAI-Perf 不再新增功能、建议新工作采用 AIPerf；旧数据保留原工具标签，不无声混合。[NVIDIA 官方公告](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)

本文件没有 H800 实测数字。负载矩阵、样本量和流程是本仓库实验设计建议；执行 Agent 仍须以安装版本的 `--help`/源码核对参数。

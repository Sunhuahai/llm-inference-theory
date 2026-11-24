# 交给实施 Agent 的首周任务书

> **已归档（2026-11-20，Round 01）。** 生命周期：`archived`；尝试记录：`user_report / attempted`。用户明确 T001–T013 均已尝试。逐项收益、质量与实施证据尚未提交，保留为 `unknown`；下文的研究状态和“待实测”措辞是原始历史记录。不得据此重新派发，也不得推断全部成功或失败。
> 当前新增候选见 [Round 03](../../rounds/round-03/README.md)；归档说明见 [Round 01 索引](README.md)。

> **历史任务书。2026-11-20 用户已反馈首轮全部尝试。** 当前工作使用[第二轮任务书](../../weeks/2026-W47/agent-tasks-round-02.md)，沿用首轮最佳配置；下文保留初始设计，不作为从头重跑指令。

## 可直接复制的提示词

```text
请接手 qwen38-next-h800-tricks 的本周推理优化实验。

先读取仓库最新 main 的 AGENTS.md、environment.yaml、docs/model-and-support.md、
docs/benchmark-protocol.md、weeks/2026-W47/README.md 与本任务书。
目标暂定 Qwen/Qwen3.8-Flash-Next-FP8；必须用我的实际部署核对完整模型 ID、revision。
固定现有所有精度，不研究新量化，不换模型，不缩短输出或关闭所需功能来换数字。

先完成 P0：收集实际 H800 数量、显存、拓扑/NUMA、框架版本和已加载后端，
记录启动命令、权重/KV/indexer/SSM/PLE 的实际 dtype，建立最小可复现 baseline。
检查 T003/T004 所述上游融合是否已存在；存在就标 baseline-covered，不重复实现。
不要把旧 Qwen3-Next 的配置直接用于 Qwen3.8-Flash-Next；检查 FP8 TP 分片限制。

然后根据 baseline/profile，只选一个任务进行单变量 A/B：
优先 T001 或 T002；若模型路径有明确瓶颈，则按对应卡片选 T003-T006/T008。
实现到我指定的推理代码仓库；若尚未指定，只完成环境/基线盘点并报告所缺入口。
每组保持相同总 GPU 预算、checkpoint、负载、输出和冷热缓存条件，至少三次交错重复。
报告 TTFT/TPOT p50/p95/p99、output/total tokens/s、错误/超时/OOM、显存及质量回归。
不得把微基准或别的GPU数据当成H800端到端收益。

交付：配置diff、实现commit、环境与数据ID、脱敏命令、结果CSV、原始证据链接、
是否通过质量/性能约束及回滚方法。把简短反馈放到本周feedback/，结果放results/。
若无法运行或不支持，说明准确门槛/报错并记blocked或unsupported；不要伪造结果。
```

## P0 · 环境确认与 baseline（首个必做任务）

| 项目 | 交付 |
|---|---|
| 模型和框架 | 填写环境清单；确认 Qwen4Exp 模型实际加载路径 |
| 硬件 | H800 SKU/卡数/显存、实际进程组与脱敏拓扑；不假设八卡 |
| 精度与内存 | 实际 dtype、PLE offload、state/KV 池、启动余量与峰值 |
| 正确性 | 最小功能集合、完整输出、终止/错误检查与质量判据 |
| 性能 | 选短对话 + 与业务最相关场景；闭环从低并发开始，记录真实分布 |
| 支持盘点 | T002/T003/T004/T005/T008/T009 的支持/已默认开启/不支持证据 |

不要一上来扫全部 batch/并发组合；不要为了完成 P0 修改权重或新增部署规模。若没有可用 H800，只交环境阻塞项，不生成示例性能行。

## P1 · 第一项可控实验

在 [T001](tricks/T001-prefill-scheduling.md) 与 [T002](tricks/T002-cuda-graphs.md) 中选 **一项**，使用 baseline 条件测试。前者重点是混合负载下 TTFT/TPOT 权衡，后者适合 launch/主机间隙较明显的场景。先验证真实支持与生效路径。

验收不采用虚构的“必须提升 20%”。需要改善超出基线波动、质量与用户约束通过，且不隐藏卡数/输出/错误的变化；未知 SLO 时提交 Pareto 表。

## P2 · 根据 profile 选择模型专属实验

| 证据 | 优先卡片 |
|---|---|
| QSA prepare 有可消除的多次 launch | [T003](tricks/T003-qsa-prepare.md) |
| 小 batch QSA 投影 GEMM 占比高 | [T004](tricks/T004-qsa-projection.md) |
| 长 prefill GR/PLE 重复计算明显 | [T005](tricks/T005-gr-ple-sequence-parallel.md) |
| GPU 等待主机取表或 PCIe 传输 | [T006](tricks/T006-ple-prefetch.md) |
| 业务多轮/分叉前缀复用显著 | [T008](tricks/T008-hybrid-prefix-cache.md) |
| 输出阶段主导、已有原生 MTP 支持 | [T009](tricks/T009-native-mtp.md) |

### 补充 P0：Prefix/上下文专项数据包（2026-11-17）

**不要立刻做有损压缩上线。** 在已核对的 H800 环境中，先脱敏采集一份按序 trace：实际 input/output token 数、固定模板与 tokenizer 版本、带隔离域的盐化 prefix key、可复用 prefix token 数、真实 QSA/GDN cache MiB、命中层级、cache eviction / preemption、冷/热 prefill、lookup/restore 与客户端 TTFT。再准备原任务的配对压缩质量集。

先用 [`cache_context.py`](../../theory/predictor/cache_context.py) 填真实参数，对 **容量受限 LRU、无淘汰完整键、免费命中 oracle** 给出情景区间。若原 trace 无前缀复用或已有策略覆盖，就记录负结论，不强行试 T008；若业务确有冗余且质量 gate 可定义，再依据 [T013](tricks/T013-context-compression.md) 做结构去重的独立单变量 A/B。不得把压缩后的 prefix key 沿用原 key，质量未验证时不得宣称可采纳收益。

一次只实施一张卡。组合试验只在单项已解释后安排，不能把叠加效果简单相乘。

## 交付目录与记录

- `feedback/YYYY-MM-DD.md`：用户/Agent 报告与现象，注明来源。
- `results/<run_id>.csv`：使用 [统一表头](../../templates/result.csv)，不填伪数据。
- `results/<run_id>-environment.yaml`：环境快照，引用数据 ID 与 sanitized launch command。
- `results/<run_id>-summary.md`：配置差异、重复统计、质量、限制与回滚；大日志仅留脱敏证据路径。
- `decisions.md`：研究 Agent 综合反馈后更新，不覆盖旧结论。

# T005｜GR / PLE 序列并行

| 字段 | 内容 |
|---|---|
| ID / 优先级 | T005 / P2 |
| 状态 | `researched`；待 H800 实测 |
| 调研日期 | 2026-11-16 |
| 固定条件 | checkpoint、所有 dtype、TP/DP/EP、总 GPU 数、输出要求 |

## 问题与机制

张量并行下部分 Gated Residual（GR）和 PLE 计算可能在多个 rank 重复。[vLLM PR #56322](https://github.com/vllm-project/vllm/pull/56322) 引入 `--enable-hc-sp`，沿序列维拆分这些工作。上游对照同时出现长 prompt TTFT 改善和 decode 小幅退化，说明它需要按负载选择。

## 适用与版本门槛

需要包含该 PR 的实现，要求 PP=1、MoE 层；相关 SP 分片投影不支持外部 LoRA。符合特定 TP/DP/EP 与通信后端条件时，MoE SP 可自动开启，见同一 PR。先核查基线，已开启则标 `baseline-covered`。

## 交给实现 Agent

| 步骤 | 操作与交付 |
|---|---|
| 1 | 记录引擎 SHA、TP/DP/EP/PP、通信后端和 GR/PLE 的实际分片状态。 |
| 2 | 使用已支持的 `--enable-hc-sp` 做单项对照；若自动 SP 无公开关闭入口，明确记录实验 patch，避免误称普通开关对照。 |
| 3 | 校验分片与重组、非整除长度、多个请求边界，以及相同输入的数值误差。 |
| 4 | 提交通信耗时、各 rank 负载和端到端指标，区分计算节省与通信增加。 |

## 单变量 A/B

固定 TP/DP/EP 和 GPU 数，仅切换 SP。测试短 prompt、8K/32K prompt、长输出三组；保持相同并发、缓存命中、采样与输出长度。报告 TTFT/TPOT p50/p95、output tokens/s、总 tokens/s、显存，以及 collective 时间。不得把增加 GPU 数的结果列入同一对照。

## 正确性与回滚

执行对应配置和分片测试，并用固定业务样例检查输出质量。长序列收益不足以补偿目标负载的 decode 退化时，恢复原 SP 配置；若有实验 patch，一并回退。

## 收益已知与未知

上游机制与合并状态明确；H800 互联、实际 batch 下的收益未知。该条目应形成适用工作负载结论，不能登记为所有场景的默认加速。

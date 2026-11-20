# T003｜融合 QSA 准备阶段

> **已归档（2026-11-20，Round 01）。** 生命周期：`archived`；尝试记录：`user_report / attempted`。用户明确 T001–T013 均已尝试。逐项收益、质量与实施证据尚未提交，保留为 `unknown`；下文的研究状态和“待实测”措辞是原始历史记录。不得据此重新派发，也不得推断全部成功或失败。
> 当前新增候选见 [Round 03](../../../rounds/round-03/README.md)；归档说明见 [Round 01 索引](../README.md)。

| 字段 | 内容 |
|---|---|
| ID / 优先级 | T003 / P1 |
| 状态 | `researched`；待 H800 实测 |
| 调研日期 | 2026-11-16 |
| 固定条件 | checkpoint、所有 dtype、采样设置、业务输出要求 |

## 问题与机制

QSA 之前的归一化、RoPE、gate 和 KV 写入若分开执行，会增加 kernel launch 与中间张量读写。vLLM [PR #57097](https://github.com/vllm-project/vllm/pull/57097) 已将相关工作合入 `qsa_prepare`，覆盖 decode、prefill 和混合 batch，保留原有 cache 精度。它是减少执行开销的候选，不改变模型结构。

## 适用与版本门槛

适用于含该 PR 的 NVIDIA Qwen4Exp 实现；检查源码和实际 dispatch，不能只看版本号。当前基线若已默认走融合路径，登记 `baseline-covered`。上游测试硬件不能代替 H800 验证。

## 交给实现 Agent

| 步骤 | 操作与交付 |
|---|---|
| 1 | 固定引擎 SHA，检查 `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py` 及调用条件，记录是否命中融合路径。 |
| 2 | 有条件开关则复用；没有则添加仅用于实验的 fused/unfused 对照入口，提交最小 patch。不得杜撰 CLI 参数。 |
| 3 | 对照 Q、gate、KV cache 的形状、值和地址边界，覆盖 decode、长 prefill、混合 batch。 |
| 4 | 用同一 build 跑 A/B，提交配置、原始结果和一段 profiler 证据。 |

## 单变量 A/B

仅改变融合路径。先测并发 1/8 的短请求，再测 8K/32K prompt 和固定输出长度；采用同一请求顺序、缓存状态及预热策略。记录 TTFT、TPOT 的 p50/p95、output tokens/s、input+output tokens/s，并查看 launch 数与 QSA 准备阶段耗时。

## 正确性与回滚

预先约定数值误差及业务验收阈值，检查非有限值、缓存越界和生成回归。失败即回退原 dispatch/patch；保持权重、KV/indexer/SSM/PLE dtype 全部不变。

## 收益已知与未知

上游已合并实现；本仓库尚无 H800 收益数据。局部耗时下降是否传导到端到端指标，取决于该阶段占比与排队负载。已覆盖时不重复包装为新优化。

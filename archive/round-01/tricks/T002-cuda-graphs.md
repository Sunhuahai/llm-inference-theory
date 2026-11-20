# T002 — 让 CUDA Graph 覆盖真实 Decode 形状

> **已归档（2026-11-20，Round 01）。** 生命周期：`archived`；尝试记录：`user_report / attempted`。用户明确 T001–T013 均已尝试。逐项收益、质量与实施证据尚未提交，保留为 `unknown`；下文的研究状态和“待实测”措辞是原始历史记录。不得据此重新派发，也不得推断全部成功或失败。
> 当前新增候选见 [Round 03](../../../rounds/round-03/README.md)；归档说明见 [Round 01 索引](../README.md)。

- ID：T002
- 状态：researched
- 调研日期：2026-11-16
- 验证：待 H800 实测

## 问题与机制

低并发 decode 若被 CPU 发射与 GPU 空洞限制，graph replay 可能降低 TPOT。vLLM 提供 full/piecewise 等模式，但混合模型由所有 backend 的兼容性共同决定最终模式；捕获更多形状也会增加显存和初始化成本。[官方设计文档](https://docs.vllm.ai/en/stable/design/cuda_graphs/)

## 模型与版本门槛

固定 FP8 checkpoint 与权重、激活、KV、递归状态 dtype。[官方 vLLM 模型配方](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) 标注最低 0.30.0，性能配方曾用 nightly；必须冻结实际 commit/digest。逐项确认 QSA、Gated DeltaNet、MoE 和已启用 MTP 对 graph 的组合支持，保留 capture 日志与最终 mode，不能强套普通全注意力模型的 FULL 配置。

## 实施 Agent 步骤

1. profile 基线实际 decode batch 分布、kernel launch gap 和 graph replay/fallback；已经正常生效的默认 graph 不记为新增收益。
2. 在框架支持的 capture 参数中，只扩展最常见且未覆盖的 batch 档位；保持相同 graph mode。
3. 如需比较 mode，另开 A/B；捕获后重新记录 state/KV 容量与显存。不要靠缩短上下文或改变精度腾空间。

## 单变量 A/B

A 为原 capture 配置；B 为新增的合法形状覆盖。固定同一短输入长输出请求集、并发、到达轨迹、缓存状态与 MTP 设置，另用混合流量复查。比较 p95 TPOT、ITL、输出 tok/s、padding 数、fallback 次数、graph buffer 和可用缓存容量。启动/capture 时间单列，不混入稳态 TTFT。

## 正确性与回滚

检查 batch 大小变化、请求加入/退出、长短交错时的输出与状态隔离。发生 capture 错误、非法访问、输出异常或缓存容量退化导致 SLO 吞吐下降，恢复原 capture 配置。

## 已知与未知

已知：graph 可减少执行发射开销。未知：本机是否 launch-bound，实际模型能支持哪些模式、收益是否抵消缓存占用；没有 H800 实测。

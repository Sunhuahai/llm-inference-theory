# T010 — MoE 通信后端与计算重叠

> **已归档（2026-11-20，Round 01）。** 生命周期：`archived`；尝试记录：`user_report / attempted`。用户明确 T001–T013 均已尝试。逐项收益、质量与实施证据尚未提交，保留为 `unknown`；下文的研究状态和“待实测”措辞是原始历史记录。不得据此重新派发，也不得推断全部成功或失败。
> 当前新增候选见 [Round 03](../../../rounds/round-03/README.md)；归档说明见 [Round 01 索引](../README.md)。

- ID：T010
- 状态：conditional
- 调研日期：2026-11-16
- 验证：待 H800 实测

## 问题与机制

专家 dispatch/combine、permute 和跨 rank 等待可能占据 MoE 执行时间。SGLang 将通信后端与计算后端分开配置，且区分偏 prefill 吞吐与偏 decode 延迟的路径；重叠也有额外开销。[官方 EP 机制](https://docs.sglang.io/docs/advanced_features/expert_parallelism)

## 模型与版本门槛

只有 profile 证明通信占比显著才进入此卡。[官方 vLLM 模型配方](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) 最低 0.30.0，Hopper 使用 `--moe-backend triton`，以此作已核查候选基线；其他后端先证明专家格式、分片维度和 H800 支持。冻结权重、激活、KV、状态及通信 dtype、路由 top-k 和尺度，记录实际框架/通信库 commit；不能照搬 Blackwell kernel。

## 实施 Agent 步骤

1. 在合法并行布局上记录每 rank token 负载、dispatch、专家 GEMM、combine 与空等时长。
2. 基线与一个受支持的通信后端比较；SGLang 核验 `--moe-a2a-backend`，vLLM 核验 `--all2all-backend`。不要顺带更换计算精度或 kernel 配方。[vLLM 后端与版本要求](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
3. 后端稳定后，另开单变量实验测试受支持的 overlap；有 rank 失衡证据再调 EPLB，三项不叠在同一次 A/B。

## 单变量 A/B

固定模型、GPU 数、合法布局、到达轨迹和缓存状态，分别测低并发 decode 与较高负载混合流量。输出 TTFT/TPOT/ITL 分位数、输出 tok/s、各通信阶段时长、最大/平均 rank 负载、额外显存。只改善微基准而端到端退化不接受。

## 正确性与回滚

检验路由重排后的 token/expert 对应关系、输出质量、取消请求和 batch 波动。后端错误、死锁、质量异常或尾延迟明显退化时回到原后端及原 overlap/EPLB 配置。

## 已知与未知

已知：官方框架有这些机制。未知：目标模型×H800×MTP×graph 的组合兼容性；未通过 capability probe 前不提供“一键全开”命令或加速比。

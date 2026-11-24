# T012 — 后置候选：Prefill / Decode 分离

> **已归档（2026-11-20，Round 01）。** 生命周期：`archived`；尝试记录：`user_report / attempted`。用户明确 T001–T013 均已尝试。逐项收益、质量与实施证据尚未提交，保留为 `unknown`；下文的研究状态和“待实测”措辞是原始历史记录。不得据此重新派发，也不得推断全部成功或失败。
> 当前新增候选见 [Round 03](../../../rounds/round-03/README.md)；归档说明见 [Round 01 索引](../README.md)。

- ID：T012
- 状态：conditional
- 调研日期：2026-11-16
- 验证：待 H800 实测；模型状态传输支持待确认

## 问题与机制

当长 prefill 持续干扰 decode 尾延迟，P/D 分离可分别安排资源和调度，但会引入状态传输、排队及额外资源开销。vLLM 将其标为实验性功能，并明确不能把它直接当吞吐提升手段。[官方说明](https://docs.vllm.ai/en/stable/features/disagg_prefill/)

## 模型与版本门槛

这是后置的多实例/多机候选；先完成分块调度，确认已有设备和互联，不预设卡数。Qwen3.8 的 QSA KV 与 Gated DeltaNet 状态必须都能正确传输/恢复。[vLLM 模型配方](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) 最低 0.30.0、PLE 初始不支持 PP；最低版本和通用 connector 均不证明 P/D 可用。完整组合仍未确认。

## 实施 Agent 步骤

1. 查所选版本的 connector 支持矩阵与代码，证实 hybrid state、layout、合法并行、MTP/cache 组合；缺失则标记 blocked，不用纯 KV 示例强行上线。
2. 固定 FP8 checkpoint 与全部 dtype，在最小受支持拓扑完成单请求状态转移与质量验证。
3. 同总 GPU 预算比较统一引擎和 P/D；先用同合法布局，异构 TP 另立实验。[SGLang connector 文档](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)

## 单变量 A/B

A 为已调优统一服务，B 为受支持的 P/D 布局；固定总 GPU、prompt/output、到达 trace、SLO 和缓存策略。测长短混合及突发流量，报告 TTFT/TPOT/ITL p95/p99、输出 tok/s、tok/s/GPU、成功率、P/D 排队、传输耗时与字节数。接收方等待也计入端到端延迟。

## 正确性与回滚

检验长上下文、递归状态恢复、重试/取消、断连及输出质量。所有实验路由可切回统一基线；模型传输不完整、错误输出或同 SLO 容量下降则停止。

## 已知与未知

已知：框架有 P/D 机制。未知：该模型在 H800 上所需 connector、状态布局兼容性与资源预算；此卡不声称开箱可用，也不承诺吞吐收益。

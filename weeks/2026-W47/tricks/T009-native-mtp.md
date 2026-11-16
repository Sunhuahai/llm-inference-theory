# T009｜原生 MTP 深度与负载适配

| 字段 | 内容 |
|---|---|
| ID / 优先级 | T009 / P2 |
| 状态 | `researched`；待 H800 实测 |
| 调研日期 | 2026-11-16 |
| 固定条件 | checkpoint、所有 dtype、采样策略、业务任务与输出要求 |

## 问题与机制

[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) 包含原生 MTP 模块，可提出候选 token，再由目标模型验证。[vLLM recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) 提供 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`。收益依赖接受率和验证开销，不应默认高并发也更快。

## 适用与版本门槛

确认固定引擎 build 支持该 checkpoint 的 MTP、当前并行布局与缓存实现。所有 KV/indexer/SSM/PLE dtype 沿用基线，不能复制配方中的 FP8 cache 参数。基线已有 MTP 时登记 `baseline-covered`，将深度调优单独归因。

## 交给实现 Agent

| 步骤 | 操作与交付 |
|---|---|
| 1 | 固定引擎 SHA 和权重 revision，确认原生 MTP 权重加载成功；登记目标与 draft 的实际 dtype。 |
| 2 | 在其他参数相同条件下比较 MTP off、1、2、3 个 speculative token。 |
| 3 | 记录提出、接受和回退的 token 数、平均接受长度及验证耗时；检查并禁用模拟 acceptance 的实验环境开关。 |
| 4 | 用真实业务文本及相同采样策略测量，提交原始请求结果、接受率和性能曲线。 |

## 单变量 A/B

仅改变 MTP 深度；短/长输出与并发 1/8/32 分组，保持输入、缓存状态与输出预算一致。报告 TTFT、TPOT p50/p95、output tokens/s、总 tokens/s及显存。吞吐只计最终接受并交付的 token，不计被拒草稿；流式多 token 合并到达不能当作多个零耗时 token。

## 正确性与回滚

验证目标模型的采样语义与输出质量；确定性用例检查 token 序列，随机采样做预先规定的统计与任务验收。不能通过关闭思考、缩短答案或改变温度制造收益。失败、接受率低或目标负载吞吐下降时，回退此前深度或关闭 MTP。

## 收益已知与未知

模型和上游 recipe 支持原生 MTP；H800 的真实接受率、最优深度及净收益尚无数据。最终应报告按负载选择的 Pareto 工作点，而不是单一“加速倍数”。

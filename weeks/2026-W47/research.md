# 首轮调研：从模型路径到可验证的系统收益

检索日期：2026-11-16。状态：研究建议，未在用户 H800 实测。

## 研究结论

当前最有价值的工作是：**确认已部署实现 → 建立同口径基线 → 用 profile 和真实负载选 2–3 个变量**。这个模型同时有 QSA、GDN、GR、PLE 与 MoE，不能只凭“LLM 推理慢”就认定是 attention 算力或带宽不足。模型结构的一手依据见 [模型核对](../../docs/model-and-support.md)。

### 1. 上游已经有可复用的模型专属优化

vLLM 已合并 [QSA prepare 融合 #57097](https://github.com/vllm-project/vllm/pull/57097)、[QSA 投影融合 #59533](https://github.com/vllm-project/vllm/pull/59533)、[GR/PLE sequence parallel #56322](https://github.com/vllm-project/vllm/pull/56322)。因此首个问题是“现有 build 有没有走到这些路径”，而不是先重写内核。

调研推断：若基线较旧，完整升级可能同时改变多项行为。可把升级当作独立候选测端到端收益，但不能把总提升归因于某一个 PR；需要同一构建内开关或最小 patch 对照时，再付出消融成本。若基线已覆盖，记录实际调用证据，将预算转向形状、调度或未解决的等待。

### 2. 这个模型的 CPU 路径值得单独检查

官方说明 PLE/N-gram embedding 支持放在主机并异步预取；其性能会涉及主机内存和数据传输。[Qwen 官方说明](https://github.com/QwenLM/Qwen3.8-Flash-Next)

调研推断：在用户实际部署中，应先确认是否 offload、各 rank 如何读表以及 NUMA 亲和性。CPU DRAM 或 PCIe 等待可能使 GPU 侧优化受限。显存不足时，“先把表全搬回 GPU”并不是合法候选；先优化已存在的 host 路径，并测尾延迟。

### 3. 前缀缓存要理解为混合状态的复用

SGLang 的 [hybrid checkpoints 文档](https://docs.sglang.io/docs/advanced_features/hybrid_cache_checkpoints) 明确讨论边界/分支状态，并给出 Qwen3.8-Flash-Next 的案例。其来源硬件与本机不同，数字不移植。

调研推断：多轮继续和从历史位置分叉需要分别测；KV 命中率高不保证整个混合模型恢复便宜。checkpoint 密度、状态池容量与复制/eviction 成本可能抵消少做的 prefill。只有业务确有复用，T008 才进入优先队列。

### 4. 多卡方案必须遵守模型布局与实际互联

当前 [vLLM recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) 对 FP8 plain TP8 有明确限制；不同框架的 EP、TP、DP 进程组含义也不能简单相乘。

调研推断：保持总 H800 数相同，先筛合法且装得下的布局，再比较通信、显存和负载分布。单请求最低延迟与最大系统吞吐可能对应不同配置，应该保留两类工作点而非混成一个“最快”。

## 指标如何共同约束选择

| 优化方向 | 可能改善 | 需要同时看见的代价 |
|---|---|---|
| 更细 prefill 分块 | decode 间隙、TPOT 尾部 | prompt 完成变慢、TTFT 上升 |
| 更大 batch token 预算 | GPU 利用率、吞吐 | 排队、长 prefill 对 decode 的阻塞 |
| Graph / 融合 | launch 与中间内存开销 | Graph 显存、形状 padding、fallback |
| GR/PLE SP | 长输入重复计算 | 小 batch 通信与 decode 回退 |
| 更密 hybrid checkpoint | 分叉时复用更多 | 状态存储/拷贝/eviction |
| MTP | 每次验证产出更多 token | draft/verification 额外工作、接受率变化 |

上表是机制分析，方向必须由本机实验确认。指标精确定义、warmup、冷热缓存、错误率和 MTP 多 token chunk 的处理见 [Benchmark 协议](../../docs/benchmark-protocol.md)。

## 首轮最小预算

先做环境与基线，不强迫完成全部场景矩阵。随后选择一项调度类和一项 profile 指向的模型专属优化，各测主场景与一个回归场景。保存失败、零收益和已默认覆盖的发现，这些同样决定明天该查什么。

尚缺：H800 卡数/SKU/互联、当前框架/版本、真实输入输出分布、业务 SLO、baseline。它们不妨碍研究建档，但决定具体命令、布局和验收阈值。

## 2026-11-17 专项：Prefix 状态复用、命中率与上下文压缩

新增 [`theory/cache-context-ceilings.md`](../../theory/cache-context-ceilings.md) 和 [可运行 trace 回放模型](../../theory/predictor/README.md)。关键研究观察：

1. **计数器不能混用**：prefix cache 的 request hit、cached input tokens、按可省 Prefill 加权的成本命中三者可相差很大；vLLM 官方的 `prefix_cache_hits / queries` 是 token Counter。
2. **状态不能只写 KV**：QSA/GDN 混合状态的 cacheability、checkpoint 位置、恢复开销与两池容量决定真实收益。LMCache 对该模型的 CacheGen 压缩标记不支持，不能当可用 trick。
3. **压缩改变命中分布**：结构去重或有损摘要虽然减少 input tokens，却会改变 prefix token IDs；必须测压缩器开销、压缩后缓存键和任务质量。
4. **上界分层**：Amdahl 全 Prefill 消失 > 免费命中 oracle > 无驱逐同键情景 > 有限池 LRU；此序列只在模型正常节省计算、精确键与简化成本假设内成立，不可直接推到并发块级真实服务。合成回放用于校验方法，不是 H800 观测。

实施建议仍只选 2 项前置工作：**收集脱敏前缀 trace** 与 **建立业务压缩质量集**；两者未到位前，不对实际加速作点估计。

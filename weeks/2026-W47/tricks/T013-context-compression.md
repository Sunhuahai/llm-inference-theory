# T013 — 上下文压缩与 Prefix 复用的协同优化

- ID：T013
- 状态：**conditional**
- 日期：2026-11-17
- 证据：`literature` + `hypothesis`；尚无 H800 实测，尚无本机业务质量结果
- 目标：不变更 FP8 模型和现有计算/KV/state 精度，降低**原输入语义约束不被破坏前提下**的 Prefill 工作量

## 机制与收益上界

逐级评估 **结构去重 → 查询相关信息剪裁 → 任务条件有损摘要**。它们都可能减少传入模型的 input tokens，从而影响 TTFT、批处理容量与 Prefill/Decode 干扰。然而：

1. 压缩器本身有时延、CPU/GPU 资源开销；长 prompt 的计算量不必与 token 数线性相关。
2. 输出质量、工具参数、引用证据、系统指令遵循不能牺牲；短回答或关闭 reasoning 不是候选。
3. 压缩改变 token IDs，原先相同的前缀可能分裂；相反也可能减少 state MiB，使缓存容量内可保留更多条目。因此 **T013 与 T008 必须联合重放不同缓存键**。

以同一请求分布为基准，冷路径的必要盈亏条件 `\bar T_{\rm compressor}<\overline{T_{\rm pre,cold}-T'_{\rm pre,cold}}`；已有缓存时必须改为对比**实际压缩前/后 LRU trace**，不能直接把“token 降 50%”认作 TTFT 降 50%。

可运行情景模型：[`theory/predictor/cache_context.py`](../../../theory/predictor/cache_context.py)；推导：[缓存/上下文压缩天花板](../../../theory/cache-context-ceilings.md)。

## 模型/版本/业务前提

- 先核实 Qwen3.8-Flash-Next（`Qwen4Exp`）的 GDN + QSA 混合状态、实际 vLLM/SGLang build 与原始前缀状态池策略；这是应用侧 prompt 压缩，不是改模型 QSA 结构。
- 部署业务确有很长的共享指令、RAG 文档、工具输出或多轮历史；没有这类冗余时无必要套压缩。
- 必须有**配对任务质量测试集**，包含否定/数字/引用/长距离依赖/工具调用；原输入信息不可静默删去。
- 不把 [LMCache recipe](https://docs.lmcache.ai/recipes/qwen3_8_flash_next.html) 的 **CacheGen（当前标不支持）**误当成已验证的混合状态压缩工具；MTP 组合也未验证。

## 最小单变量 A/B

| 组别 | 唯一变量 | 预期 | 代价及观察 |
|---|---|---|---|
| A 冷基线 | 原文 / 无缓存 | 锚定冷 Prefill 与任务答案 | TTFT、输入/输出 token、质量 |
| B 冷结构去重 | 仅去重复内容 | 可能减少 TTFT | tokenizer 长度、去重耗时、信息保留 |
| C 暖原文 | 原文 / 当前缓存 | 真实缓存收益 | 命中 request/token、hit cost、state MiB、eviction |
| D 暖去重 | 去重 / **相同缓存预算** | 可能改变 prefix 命中与容量 | 重新哈希、压缩后命中、TTFT / output TPS |

**A↔B、C↔D 才是直接单变量对照**；随后另开独立试验研究检索剪裁与摘要，不在一次 A/B 同时更换模型、提示模板、缓存容量或 GPU 总数。每组固定 checkpoint、sampling、业务目标、原始问题集合与输出长度测量规则；交错至少三次，在短/长/重复前缀/随机前缀组分层报告。

## 质量、验收、回滚

- 先固定质量标准再测速度；**未完成质量验证时状态只能是 conditional/researched**，不得给 validated。
- 结构去重须证明未删除独立事实或必要的工具消息；有损摘要必须通过配对的事实/引用与工具调用验收。记录 **压缩器运行时间、效果退化、错误率、输出长度变化**。
- 缓存/压缩后必须重算真实 prefix keys，验证跨租户隔离，避免语义相似却哈希碰撞；公开提交脱敏 hash，不存原始 prompt。
- 在任一任务群出现不可接受的关键信息丢失、OOM、p95/p99 回退或输出错误时立即关闭相应压缩步骤，恢复原文；缓存参数单独回退。

## 一手研究

1. [LLMLingua-2（ACL 2024）](https://aclanthology.org/2024.findings-acl.57/)：prompt compression 研究，不代表本 H800/模型收益。
2. [LongLLMLingua（ACL 2024）](https://aclanthology.org/2024.acl-long.91/)：长上下文与问题相关压缩。
3. [Lost in the Middle（TACL 2024）](https://aclanthology.org/2024.tacl-1.9/)：长上下文证据位置效应。
4. [vLLM Prefix Cache 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)：缓存基于 token 前缀块与 hash。
5. [LMCache 专用 recipe](https://docs.lmcache.ai/recipes/qwen3_8_flash_next.html)：混合状态缓存/压缩的实际上游支持边界。

下一步所需最小输入：一份 50–200 条**脱敏、有序**业务 trace 摘要、缓存状态预算与 20–50 条带标准答案的配对质量样本。规模仅为建议起点，不是统计充分性的保证。

# Qwen3.8-Flash-Next：前缀状态复用、上下文压缩与性能天花板

> 研究日期：2026-11-17；证据等级：上游文档 + 解析模型 + 合成算例。**尚无用户 H800 实测数据、实际请求 trace 或上下文压缩质量结果。** 固定原有 FP8 checkpoint、权重/计算/KV/indexer/SSM/PLE 精度和 H800 总卡数。

## 摘要与工程判断

在固定精度下，把两个概念分开：

1. **复用计算**：重复前缀被命中后，恢复兼容的混合状态，减少之前已经做过的 Prefill；典型是长 system prompt、共享 RAG 文档、多轮对话。
2. **减少计算**：删去冗余 token、对文档去重或做有损摘要，使送入模型的 prompt 更短；节省 Prefill，但可能损害答案完整性、破坏前缀键并引入压缩耗时。

这两条路线不互斥，却**不应相加收益**：输入 token 序列一旦变化，新的前缀键、命中率、state 容量、实际输出都可能变化。先用 trace 回放估计缓存，再独立测压缩质量，最后重放压缩后的 trace。

| 问题 | 可得到的上界 | 主要缺失证据 | 首选试验 |
|---|---|---|---|
| 本地 Prefix/KV 命中是否值得优化？ | 有限容量 LRU vs 无淘汰同键回放；更松的免费恢复 oracle | 实际前缀 token hash、状态大小、冷/热 Prefill、eviction | T008，同一 trace 两种缓存池/策略 |
| 上下文是否值得压缩？ | 原 prompt 与压缩 prompt 的分段 Prefill 成本差减压缩器开销 | 语义质量、错误恢复、压缩后前缀复用 | T013，无压缩/去重/摘要三组 |
| 能否把 TTFT 的 2×–3× 变成吞吐 2×–3×？ | **不能由 TTFT 推断**；须用资源需求和饱和容量模型 | 同一负载输出 tok/s、CPU/HBM/通信需求、缓存/压缩开销 | 执行 [原预测器](../../models/predictor/README.md) 的容量校准 |

## 1. 模型特性决定缓存边界

Qwen3.8-Flash-Next 的执行是 **Gated DeltaNet（GDN）+ Qwen Sparse Attention（QSA）混合架构**，不是传统 Transformer 所有层都拥有同一种 KV cache。输入复用必须同时检查 recurrent state、QSA 可复用索引/缓存，以及不可复用的暂存状态；**某个 KV 命中数字不等于全模型计算已省掉相同比例**。

官方文档的实际能力边界：

| 组合 | 已核实状态 | 对 H800 项目的限制 |
|---|---|---|
| vLLM Automatic Prefix Caching | 对相同 token 前缀按块复用；混合 Mamba 模型有特殊约束 | 检查实际 vLLM git SHA、Mamba cache mode 和缓存是否真的生效 |
| SGLang hybrid checkpoints / RadixCache | 对混合状态提供 checkpoint 与复用机制 | 不移植上游其他 GPU 的速度数字；须区分 continuation / fork |
| LMCache + vLLM + `Qwen/Qwen3.8-Flash-Next` | 官方 recipe 标注 8 GPU 文本路径已验证，`align` 模式 | 并未证明用户的 **FP8 制品、H800 拓扑、实际版本**受同等验证 |
| LMCache 的 CacheGen 状态压缩 / CacheBlend | 该 Qwen3.8-Flash-Next recipe 标记不支持 | **不作为近期可实施的 KV/state 有损压缩 trick** |
| LMCache + 该模型 MTP | 官方标记尚未验证 | 不把 MTP 与外部缓存的组合收益默认算进去 |

LMCache 官方还说明：QSA 压缩组的 per-request scratch ring 不缓存；线性注意力分组以 opaque bytes 存储；`align` 模式的 Mamba prefix caching 属于实验性，缓存输出不保证 bit-exact。以上都意味着“外部缓存命中”必须结合**数值/任务质量**和端到端 TTFT 验证。

**防混淆**：QSA 模型内部对历史注意力的稀疏/压缩处理，与业务侧删减 prompt 的“上下文压缩”不是同一件事；后者会改变模型的条件输入，不能当作无损算子加速。

## 2. 缓存命中率的三个分母

对 N 个按到达顺序处理的请求，输入长度为 `L_i`、可复用前缀长度为 `P_i`、实际命中指示为 `H_i\in\{0,1\}`。区分：

\[
h_{\rm req}=\frac{\sum_i H_i}{N},\qquad
h_{\rm tok}=\frac{\sum_i H_iP_i}{\sum_i L_i},
\]
\[
h_{\rm cost}=
\frac{\sum_i H_i\,[C_i-S_i]_+}{\sum_i[C_i-S_i]_+}.
\]

`C_i` 是同条件**未复用**的 prefill wall time，`S_i` 是命中后**剩余计算** wall time；成本加权比使用单个请求能避免的最大计算成本作为权重。它是定位指标，不等于净加速：还要扣查找、状态恢复/传输、eviction、队列及共享 batch 的干扰。

**vLLM 指标注意**：`vllm:prefix_cache_hits` / `queries` 在当前官方指标表中是**缓存 token 计数**，而不是“命中请求个数”；`prompt_tokens_cached_by_source` 按 device / host / disk / p2p 等来源区分。计算 rate 应使用同窗口 Counter **增量**，区分 local、external tier，并核对用户安装版本。`kv_cache_usage_perc` 是容量使用率，不是命中率；H800 的 **L2 hardware cache hit** 又属于另一层硬件性能指标。

还应记录多租户隔离、模版/tokenizer 版本、prefix hash、session affinity、前缀块边界与共享缓存引用。**公开仓库只能存脱敏的、带隔离域的 token 前缀哈希，不存原始业务 prompt。**

## 3. 从 Amdahl 到实际容量的多层天花板

设每请求固定 TTFT `F_i`（tokenize/调度/首步/网络等），忽略队列的低负载基线：

\[
\bar T_0=\frac1N\sum_i(F_i+C_i).
\]

对指定部署方案，令 `H_i(B,\pi)` 为池容量 `B` 和缓存策略 `\pi` 回放后的命中标记，查找成本 `\ell_i`，读取/拷贝状态成本 `R_i`。则：

\[
\bar T_{\rm cache}(B,\pi)=\frac1N
\sum_i\big[F_i+\ell_i+(1-H_i)C_i+H_i(S_i+R_i)\big].
\]

每个真实命中的**必要条件**是其预期节省 `C_i-S_i` 足以支付 `R_i` 和相应查找成本；服务队列和容量竞争会继续改变实际收益。理论上：

| 层次 | 定义 | 能用于何种结论 |
|---|---|---|
| L0：全 Prefill 移除 | `S_{\rm absolute}=\bar T_0/\bar F`，假设所有 Prefill 时间消失 | **极松数学上界**，不可当工程目标 |
| L1：免费命中 oracle | 对相同完整 prefix key 的首次出现冷执行、后续命中完全免费，容量无限 | 当前**完整键模型**中的乐观上界；无视必要 suffix / state 开销 |
| L2：无淘汰的精确键缓存 | 首次冷执行、后续复用但保留 `S_i+\ell_i+R_i` | 区分“重复前缀潜力”与驱逐损失 |
| L3：有限容量缓存 | 按真实 state MiB、访问顺序和 LRU/其他已知策略回放 | 可以作为工程**情景预测**；不包含并发/部分块命中 |
| L4：真实在线服务 | 多并发、hybrid state 多池、队列、不同 cache tier、通信及 p99 | 必须由真实服务回放和实验验证 |

L1 / L2 不一定是**整个 vLLM 前缀缓存算法的全局上界**：现实的分块匹配可复用共同子前缀，即使请求的“完整键”不同。现有可运行脚本采用 **单对象、完整 key、按序 LRU**，不冒充 block trie、多池 checkpoint 或离散事件模拟。

**再加一个内存约束**：对并发请求集合，实际可复用状态总量 `\sum_jM_j` 必须受可用缓存预算限制；显存用于更多 checkpoint，也可能挤占活跃 batch / CUDA Graph / KV token 池，使吞吐或 TPOT 下降。提高命中率不等于任意增大缓存池都是正收益。

## 4. 上下文压缩：节省 token 与保留答案之间的约束优化

可分三档，按风险排序：

| 方法 | 对输入的改变 | 质量风险与机制 | 优先 |
|---|---|---|---|
| 结构级去冗余 | 去掉重复工具输出、重复文档段；稳定序列化/排版（必须不丢信息） | 工具协议与引用位置必须保持；可能改变 token hash | P1，先证明语义等价 |
| 查询相关剪裁 | 只给相关 RAG passages；按检索分数与位置组织 | 检索遗漏长尾证据；答案可能过度自信 | P2，按事实召回/引用正确率验收 |
| 有损摘要/模型压缩 | 压缩历史对话、长文档与低优先级上下文 | 数字、否定、指令层级、跨段推理与长位置依赖可能丢失 | P3，只有质量门槛明确才评估 |

原型方法有 [LLMLingua-2（ACL 2024）](https://aclanthology.org/2024.findings-acl.57/) 与 [LongLLMLingua（ACL 2024）](https://aclanthology.org/2024.acl-long.91/)。论文展示的是它们各自模型/任务上的结果，**不能移植为 Qwen3.8-Flash-Next + H800 的预期加速或质量保证**。此外长上下文的证据利用也有位置敏感性（[Lost in the Middle，TACL 2024](https://aclanthology.org/2024.tacl-1.9/)）。

记压缩前后输入 `L_i,L'_i`，压缩器成本 `A_i`，压缩后的冷 prefill `C'_i`，以及**重新计算**的压缩前缀命中 `H'_i`。无缓存的必要盈亏条件：

\[
\bar A < \overline{C-C'}.
\]

若基线已经启用缓存，实际要比较的是：

\[
\bar T'_{\rm combo}=\frac1N\sum_i
\big[F_i+A_i+\ell'_i+(1-H'_i)C'_i+
H'_i(S'_i+R'_i)\big]
\quad\text{与}\quad
\bar T_{\rm cache}.
\]

**不能直接把“压缩 50% token”换算成“TTFT ×2”**：Prefill FLOPs 不必随 token 数线性变化；这个混合模型已有 QSA 内部压缩；外层 tokenizer、首步、恢复、排队不消失。也不能把未压缩 trace 的缓存命中率用于压缩后 trace。

质量约束建议作为严格 gate：按具体业务设置事实/引用召回、工具参数精确性、系统指令遵循、算术与长文档问答的一致性，保留长尾失败集合；任务成绩、失败率、实际输出长度同时报告。**任何性能提升都不以无提示截断输入、关闭 reasoning 或降低精度替代**。若质量结果缺失，只能报告“条件加速”，不能晋级 validated。

## 5. 可运行模型与合成校验

最小原型位于 [`predictor/cache_context.py`](../../models/predictor/cache_context.py)，以时间序列 `requests` 逐条回放：

- 原始 trace：无缓存、有限 MiB LRU、无限完整键缓存、免费命中 oracle。
- 压缩后 trace：无缓存、有限 MiB LRU、无限完整键缓存；使用**独立的** `compressed.prefix_key`，叠加压缩器耗时，输出质量门槛。
- 输出：请求命中率、命中输入 token 比例、可节省 Prefill 的成本加权命中率、eviction、平均 TTFT 与不同基线的相对改善。
- 不输出：p95/p99、排队、块级/两池行为、GPU L2 命中率、饱和 output TPS 或“已通过质量”的无证据结论。

命令：

~~~bash
python3 models/predictor/cache_context.py \
  models/predictor/examples/synthetic-cache-context.json --format markdown

python3 -m unittest discover -s models/predictor/tests -v
~~~

**全合成 8 请求算例**（不是 H800 性能）：

| 方案 | 请求命中率 | 平均 TTFT | 相对冷原文 | 解释 |
|---|---:|---:|---:|---|
| 原文 / 无缓存 | 0% | 121.50 ms | 1.000× | 共同起点 |
| 原文 / 40 MiB LRU | 12.5% | 114.62 ms | 1.060× | 容量不足，发生 6 次驱逐 |
| 原文 / 无限完整键 | 62.5% | 86.00 ms | 1.413× | 显示多大一部分损失与驱逐有关 |
| 原文 / 免费命中 oracle | 62.5% | 60.62 ms | 2.004× | 假设恢复+suffix 都免费，非实施目标 |
| 压缩 / 40 MiB LRU | 50.0% | 75.92 ms | 1.600× | 压缩后重建键；**尚无质量证明** |

该构造中压缩 token 降 47.5%，压缩器为 8 ms/请求；原文有 A/B/C 三组重复键，压缩后 A 的摘要分裂成 A1/A2，说明“token 少了”和“缓存命中变了”需同时建模。合成结果**仅验证代码与解释能力**。

## 6. 为真实天花板补哪些测量

| 优先级 | 实施 Agent 需交付的脱敏字段 | 用于辨识 |
|---|---|---|
| P0 | 模型 revision、tokenizer/chat template hash、推理框架 SHA、cache mode、真实固定精度、H800 拓扑 | 排除基线漂移和不合法复用 |
| P0 | ≥一段代表性有序请求 trace：输入/输出 token 长度、盐化 prefix key、可复用 token 数、时间戳、隔离域 | `h_req`、`h_tok`、跨会话共享潜力 |
| P0 | 每类前缀的 QSA / GDN 状态 MiB、block 大小、cache pool 占用、eviction/preemption；按来源分层命中 token | 估计容量敏感曲线与实际 H800 内存代价 |
| P0 | 配对 cold/warm prefill、查找/恢复/传输时间、客户端 TTFT、TPOT、输出 tok/s | 量化 `C_i-S_i-R_i-\ell_i` |
| P1 | 3 档缓存容量下按同一顺序重放，控制并发及 session affinity；再测实际线上流量 | 从 LRU 情景到真实混合状态缓存 |
| P1 | 压缩前后 token IDs hash、压缩器耗时、压缩后命中序列、配对任务质量/实际输出 | 识别有损压缩的净效益与可否上线 |
| P2 | 饱和 profile：GPU Compute/HBM/互联/CPU 的每 output token 需求 | 接回 [容量模型](../../models/predictor/README.md)，而非 `1/TTFT` |

按 `[Benchmark 协议](../../docs/benchmark-protocol.md)` 同时记录 p50/p95/p99、负载、失败、OOM 与质量；缓存污染测试要做 cold/warm A/B 和交错重复。真实参数记录为 `measured`；估计参数写 `hypothesis`，避免合成字段与真实字段混杂。

## 7. 研究来源（均为一手资料）

1. [Qwen 官方模型与架构](https://github.com/QwenLM/Qwen3.8-Flash-Next)：GDN、QSA、PLE、MTP 架构边界。
2. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)；[Prefix Cache 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)：前缀块复用、混合状态与哈希。
3. [vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/)：`prefix_cache_hits/queries`、来源分层与使用率的确切含义。
4. [LMCache 的 Qwen3.8-Flash-Next recipe](https://docs.lmcache.ai/recipes/qwen3_8_flash_next.html)：`align`、实验性 GDN、CacheGen 与 MTP 的支持边界。
5. [SGLang hybrid checkpoints](https://docs.sglang.io/docs/advanced_features/hybrid_cache_checkpoints)：混合状态续写与分支复用。
6. [LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/)、[LongLLMLingua](https://aclanthology.org/2024.acl-long.91/)、[Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)：上下文压缩方法与任务敏感性。

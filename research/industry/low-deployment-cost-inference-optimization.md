# 低部署成本的推理优化：从状态复用到自适应调度

> 2026-11-26 · 工程机制调研及实验指南  
> 目标：在已有 vLLM / SGLang 推理服务中，优先寻找通过启动参数或客户端请求配置就能验证的优化，不增加 GPU 数量，不部署新的缓存服务、路由服务或 P/D 集群，不重新训练、量化或改变模型任务。  
> 本文将上一轮六项轻量技术与本轮新增九类机制汇总。它是按机制组织的理论实例与实验入口，不是固定数量的 Trick 任务清单。公开资料核对日期为 2026-11-26；latest / stable 文档会随上游更新，尚未确认本项目实际引擎版本、模型后端与运行兼容性，未运行 GPU 实验。示例参数仅作起点，不是最优值或加速承诺。

## 1. 首先判断部署是否真的简单

| 等级 | 定义 |
| --- | --- |
| A | 已有引擎支持时，仅修改启动参数；无需额外依赖或服务 |
| B | 在现有能力上增加少量配置或客户端字段，需要额外正确性/协议验证 |
| C | 参数虽少，但依赖特定模型、后端、driver、并行布局或实验性实现；条件不满足就暂停 |

所有等级都以目标版本已有功能为前提；如果缺少支持，需要大规模升级依赖或重新编译，不能仍然称为低部署成本。保持原有 checkpoint、权重/激活/KV/循环状态精度、采样语义、并行拓扑、请求输入输出与总设备预算。不能以缩短输出或关闭 reasoning 制造收益。

## 2. 优先级总表

| 机制 | 引擎 | 难度 | 优先调查的现象 | 主要指标 |
| --- | --- | --- | --- | --- |
| **细粒度前缀匹配与共享检查点** | vLLM | A/C | 混合架构的共享前缀在块边界被截断 | 热请求 TTFT、实际复用 token |
| **混合模型检查点保留策略** | SGLang / vLLM | A/C | 中间状态过多，或中途分叉需重算 | TTFT、状态容量、写流量 |
| **代价感知 Radix 淘汰** | SGLang | A/C | 有效状态反复被淘汰 | 尾 TTFT、重算、吞吐 |
| **混合模型显存池分配** | SGLang | A | KV 与循环状态的一个池耗尽、另一个闲置 | 并发、命中、吞吐 |
| Session-Aware Radix Cache | SGLang | B/C | Agent 两次调用之间状态易丢失 | 多轮 TTFT |
| Adaptive Long Prefill Threshold | vLLM | A | 长 Prompt 阻塞短请求 | TTFT 分位数、TPOT |
| Stream Interval | vLLM | A | 高频 SSE 输出、CPU 前端负载 | 吞吐、客户端体验 |
| Dynamic Speculative Decoding | vLLM / SGLang | B/C | 静态投机长度在高低并发下表现不同 | TPOT、有效接受长度、吞吐 |
| Extensible KV Cache | vLLM | A/C | 预估显存分配留下大量不可利用余量 | KV 容量、抢占、吞吐 |
| Native CPU KV Offloading | vLLM | A/C | GPU 缓存工作集超出容量 | 热 TTFT、吞吐 |
| GPU N-gram | vLLM | A/C | 重复生成片段、低并发 Decode | TPOT、接受长度 |
| KV Watermark | vLLM | A | KV 紧张、频繁抢占 | 尾延迟、吞吐 |
| Time-Sliced Prefill | vLLM | A | Prefill 对 Decode 的间歇干扰 | TPOT、TTFT |
| Deeper Async Scheduling | vLLM | A/C | CPU/GPU 执行间存在空隙 | TPOT、吞吐 |
| Dynamic Batch Tokenizer | SGLang | A | 高并发输入端分词受限 | TTFT、CPU、吞吐 |

前九项是本轮重点补充；后六项保留上一轮内容并重新标注前提。不是所有方法都要试，尤其不应同时开启多项互相作用的缓存机制。

## 3. 本轮新增：状态复用与内存分配

### 3.1 细粒度前缀匹配与共享 SSM 检查点（vLLM）

**基本思想：** 普通完整 Attention 为每个 token 保留 KV；线性 Attention、Mamba、Gated DeltaNet 等循环状态层则需要在特定位置保存可恢复的状态。即使相同输入前缀的 Attention KV 仍然存在，也不意味着所有混合层都能从任意 token 恢复。匹配位置、物理页大小和实际状态检查点是三件不同的事。

设两个请求共享 1500 个 token。如果只能每 1024 token 命中一次，最多从 1024 处恢复；如果匹配单位为 64 且在 1472 处有完整可恢复状态，则理论上只需要重算其后的部分。这里的数字是教学示例，不代表真实模型的状态布局。

支持该功能的 vLLM 可考虑：

~~~bash
--enable-prefix-caching \
--prefix-match-unit 64 \
--enable-mamba-shared-prefix-checkpoint
~~~

这是一组机制参数，不是任意模型的通用命令。先对照实际 cache group 的块大小；匹配单位必须满足各组的对齐/整除及模型特定约束。共享前缀检查点针对已经观测到的分叉恢复位置，并非所有潜在位置都预先保存。部分滑窗组要求按整块查找，不满足时显式设置可能启动失败。

**最小实验：** 原配置 → 仅改变合法的 prefix-match-unit → 在该配置上额外打开共享检查点。构造相同长前缀、不同后缀的 sibling requests，区分首次分叉与后续重复分叉，检查实际缓存命中位置和重复 Prefill token。单纯比较缓存 hash 命中不能证明完整混合状态可恢复。

**理论关系：** 状态可恢复性、缓存匹配粒度与物理状态成本。收益来源是减少原本需要重算的 token，而非加速 Attention Kernel。

来源：[vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)；[CacheConfig](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)。

### 3.2 按工作负载选择循环状态检查点密度（SGLang / vLLM）

SGLang 已经提供混合模型检查点级别选择。

~~~bash
--hybrid-cache-checkpoints-level 2
~~~

~~~bash
--hybrid-cache-checkpoints-level 3
~~~

上游现行文档中，level 2 为默认方案，保留主要分叉和请求结束位置；level 3 还增加 Prefill 分块及周期性 Decode 位置的检查点。包含 SWA 和循环状态的模型有额外例外。默认级别已开启时不能将其算作新增优化。

多轮线性追加通常更依赖历史结束位置，而分支式 Agent 请求可能需要复用早期中间状态。减少检查点可以释放状态空间、减少写入，但可能增加迟到分叉的重算。上游 SGLang 文档中列出了 Qwen3.8 系列和 GLM 混合模型的测量例子，其效果与特定 B300 工作负载及配置相关，不能外推到 H800。

vLLM 有另一个机制参数：

~~~bash
--prefix-cache-retention-interval N
~~~

N 是合法的正整数 token 间隔，需为调度器块大小的整数倍；默认 0 仅保留语义相关检查点，正值增加周期保留。该参数仅适用于 SWA 和 Mamba 缓存组，不是普通全量 Attention 的通用淘汰周期。

**验证：** 对线性多轮续写及中途分叉分别做 A/B，记录状态槽位、Prefill token、checkpoint 写流量、缓存命中、TTFT P95/P99；在同一总显存下比较。回退到原级别或原 interval。

来源：[SGLang Hybrid Model Prefix-Cache Checkpoints](https://docs.sglang.io/docs/advanced_features/hybrid_cache_checkpoints)；[vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)。

### 3.3 代价感知缓存淘汰：不仅仅 LRU（SGLang）

传统 LRU 按最近访问时间淘汰。对于有大量固定前缀及长会话的服务，更有价值的缓存未必就是最近访问的缓存。

| 策略 | 适用机制 | 参数 |
| --- | --- | --- |
| LRU | 一般负载与默认对照 | --radix-eviction-policy lru |
| LFU | 稳定高频前缀，不希望一次性流量冲掉它们 | --radix-eviction-policy lfu |
| SLRU | 将反复命中的缓存与尚未证明价值的新缓存分开 | --radix-eviction-policy slru |
| T-LRU | 结合下一轮未缓存 token 预算，优先降低多轮尾延迟 | --radix-eviction-policy tlru |
| State-cost | 考虑循环状态检查点的重新计算代价和访问历史 | --radix-eviction-policy state_cost |

**最推荐先看 state_cost。** SGLang 官方说明指出，该策略针对混合模型的 recurrent-state pool；普通全量 Attention KV 仍按 recency 管理。它拒绝在混合模型中与 session-aware radix cache 同时启用。策略灵感来自 FlashStates，默认成本参数基于另一种模型，应使用本模型的 Prefill 时间曲线重新拟合，不能照搬。

~~~bash
--radix-eviction-policy state_cost
~~~

可选的 --radix-eviction-policy-config JSON 内有 d、e 参数，对应基础和上下文依赖的 Prefill 成本项。先测定目标模型不同起始位置和增量长度的 Prefill 成本，再拟合并确认排序有意义。

另一个低门槛方案是 SLRU：

~~~bash
--radix-eviction-policy slru \
--radix-eviction-policy-config '{"protected_threshold": 2}'
~~~

T-LRU 需要给定下一轮输入增长和允许的未缓存 token 阈值：

~~~bash
--radix-eviction-policy tlru \
--radix-eviction-policy-config '{"threshold":4096,"next_prompt_estimate":512}'
~~~

此处 4096 和 512 仅是官方示例量级；实际参数应满足 threshold 大于 next_prompt_estimate，并依据真实 trace 及 TTFT 目标校准。淘汰候选只包含符合树依赖和运行状态约束的节点；策略不能永久 pin 内存。

**最小实验：** 固定状态池与 KV 池，回放相同会话 trace；至少包含稳定热点、热点迁移、长会话间歇和一次性请求。比较 LRU 与一个有机制依据的策略，记录真实重算量、P95/P99 TTFT 和缓存压力，不要只看 hit rate。

**理论关系：** 对缓存块 i，简化的容量归一化价值可写成

$$
V_i \approx \frac{p_i \max(0, T_{\rm recompute,i}-T_{\rm restore,i})}{M_i}.
$$

这是候选理论而非上游 state_cost 的精确公式。真实问题需考虑状态依赖、访问概率、替换成本和树结构。参见[状态理论](../../theory/memory/README.md)。

来源：[SGLang Radix Cache Eviction Policies](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)。

### 3.4 混合模型显存池重新分配（SGLang）

**这是最值得针对混合 Attention 模型做的一项低成本实验。**

SGLang 的混合模型需要为 per-token KV 和循环状态分别预留空间。这两种资源的需求增长规律不同：前者主要随上下文 token 增长，后者与运行请求数、每请求工作状态数、保留检查点有关。

设模型权重和运行时开销之后有可分配显存 M。简化为两池：

$$
M_{\rm KV}+M_{\rm state}\le M.
$$

若每个运行请求使用 s 个状态槽，每槽 m 字节，峰值运行数为 r，另需 c 个非重复保留的检查点，最简容量估算为

$$
M_{\rm state} \gtrsim (sr+c)m.
$$

实际还要加入投机状态、内存分配粒度和安全余量。该关系用于排除不合理分配，不是上游实现的完整分配器。

现行参数：

~~~bash
--mamba-full-memory-ratio 0.7
~~~

或固定状态池槽位：

~~~bash
--max-mamba-cache-size N
~~~

其中 ratio 是 state pool 与 KV pool 的内存比，而不是总 GPU 显存占比；不同模型可能有默认覆盖。N 要替换为根据启动日志和目标负载计算出的槽位数。不要同时改变两种控制方式。

**实验步骤：** 先记录当前两池大小、max_running_requests、实际工作状态数、状态淘汰、KV token 容量。若状态池限制短请求并发而 KV 空闲，增加状态池；若 KV 驻留不足而状态池有余量，适度缩小状态池，同时验证空闲会话检查点是否被挤出。0.7 只是可选试点，不应在未经测量的生产环境直接采用。

SGLang 文档报告了特定 GLM-5.3-Flash trace 在不同池分配下出现明显性能变化，这说明分配可能具有容量临界点，而不是保证 H800 可获得相同收益。

来源：[SGLang Hybrid Model Memory Pools](https://docs.sglang.io/docs/advanced_features/hybrid_model_memory)。

### 3.5 Session-Aware Radix Cache：不给多轮 Agent 的状态轻易淘汰（SGLang）

无需独立 Dynamo 路由器，可以先在单个现有 SGLang 实例内启用：

~~~bash
--enable-session-radix-cache
~~~

使用原生 /generate API 在同一会话的每个请求传入相同顶层 session_id；结束时调用 /close_session。完整 prompt 仍由客户端负责发送，session_id 并不自动拼接或恢复对话内容。

~~~json
{
  "text": "FULL_PROMPT_FOR_THIS_TURN",
  "sampling_params": {"max_new_tokens": 128},
  "session_id": "experiment-session-01"
}
~~~

这里 max_new_tokens 只是 API 格式演示，实际实验保持与基线完全一致。关闭会话，包括异常退出路径，需要客户端少量修改。该功能通过缓存引用建立软保护，不保证状态永不被淘汰；仅适用 UnifiedRadixCache，且与某些淘汰策略不兼容。不要把 session_id 当作租户认证或权限隔离机制。

**验证：** 在固定显存下回放多轮间隔请求，与不标记会话的相同 trace 比较；记录活跃和结束会话的命中、冷请求公平性、释放情况及重算时间。

来源：[SGLang Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache)。

## 4. 本轮新增：调度、投机与运行时

### 4.1 Adaptive Long Prefill Threshold：在长短请求间动态调整分块额度（vLLM）

长输入可能反复抢占一次迭代的 token budget。直接限制长 Prefill 每次处理的 token，能让短请求更快获得资源，但限制过严会增加长请求完成时间。自适应版本基于运行与等待请求数调整限制下界。

~~~bash
--enable-chunked-prefill \
--long-prefill-token-threshold 1024 \
--long-prefill-token-threshold-adaptive
~~~

该 threshold 在当前引擎语义下是长 Prefill 分块上限相关参数；自适应行为将阈值下限抬到 token budget 的公平份额。单独运行的请求不受这一限制。1024 为示例，先确认本地 CLI 和调度实现，不能假设所有历史版本均支持同名语义。

**验证：** 固定负载，分别比较原配置、静态限制、自适应限制；混合 1K 与 16K 等具有代表性的输入长度（不超过当前支持上下文），报告各请求组 TTFT、TPOT、输出吞吐和 SLO。回退为原配置。

来源：[vLLM serve CLI](https://docs.vllm.ai/en/latest/cli/serve/)；[调度配置](https://docs.vllm.ai/en/latest/configuration/engine_args/)。

### 4.2 Stream Interval：减少每个输出 token 的主机工作（vLLM）

~~~bash
--stream-interval 4
~~~

默认 1。将多个输出 token 合并为较少的流式传输事件，可以降低序列化、回调及网络发送的频率。这类似在输出侧做 Batch，不减少模型实际生成 token，也不直接提升 GPU Kernel 速度。

最简输出侧成本可用 $T_{\rm host}\approx aN_{\rm events}+bN_{\rm tokens}$ 表示；只有事件开销暴露在关键路径或消耗显著 CPU 时，减少事件数才可能提高系统吞吐。更大的 interval 会使客户端收到的流式输出更不连续，并可能改变观察到的首包与 token 间隔。

**验证：** 1、4、8 为单变量试点，记录 GPU 侧生成间隔、客户端 TTFT/流式 chunk 间隔、总输出吞吐、CPU 利用率；不能把每个 SSE chunk 误计为一个 token。尤其适合不要求逐 token 展示的批量任务或 Agent 工具处理路径。

来源：[vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)。

### 4.3 Dynamic Speculative Decoding：根据并发调整投机长度（vLLM / SGLang）

投机长度越大并不一定越快，因为目标模型的验证成本与实际 Batch 有关。若每轮有效提交 A 个 token，轮次耗时为 T，则稳定条件下单请求的生成成本约为

$$
{\rm TPOT}\approx \frac{\mathbb E[T]}{\mathbb E[A]}.
$$

应比较增加有效产出与增加草稿、验证、拒绝和回退成本之间的关系。

vLLM 已支持在**已经验证可用的投机配置**中加入以下字段。这里只是 JSON 增量示例，不是可独立启动的完整 --speculative-config：

~~~json
{
  "num_speculative_tokens_per_batch_size": [
    [1, 16, 3],
    [17, 64, 1],
    [65, 128, 0]
  ]
}
~~~

区间首尾均包含在内，末端的 0 表示该区间不进行额外草稿。范围和 K 必须根据现有 max_num_seqs、原有投机配置和扫描结果确定。官方明确仅对 EAGLE、EAGLE3、DFlash 等列出的方案做过测试，其他方法需检查；MoE + DP 多副本路径会禁用动态 K 并回退静态配置，不能假定 Qwen 原生 MTP 或 GPU N-gram 已兼容此功能。

SGLang 有另一种自适应实现：

~~~bash
--speculative-adaptive
~~~

要求现有 EAGLE/EAGLE3 且 speculative-eagle-topk=1，其他路径回退静态。该机制用按 batch 分组的接受长度统计调整 tier，需要额外 graph/state 容量，不能把它当作无需已有草稿机制的方案。

**验证：** 先对现有静态投机长度扫描不同并发，记录真实接受长度、draft/verify 耗时和尾延迟。只有最优 K 随 batch 改变时再尝试动态策略；保持精度和采样语义，拒绝 synthetic 或 lossy 验证。比较实际动态选中的 K，而非仅检查参数是否被解析。

来源：[vLLM Dynamic Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)；[SGLang Adaptive Speculative Decoding](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding)。

### 4.4 Extensible KV Cache：根据实际可用显存而非预估值分配（vLLM）

旧式初始化通常需要预估 CUDA Graph、临时激活和 KV 可用空间。新实现使用虚拟内存预留地址空间，在 graph capture 和 warmup 之后基于实际剩余空间分配物理页。

~~~bash
--enable-extensible-kv-cache
~~~

根据当前 vLLM 文档，在 CUDA/ROCm 支持路径中通常默认启用，因此第一步不是再加一行 flag，而是确认现有模式、实际分配的 KV token 和显存限制是否已经覆盖该机制。显式强制开启在不支持的组合中可能报错；部分 KV connector、driver 及手动固定缓存容量会影响适用性。

注意：不建议在共享 GPU 或未核对后续通信缓冲、其他进程、外部连接器分配的情况下直接取消显存上限。对独占设备且有证据表明旧的保守 cap 留下可用空间时，才在隔离实验中尝试新的分配方式，监控峰值和 OOM。

**验证：** 比较同样的启动、预热、graph、负载与真实 KV 容量。只有减少抢占或增加有效并发时才会转化为吞吐收益。不要把预留虚拟地址大小当作已分配显存。

来源：[vLLM Optimization and Tuning：GPU Memory Utilization](https://docs.vllm.ai/en/latest/configuration/optimization/)；[CacheConfig](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)。

## 5. 上一轮六项方案：复核与实验入口

### 5.1 Native CPU KV Offloading（vLLM）

~~~bash
--kv-offloading-size 32 \
--kv-offloading-backend native
~~~

32 表示 GiB，TP 多卡合计的 CPU offload buffer；具体当前版本解析以本地 help 为准。利用 CPU 缓存复用已经完成的前缀状态，不是将所有活跃 Decode KV 放在 CPU 上直接计算。使用内存前先确认宿主机可用 RAM、pinned memory、NUMA 和传输带宽。混合状态需要实际 connector 支持及恢复路径验证。

一个基础判断为：若 GPU 未命中，从 CPU 恢复的暴露耗时低于相应前缀重算的耗时，才有单请求收益。还需考虑缓存写流量和对其他请求的竞争。

可选的请求级研究：当前 OffloadingConnector 文档提供实验性 kv_transfer_params.max_load_tokens，设置 0 可禁止该请求从外部缓存加载而保留 GPU 命中，便于比较传输与重新计算；不应假定所有 native backend 或旧版本都支持同样的请求字段。

来源：[vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)；[KV Offloading Usage Guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)。

### 5.2 GPU N-gram（vLLM）

~~~bash
--speculative-config '{"method":"ngram_gpu","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'
~~~

借助当前文本中的重复片段生成低成本草稿，无需另装 Draft Model。此示例基于现行 speculative schema，实际 GPU N-gram 后端及混合模型的验证支持需核对。先关闭其他投机机制做同条件对照，避免重复算收益。若低接受率或高 batch 验证开销使 TPOT 变差，保留原配置。

来源：[vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。

### 5.3 KV Cache Watermark（vLLM）

~~~bash
--watermark 0.05
~~~

在等待或被抢占请求的准入阶段留出 KV 容量余量。当前默认 0。仅当缓存紧张、重复抢占显著时优先对照 0、0.05、0.1。限制准入可能减少并发、提高等待，因此必须同时测量 TTFT 尾部、preemption 和吞吐。

来源：[vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)。

### 5.4 Time-Sliced Prefill（vLLM）

~~~bash
--prefill-schedule-interval 3
~~~

每隔若干 engine step 才调度 Prefill，其他步仅处理 Decode；当前默认 1。无需物理 P/D 分离。长 Prompt 的分块执行可能被多次推迟，导致 TTFT 增加。适合发现生成间隔受 Prefill 干扰时对照 1、2、3，不应只凭 TPOT 改善接受方案。

来源：[vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)。

### 5.5 Deeper Async Scheduling（vLLM）

~~~bash
--async-scheduling \
--async-scheduling-depth 3
~~~

默认深度 2。加深 host/GPU 流水线的在途 step 数有助于隐藏主机往返开销，但也可能引入额外资源、调度和依赖限制。现有混合模型、投机、PP 配置能否组合需核对。若 GPU 时间线没有可隐藏的主机等待，不应预期显著收益。

来源：[vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)。

### 5.6 Dynamic Batch Tokenizer（SGLang）

~~~bash
--enable-dynamic-batch-tokenizer \
--dynamic-batch-tokenizer-batch-size 32 \
--dynamic-batch-tokenizer-batch-timeout 0.002
~~~

把短时间到达的多个分词任务组织成批量工作，适合 CPU 分词受限的高并发输入。32 和 0.002 秒为现行文档默认参数量级；低负载可能增加排队等待。验证原始 token IDs 和模板完全一致，观察 CPU、TTFT、端到端吞吐。

来源：[SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)。

## 6. 不应该先做什么

以下方案有研究价值，但不满足本轮“易部署”的优先筛选条件：

| 方案 | 暂不作为优先任务的原因 |
| --- | --- |
| Mooncake / FlexKV 集群、Dynamo 路由、完整 P/D 或 A/F 分离 | 引入新的服务、传输、拓扑及运维复杂度 |
| CacheBlend 等非前缀近似复用 | 不保证完整 Prefill 的语义等价，需独立质量合同 |
| 新的 Draft Model | 需要额外权重与运行路径；优先利用已经存在的原生能力 |
| CUDA Graph persistence | 主要改善重复启动，依赖额外 Foundry 及特定 Torch/CUDA 组合，不应当作稳态 TPOT 优化 |
| Breakable CUDA Graph 的源码级改造 | 新的 graph break 需要代码配合，且 Prefill 在当前 SGLang CUDA 路径已默认采用相关 backend |
| ReplaySSM | 有现成 --use-replayssm 参数且有混合架构研究价值，但历史 Qwen/H800 已有相关候选 T017；先查既有实验和当前实现，不重复归为新发现 |

来源：[CUDA Graph Persistence](https://docs.sglang.io/docs/advanced_features/cuda_graph_persistence)；[Breakable CUDA Graph](https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph)；[vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)。

## 7. 在实际 H800 / Qwen 项目里怎样选择？

首先不更换引擎，只核对当前 checkout / 镜像支持。下面的命令用于查看本地版本与参数，不启动服务：

~~~bash
python -m pip show vllm sglang
vllm serve --help > vllm-help.txt
python -m sglang.launch_server --help > sglang-help.txt
~~~

只运行当前已安装引擎对应的命令；必要时核对实际 server startup log、源码 commit 与模型 config。不要因为文档存在参数就强行升级整个依赖栈。尤其要确认 hybrid 状态管理、既有 MTP/N-gram、cache groups、TP/EP/DP 布局、graph 以及实际默认值。

### 7.1 条件式优先级

| 若观测到 | 优先候选 | 不满足时 |
| --- | --- | --- |
| SGLang 混合模型 state pool / KV pool 明显失衡 | 显存池分配 | 先保留 baseline |
| 存在中途分叉，命中对齐损失或缺失 SSM 状态 | vLLM 细粒度匹配与共享检查点，或 SGLang level | 做最小分叉重算测量 |
| 有大量循环状态淘汰与重算 | state_cost，与 LRU 独立比较 | 若不是状态池问题，不切换 |
| 高负载发生抢占 | watermark；必要时 native offloading | 先排查容量和默认配置 |
| Prefill 干扰 Decode | prefill interval / adaptive threshold | 对照原有 chunked prefill |
| CPU 或输出路径占用高 | stream interval / tokenizer / async depth | 不以低 GPU 算力利用率单独诊断 |
| 已有投机在不同并发下收益不同 | dynamic K（支持时） | 保留静态最优或关闭候选 |

对于之前关注的混合架构，尤其推荐**状态池分配、共享前缀检查点与代价感知状态淘汰**。这三者改变的是状态的可恢复性、保留价值和容量，而不是重新编写 Kernel。

### 7.2 最小对照协议

每次实验只改变一个机制（需要配套参数时将其记录为一个明确干预），使用当前最佳、已经验证的真实配置作为基线。

- 同一模型 revision、全部实际精度、GPU 拓扑、相同输入输出、采样与工作负载 trace。
- 区分冷缓存、热缓存与预热过程，至少做多次相同负载的 baseline / candidate / baseline；记录 GPU 时钟、主机资源与干扰。
- 输出 TTFT、TPOT（引擎侧与客户端侧区分）、输出/总吞吐、P95/P99、错误和完成率；缓存实验还记录实际重算 token、状态槽位、命中、淘汰和抢占。
- 验证参考语义、任务质量、缓存恢复/回退及多轮正确性。固定精度不代表不同 batch 或 kernel 的浮点运算逐位一致。
- 不将已经默认启用的功能登记为新增收益。启动不支持、特定负载没有收益或引入退化，都记录为有效负证据，并回退到已验证配置。
- 对两项分别有效的优化，再考虑 A/B/AB 组合实验；特别是 watermark 与 offloading、checkpoint 密度与池分配、Prefill 调度与异步/投机之间可能互相影响。

具体使用[Benchmark 协议](../../docs/benchmark-protocol.md)、[项目优化指南](../../docs/project-optimization-guide.md)和[现有 Skill](../../skills/llm-inference-optimizer/README.md)，事前冻结模型预测与验收条件，事后比较真实结果。不要为了完成数量而投入无证据候选。

## 8. 统一理论视角

这些轻量配置虽然看似开关，但背后是四类不同的资源变换：

| 理论机制 | 示例 |
| --- | --- |
| 以状态空间换重复计算 | checkpoint、CPU offloading、session-aware cache |
| 将有限容量分配给更有价值的状态 | state_cost、SLRU、双池分配、extensible KV |
| 以等待和调度粒度换服务稳定性 | watermark、time-sliced prefill、adaptive threshold |
| 改变每次有效输出的执行成本 | N-gram、动态 K、异步流水线、输出事件批处理 |

这些实例应进入统一的[优化机制](../../theory/optimization/mechanisms.md)和[收益耦合](../../theory/optimization/coupling.md)研究，但本次仅补充文献与工程使用指南，不更改结构化理论库、实际 Skill 逻辑或实验环境。

**证据边界：** 所有命令和机制以所引用上游公开文档为来源，示例值不是本项目结果。未固定全部上游 release/commit；部署前以实际版本 help、启动日志和对应源码为准。未执行真实 H800 实验，也不声称这些方案在目标模型上已经通过正确性验证或一定获得加速。

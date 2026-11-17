# 一手来源索引

检索日期统一为 **2026-11-16**。下列 `main` / 当前文档是滚动来源；实施前把实际依赖锁定到 release/git SHA。PR 是否合并只表明上游状态，不证明用户 build 已包含。

| 来源 | 支持哪些判断 | 使用边界 |
|---|---|---|
| [Qwen FP8 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | 官方目标名称与制品 | 本机 revision 待核对 |
| [模型 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json) | 架构、格式、各类模块配置 | 不是全部张量都为 FP8 |
| [Qwen 官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next) | QSA/GDN/GR/PLE/MTP 背景 | 架构效果不是本机调优收益 |
| [vLLM 模型 recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml) | 支持版本、布局/PLE 部署约束 | 不复制会改变 dtype 的参数 |
| [SGLang 模型 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx) | 该模型实现与参考配方 | 各硬件支持条件不同 |
| [vLLM #57097](https://github.com/vllm-project/vllm/pull/57097) | T003 QSA prepare 融合，已合并 | 本机是否覆盖未知 |
| [vLLM #59533](https://github.com/vllm-project/vllm/pull/59533) | T004 QSA 投影融合，SM90 小形状调度 | 不移植其他GPU微基准收益 |
| [vLLM #56322](https://github.com/vllm-project/vllm/pull/56322) | T005 GR/PLE sequence parallel | 需核对 PP/LoRA 等约束 |
| [SGLang hybrid checkpoints](https://docs.sglang.io/docs/advanced_features/hybrid_cache_checkpoints) | T008 混合状态与前缀复用 | 官方例子非 H800 本机数据 |
| [vLLM Benchmark CLI v0.32.0](https://docs.vllm.ai/en/v0.32.0/benchmarking/cli/) | TTFT/TPOT/stream gap 与缓存污染 | 实验固定实际工具版本 |
| [NVIDIA AIPerf metrics](https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference) | 跨工具指标公式与 first chunk 处理 | 名称相同也需核对公式 |

各 trick 卡片还包含对应的调度、CUDA Graph、通信、NUMA 等官方文档。研究排序、实验矩阵与验收流程属于本仓库的设计判断，来源不声称这些是 H800 的最优参数。

## 2026-11-17 · Prefix 缓存与压缩补充来源

| 来源 | 支持哪些判断 | 使用边界 |
|---|---|---|
| [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | token 前缀重复可免重算，hybrid Mamba 有特殊路径 | 具体已部署版本和后端需实查 |
| [vLLM Prefix 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/) | 基于前缀块/hash 的复用，支持隔离 salt | 完整键 LRU 仅是本仓库简化模型 |
| [vLLM 生产指标](https://docs.vllm.ai/en/stable/usage/metrics/) | prefix hits/queries 是 token Counter，按 tier 区分缓存来源 | 同窗使用计数器增量，注意版本 |
| [LMCache 专项配方](https://docs.lmcache.ai/recipes/qwen3_8_flash_next.html) | 混合状态 GDN align、QSA scratch、CacheGen/MTP 支持现状 | 8 GPU 文本路径验证不能外推到本机 FP8/H800 |
| [LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/) | 任务无关 prompt 压缩方法 | 不代表本模型质量或加速 |
| [LongLLMLingua](https://aclanthology.org/2024.acl-long.91/) | 长上下文与查询相关压缩 | 风险由实际任务决定 |
| [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) | 长上下文证据位置效应 | 不将位置敏感性简单当压缩结论 |

这些链接的“最新”页面可滚动变化，实测前须锁定版本/commit；原表仍反映 2026-11-16 检索快照。

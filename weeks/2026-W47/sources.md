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

# T008 — 混合模型的前缀 Checkpoint 与两池分配

- ID：T008
- 状态：researched
- 调研日期：2026-11-16
- 验证：待 H800 实测

## 问题与机制

QSA/Gated DeltaNet 混合模型复用前缀时，还需要对应的递归状态。只有 KV 命中不保证整段可复用。SGLang 官方 checkpoint 页面明确列出 **Qwen3.8-Flash-Next** 的例子，使用 B300，数字不能移植到 H800。[模型相关证据](https://docs.sglang.io/docs/advanced_features/hybrid_cache_checkpoints)

## 模型与版本门槛

[SGLang 模型配方](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx) 要求至少 0.5.22，但最低模型支持版本不保证已有 checkpoint 调节功能。核验实际 commit 的 `--hybrid-cache-checkpoints-level`：level 2 偏向分叉/请求结束位置，level 3 增加中间 checkpoint。固定模型 revision、所有 dtype、模板及采样，不改变缓存精度。

## 实施 Agent 步骤

1. 读取启动日志中的 state slots、每请求状态需求、KV token 容量及并发限制。状态池和 KV 池限制不同。[两池说明](https://docs.sglang.io/docs/advanced_features/hybrid_model_memory)
2. 先固定两池，独立对比 checkpoint level 2/3；测完整 continuation 与中途分叉两类复用。
3. 有明确池瓶颈后另开实验，只改 `--mamba-full-memory-ratio` 或 `--max-mamba-cache-size` 之一。多副本路由优化另设实验，不混入本次收益。

## 单变量 A/B

相同总显存预算与并发下，A/B 每次只改一个 checkpoint 或池参数。分别测引擎已预热但前缀为空、固定公共前缀、多轮 Agent trace。报告 TTFT 分位数、TPOT、输出 tok/s、真正复用 token 数、复算量、两池 eviction 和驻留 checkpoint 数。

## 正确性与回滚

检查续写、中段分叉、缓存驱逐后恢复及不同会话状态隔离；基线和候选输出须通过固定质量集。无支持路径、状态错位、OOM 或 SLO 回归即恢复旧配置并清空相关运行缓存。

## 已知与未知

已知：官方已有该模型的相关路径。未知：用户版本支持度、会话分布和最优池比例；没有 H800 收益，不照抄其他模型比例。

## 2026-11-17 · 命中率与天花板补充

**区分三类指标**：请求命中率、命中输入 token 比例和按 `cold−hit` 计算成本加权命中率。vLLM 当前官方 `vllm:prefix_cache_hits`、`prefix_cache_queries` 为**token Counter**，不是请求 Counter；应对同一监测窗口计数器增量取比，结合 `prompt_tokens_cached_by_source` 区分 device/host/disk/p2p。

同一 trace 按真实 QSA/GDN 状态 MiB 对比**有限容量 LRU → 无淘汰完整键 → 免费命中 oracle → 全 Prefill 消失的 Amdahl 极松上界**。不把完整键命中回放冒充真实 block-level checkpoint 性能；并发、two-pool、状态恢复和驱逐仍需引擎验证。

上游 [LMCache Qwen3.8-Flash-Next 配方](https://docs.lmcache.ai/recipes/qwen3_8_flash_next.html) 对 vLLM 8-GPU 文本路径标已验证，但该模型的 GDN `align` 属实验性、QSA scratch ring 不缓存、CacheGen 不支持、MTP 组合未验证。该验证**不等于**用户的 FP8/H800/具体版本已验证。

与 [T013 上下文压缩](T013-context-compression.md) 联合评估时，应重新计算压缩后的 prefix key 和状态容量，不相加 TTFT 改善百分比。完整推导与脚本见 [theory/cache-context-ceilings.md](../../../theory/cache-context-ceilings.md)。

# 前缀缓存与上下文压缩 trace 契约（schema_version = 1）

入口：[`cache_context.py`](cache_context.py)；合成示例：[`examples/synthetic-cache-context.json`](examples/synthetic-cache-context.json)。这是**与核心 Amdahl/容量 JSON 不同的输入契约**，不应把两者的 `baseline` 字段混填。

## 顶层字段

| 字段 | 类型 | 约束与解释 |
|---|---|---|
| `schema_version` | 整数 | 当前恰为 1 |
| `data_kind` | 字符串 | `synthetic` 或 `measured`；非自动验真 |
| `config.fixed_ttft_ms` | ms，正数 | 每请求相同的固定非 Prefill 低负载 TTFT；未拟合队列 |
| `config.cache_capacity_mib` | MiB，非负 | 一个简化 cache state pool 的 LRU 容量；0 表示无法存对象 |
| `config.lookup_ms` | ms，非负 | 每次查找的暴露开销；cold miss 也要支付 |
| `config.restore_ms_per_mib` | ms/MiB，非负 | 每个 hit 状态恢复开销，未模拟与其他操作的 overlap |
| `config.minimum_quality_pass_rate` | [0,1]，缺省 1 | 压缩候选的任务质量准入门槛，单纯 bool 均值并不等于任务分数 |
| `requests` | 非空有序数组 | **按请求到达顺序**；id 唯一，无并发回放 |

每个请求必须包含 `id` 和 `baseline`；若有 `compressed`，则**每条请求**都必须有，才可做配对比较。示意（下列数值仅表示字段，不是 H800 实测）：

~~~json
{
  "id": "synthetic-request-1",
  "baseline": {
    "prompt_tokens": 1000,
    "prefix_tokens": 700,
    "prefix_key": "salted-key-A",
    "state_mib": 30,
    "cold_prefill_ms": 100,
    "hit_prefill_ms": 35
  },
  "compressed": {
    "prompt_tokens": 510,
    "prefix_tokens": 350,
    "prefix_key": "compressed-key-A1",
    "state_mib": 15,
    "cold_prefill_ms": 55,
    "hit_prefill_ms": 24,
    "compressor_ms": 8,
    "quality_pass": null
  }
}
~~~

| 每请求字段 | 约束 | 含义 |
|---|---|---|
| `prompt_tokens` | 正整数 | 实际进模型的输入 token，包含模板 |
| `prefix_tokens` | 0 至输入长度 | 在**该方案**中可重用的完整前缀 token 数；不代表已命中 |
| `prefix_key` | 字符串/null | 相同 key 意味着完全相同的可复用前缀与隔离域；前缀为零必须为 null。压缩后的 key 单独计算 |
| `state_mib` | MiB | 对应混合 QSA/GDN 及相关缓存对象的**合计真实驻留量**；前缀为零时必须为 0 |
| `cold_prefill_ms` | ms | 完全冷请求的实测 Prefill（或明确标为假设）均值 |
| `hit_prefill_ms` | ms | 恰该前缀重用后仍需执行的剩余 Prefill；**不包括上面的 lookup/restore** |
| `compressor_ms` | ms | 压缩造成的额外暴露耗时，包括依赖的分词、模型调用及处理；只有压缩版 |
| `quality_pass` | bool/null | 这条配对任务是否通过预先规定的质量 gate；未测填 null，不能填 true |

哈希必须同时包含模型/tokenizer/template/version、缓存隔离域以及**实际 token 序列**；仅语义相似不能标成同一 key。完全相同 key 的 `prefix_tokens` 和 `state_mib` 必须相同，脚本会拒绝含糊的伪碰撞。为安全起见用盐化、脱敏的不可逆键值；不将真实系统提示词、私密文档或日志提交公开仓库。

## 结果解释

| 结果 | 正确读法 | 禁止外推 |
|---|---|---|
| `request_hit_rate` | cache 命中请求数/总请求数 | 不等于官方 vLLM 的 token hits/queries |
| `cached_token_fraction` | 命中的前缀 token / 总输入 token | 不代表总 Prefill 成本减少同比 |
| `reusable_compute_hit_fraction` | 命中潜在可省 Prefill 成本占比 | 尚未减查找/状态恢复 |
| `ttft_mean_ms` / `ttft_speedup` | 该简单 trace 情景的平均低负载 TTFT 与对冷原文比率 | 不代表 p95/p99、饱和输出 tok/s |
| `relative_to_uncompressed_lru_speedup` | 压缩版相对原文同容量 LRU 的比率 | 未通过质量 gate 时只是条件改善 |
| `qualified_ttft_speedup` | 只有全部质量结果已知且通过指定门槛才返回值 | bool 门槛不足以替代真实综合评分 |
| `ideal_free_hit_exact_key` | 同键无限缓存 + 免费命中，乐观 oracle | 不代表 block-level 或真实多池全局上界 |

结果另有 `absolute_all_prefill_removed_speedup`：全部 Prefill 无条件消失的 Amdahl **极松上界**。这个数字并不代表缓存或压缩可以实现的收益。

## 严格的简化边界

1. **完整键命中**：一个请求只有一个可复用的完整前缀对象；不支持更短公共块、前缀树、不同层按块对齐。
2. **按序单请求**：不建模并发时的 pinned state、请求排队、prefix 在处理中失效、多租户缓存隔离开销。
3. **单内存池 / 单 restore tier**：不模拟 GDN/QSA 物理两池、device/host/disk/p2p 带宽分层；实际版本在这些方面不同。
4. **固定非 Prefill 时延**：压缩可能影响首 token Decode、实际输出长度和后续 TPOT，当前需要单独测量再接回主容量预测器。
5. **质量布尔 gate**：只表示测试集的准入，不表示用户业务已无信息损失；未测结果永远为 null。

若要获得可投产预测，先按 [研究与数据收集任务书](../../theory/foundations/cache-context-ceilings.md) 的 P0 表采集真实 trace，再以分组/位置/长度的留出集校验误差。

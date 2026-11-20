# T029 · GDN / PLE 状态页去统一尾填充

实施前先读 [optimize-qwen-h800 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行本卡的一个阶段，通过检查点后再继续。

- 状态：`conditional`；日期：2026-11-20（Asia/Shanghai）。
- 证据：`literature + hypothesis`；H800 实测：无；优先级：容量瓶颈时的储备。
- 目标：在同一显存预算下减少状态页物理空洞；先验证实际 bytes，再看同负载尾延迟/容量。
- 固定：所有状态内容/dtype、逻辑 block token 数、prefix 策略、checkpoint 频率、MTP、请求 batch、GPU 预算。

## 新点：现成 packing 之后的剩余填充

固定 vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`。当前 [QSAIndexerBackend][qsa] 已支持 BLNHC/BLHNC；raw key 用 CircularBufferManager，compressed key 以 `tokens_per_state=compress_ratio` 加入 FullAttentionManager。[通用 planner][planner] 已有 `_get_packed_kv_cache_groups` 与异构页物理打包。因此“打开异构 allocator”是 baseline 已有能力，不能计为新 trick。

待验证缺口是：[Platform._align_hybrid_block_size][platform]取模型最大 Mamba 页后设置统一 `mamba_page_size_padded`，[MambaBase.get_kv_cache_config][mamba]将它套到各 MambaSpec。[Qwen4Exp.get_mamba_specs_from_config][model]明确有 GDN 与 TP-replicated PLE 两类实际大小，当前没有 `get_packed_hybrid_block_size`；[Kimi K3][kimi]已用该 hook 跳过统一填充。这仅证明存在适配入口，不证明 Qwen4Exp 的最终物理预算一定下降。

本卡只把 **GDN/PLE 页改为各自真实大小加必要对齐，再交给现有 packed planner**。不改 QSA raw/压缩格式，不新做压缩，不复用状态计算；不同于 T008 的前缀命中/checkpoint/两池比例，也不同于 R01 ReplaySSM。实际 build 已经消除该尾填充即标 `baseline-covered`。

## 模型容量账本与前置分支

本模型 36 层 GDN、12 层 QSA、1 个 PLE 模块。GDN recurrent 核心为每层 `48×128×128×4` 字节，实际本地页再加卷积/MTP/对齐；PLE 状态 TP replicated，不能除以 TP。QSA 还包含 raw ring、compressed index 与可选精确 MRoPE 位置，不能只计算普通 KV。[模型字段来源][config]。

对 rank 上每类真实页 `p_l`、当前填充页 `P_l` 和实际同时存活逻辑块数 `n_l`，`Σ n_l(P_l-p_l)` 只是局部空洞账本。**不是可兑现释放量**：packed 组从同一 backing pool 的 byte 0 叠放，块一次只能属于一个组；实际必须比较 `B_A×N_A` 与 `B_B×N_B`，其中 B 是 planner 的物理 block stride，N 是同一 trace 的池占用。组被拆更多可能抵消页变小；QSA 组若决定最大 stride，减少一页也可能不省一字节。

## 实施步骤（最多六步）

1. **C0 CPU 账本**：从已运行配置导出每 rank 的真实 `KVCacheLayerConfig`、group/manager、block_size、state shapes/dtypes、page_size_padded、offset/stride、prefix/MTP 保留块数。用当前 planner 原样重建；不得只凭 36:12 比例改组，GDN/PLE/QSA 的 manager 语义不同。
2. 同一逻辑 block/checkpoint 下，仅模拟移除 GDN/PLE 的公共尾 padding，保留各自 state/block stride alignment；重跑现有分组和同一分配/释放 trace。若真正 pool bytes、可用 slots 或碎片无改善，记录负结果退出，不继续写 GPU 代码。
3. **C1 最小 commit**：在目标模型专用 spec 生成处加默认关闭的分支，将 `page_size_padded` 改为合法的独立值或 None；不全局修改 MambaBase。`_align_hybrid_block_size` 若需适配，返回/保留基线已解析的 block token 数，首项不缩 block、不调池配额。记录修改前后逐层 spec diff。
4. 审计 `get_kv_cache_config_from_groups`、`MambaBase.bind_kv_cache`、QSA cache bind、state copy/swap：所有 view 指向同一预期 allocation，`view/reshape` 不产生隐藏复制；同组页不重叠，跨组 alias 只在互斥 block ownership 下成立。核对 Graph 捕获地址/stride、释放与重用 generation；有 connector/PP 时逐项检查 packed layout 限制，不绕过校验。
5. 回放 prefill→decode、prefix 命中/分叉、取消/抢占/恢复、MTP 全拒绝/部分接受、零填充行；对比 GDN/PLE 有效 state、QSA 索引/位置、输出。沿用现有 [packed-cache 测试][tests] 的 bound-view/offset 检查补真实模型形状，不能把已覆盖测试当收益。
6. **C2 A/B**：相同权重、Graph、batch、KV/state 预算和 trace，唯一变量是 state 页物理规格；按[协议](../../../docs/benchmark-protocol.md)至少三次交错。先比较真实峰值和尾延迟；释放量确认后另开第二组试验扩大可用 slots，不能同时加 batch 制造首项收益。

## 验收、退出与交付

主报每类真实/填充 bytes、组数、物理 block stride、峰值活跃块、alloc/free 碎片、实际可用 state slots；服务报 TTFT/TPOT p95/p99、输出吞吐、抢占/OOM。纯布局通常不降低算术量；可能只增加容量，也可能因 stride、局部性或更多 metadata 变慢，无固定百分比预测。

只有合法 state 页缩小、**整体**池使用改善且完整状态/输出回归通过才继续。任一 backend 需连续页、Graph alias 失效、MTP ring/检查点对齐变化、总分配不降或尾延迟越界则退出。回滚模型 spec 分支并重启重建原 cache/Graph，不在线解释旧布局。交付 CPU 分配账本、唯一 spec diff、布局图/别名审计、A/B 记录；用户实际 layout、实现 commit、收益均 `unknown`。上游是通用 CUDA 物理布局机制，无本模型 H800 实测证明。

[planner]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/v1/core/kv_cache_planning.py
[platform]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/platforms/interface.py
[mamba]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/mamba/abstract.py
[model]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/nvidia/model.py
[qsa]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/common/qsa_indexer.py
[kimi]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/kimi_k3/common/dsa_cache.py
[tests]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/tests/v1/core/test_contiguous_kv_packing.py
[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json

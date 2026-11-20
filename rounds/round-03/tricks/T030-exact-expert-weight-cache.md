# T030 · 原字节专家权重的按需驻留

实施前先读 [optimize-qwen-h800 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行本卡的一个阶段，通过检查点后再继续。

- 状态：`conditional`；当前上游移植到 EP/完整 Graph 的分支 `blocked`。
- 日期：2026-11-20（Asia/Shanghai）；证据：`literature + hypothesis`；H800 实测：无。
- 优先级：高风险容量储备；先做 trace 否决，不列立即 GPU 实验。
- 固定：FP8 权重字节、block128 scale、activation/math、top-10、专家归属、并行拓扑、Graph、总 GPU 预算。

## 新点与上游已做事项

本卡保存 immutable host 权重，用较少固定地址 GPU slots 驻留完整专家；命中复用，miss 搬原始权重和对应 scale。不改变哪个 rank 计算哪个 expert，区别 T010 EPLB；不是 T006 的 PLE 预取，也不是 T020 PLE 行缓存。vLLM [现成 UVA/分层 prefetch 配置][offload]已经存在，直接开启不算新项。

主线入口固定 `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f` 的 [Fp8MoEMethod.apply / get_fused_moe_quant_config][fp8]及 [FusedMoEModularMethod.apply][modular]，实际选中哪个先记录。新点是 provider 在原专家 kernel 前给出**原布局、slot 索引的 weights/scales**，外层 batch 与原 kernel 一次调用均保留。

[RFC #38256][rfc] / [PR #37190][pr]仅作参考；PR head 固定 `c66868c8c12f10fa931e108c3a8128ccddfdcdbb`。正文落后于源码：最新 [FP8 接线][prfp8]已允许未重排 backend 的完整 block scale 行；[config][prconfig]允许 MoE 在 piecewise Graph 段间 eager 执行，并可自动降级 FULL。不能据正文说“只支持 per-tensor / 必须全部 eager”，也不能接受自动降级作为同 Graph 对照。[RoutedExperts 校验][prrouted]仍拒绝 EP、DP、SP；[provider][provider]使用 `.unique().tolist()` 同步 host。没有本模型 H800 的可直接用实现或收益证据，外部小卡模型结果不外推。

## 本模型的容量与流量门槛

[官方 config][config]：H=2560、I=640、512 experts、top-10、48 层。完整 routed expert 的 FP8 权重为 `3HI=4,915,200 B`；若 scale 为 FP32，`4×(10×20+20×5)=1,200 B`，合计 `W_e=4,916,400 B`，未含 allocator 对齐/转置副本。EP8 只作算例：64 experts/rank/layer；48 层原权重约 14.0625 GiB/rank，另加 scale，不是本机容量结论。shared BF16 expert 保持原驻留、每次只执行一次。

令每层本地所需 unique 数 U、slot 数 C、miss 数 m，首次仅允许 `U≤C`、原 kernel 完整运行一次。`Σ_l m_l W_e / BW_H2D` 是无重叠搬运时间下界；还有路由同步、映射、传输启动和最慢 rank 等待。均匀路由的**反例算式** `U≈512[1-(1-10/512)^B]` 显示 batch 增大会快速扩大工作集，不能把少数 top-10 当成永远只搬十个专家。真实命中率必须来自本模型 trace。

## 实施步骤（最多六步）

1. **C0 trace**：只记录脱敏 layer/rank/expert ID、顺序、复用距离与 unique 集，模拟 C 和固定 LRU/LFRU；同时核 HBM 是否限制实际 state slots，测已有 H2D 带宽/暴露等待。没有容量压力、预估 miss 成本已超过可省时间，直接 `parked`。
2. **支持门槛**：核实际 prepare/finalize、FP8 layout、scale dtype、EP/DP/SP/Graph。已运行等价 provider 则 covered；EP 或 FULL Graph 不得删除上游断言后硬开。若需要改变拓扑/Graph，当前卡停止，另出适配设计；不得把这些代价藏进 cache 收益。
3. **C1 最小 commit**仅在兼容原路径：单层 provider，CPU immutable backing、固定 GPU slots、`layer+expert+weight_revision` 标签与 generation。weights 和 scale 同步换入；量化 config 持有的 scale view 必须指向 slot buffers，不能只替换 `w1/w2`。不改原字节/stride，不引入重新量化。
4. 所有本次 unique 专家均 pin 至 FC1/FC2 完成；禁止驱逐仍被 kernel/Graph 引用的 slot。审计并发调用、capture address、host pinned RAM、GPU cache/metadata/临时 staging 与旧完整权重是否真正释放。shared expert、通信/late all-reduce 仍各执行原次数。
5. 必须禁止 PR 的 `expert` split：它先把各组结果转模型 dtype 再 FP32 求和，改变舍入；`token` split 属 T019/R03，本卡也不启用。容量不足在执行前明确退出该候选并恢复基线，不丢专家、不静默分段。对错配/缺 scale、热点突变、连续 miss、重复调用及取消做原字节校验与输出回归。
6. **C2 A/B**：相同 batch、Graph 与 state 池预算，唯一变量为权重驻留 provider；先测单层完整生命周期，后按[协议](../../../docs/benchmark-protocol.md)三次交错服务回放。释放 HBM 后再独立试扩大 slots；若现有 Graph 无法保留，交付 `blocked` 证据即结束，不自动改 eager。

## 验收、回滚与未知项

交付 cache miss 曲线、每层 U/C、raw/scale hash、mapping/eviction 审计、峰值 HBM/host RAM、H2D bytes、逐层同步与最慢 rank stall；服务报 TTFT/TPOT 尾部、输出吞吐、错误/OOM。只有固定预算下收益超噪声且 math/质量/功能通过才接受；“显存少了”不足以证明吞吐提升。

容量不足、trace 域迁移使 miss 风暴、Graph/并行支持缺口、量化 config 悬空引用或释放量被 staging 抵消均退出。回滚 provider，重启恢复原完整权重和 Graph，不复用旧 slot map。实际部署 backend、完整 Graph 使用、EP 规模、命中率、带宽与实施 commit 均 `unknown`。本卡允许得出“不适用”，不承诺 offload 必然提速。

[offload]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/config/offload.py
[fp8]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/quantization/fp8.py
[modular]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py
[rfc]: https://github.com/vllm-project/vllm/issues/38256
[pr]: https://github.com/vllm-project/vllm/pull/37190
[prfp8]: https://github.com/vllm-project/vllm/blob/c66868c8c12f10fa931e108c3a8128ccddfdcdbb/vllm/model_executor/layers/quantization/fp8.py
[prconfig]: https://github.com/vllm-project/vllm/blob/c66868c8c12f10fa931e108c3a8128ccddfdcdbb/vllm/config/vllm.py
[prrouted]: https://github.com/vllm-project/vllm/blob/c66868c8c12f10fa931e108c3a8128ccddfdcdbb/vllm/model_executor/layers/fused_moe/routed_experts.py
[provider]: https://github.com/vllm-project/vllm/blob/c66868c8c12f10fa931e108c3a8128ccddfdcdbb/vllm/model_executor/layers/fused_moe/expert_weight_provider.py
[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json

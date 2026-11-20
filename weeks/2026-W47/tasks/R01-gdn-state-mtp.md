# R01 · GDN 状态重构与原生 MTP

> **辅助执行：** 先读 [optimize-qwen-h800](../../../skills/optimize-qwen-h800/SKILL.md)，从本工作流选择一个阶段和对应 T 编号。每次只做一个变量，不将整条工作流一起叠加测试。

- 状态：`conditional`；后续 adaptive 阶段仍为 `blocked`。本卡承接 [T017](../tricks/T017-replayssm-fixed-precision.md) 与 [T015](../tricks/T015-adaptive-verification.md)，不新增 trick ID。
- 更新：2026-11-20（Asia/Shanghai）；证据为 `literature` / 源码核验 / `hypothesis`，无 H800 实测。仓库基点 `be556f74a34d32daab28754183c1a1dff2823c2b`。

## 目标与非目标

在用户现有 Qwen3.8-Flash-Next-FP8 / Qwen4Exp + H800 中，用 checkpoint 与短历史减少 36 层 GDN 写回及投机快照。固定 checkpoint、权重/激活/卷积/SSM/KV/indexer/PLE、所有新增 buffer/correction/累加精度及 math mode；不改 MTP 提案器、采样、thinking、回答长度或 GPU 预算。

主性能基线是 **Week 1 当前最佳有效配置**，另保留最佳静态 MTP 作机制对照。非投机单层仅用于正确性与瓶颈诊断，不能用它替代已加速的 MTP 基线。实际 runner 未知；升级引擎或切 runner 必须独立归因。

## 本次新核验：不能把 PR 正文当可运行接口

| 固定证据 | 本次落实的结论 |
|---|---|
| [current main][main] `aefea5c8e00c47032cda6d0b75704f6cd0a7c66d` | 目标 model/GDN 仍无 ReplaySSM 接线；PLE、QSA/indexer 的 adaptive 能力缺口仍在。旧卡 `12924016…` 是初筛证据，不要求部署升级 |
| [#62004][pr62004]，open，head `0880cb87383330070f6a2546536ee99496306bab` | 仅普通 GDN decode，配置拒绝投机；默认 BF16 activation 的 d/k ring 却存 FP16。正文称 V1 范围及 host cursor，但 head 已是 GPU planner；runner 支持须沿实际入口验证，不能写成 V2 硬拒绝 |
| [#65657][pr65657]，open，head `682a577e7e50bfcc9786544152244ece7f2c7be2` | 有 Qwen4Exp 接入与状态生命周期，但要求 SM100、FP16 state，且排除 adaptive、PP、microbatch 与 KV connector；不能作为本机可直接开启的实现 |

新增精度反证：[dtype calculator][dtype]将 d/k 设为 FP16；[重建 kernel][kernel]把 FP32 输入的 `tl.dot` 设为 `tf32x3`。只改 recurrent state 或 ring 的 dtype 不足以证明原计算精度不变。先逐项对照原算术；需要完整 FP32 时采用 IEEE 实现或经核验满足原精度合同的路径，重新测量。原论文 FP32-state 数字含 FP16 cached corrections，不能作为本任务收益预测。

## 前置 profile 与准确落点

复用 W1-best 的 commit、负载与已有 profile，只补各 rank 的 GDN 层数、head 分片、实际 dtype/math mode、state page bytes、Graph/runner、普通与投机 dispatch，以及核心 kernel、状态复制、cursor 准备的完整链耗时。

本模型 recurrent shape 为 `[48/TP,128,128]`；按 FP32，全模型聚合 108 MiB/request，本 rank 用 `n_local_GDN×(48/TP)×128×128×4`，不含其他状态或 padding。只有写回/回滚在真实关键路径显著才继续；小 batch 被 MoE/launch 主导时暂停。

入口是 `QwenGatedDeltaNetAttention` 的普通 decode 与两条 spec 分支，模型 `get_gdn_mamba_state_*` / `get_mamba_specs_from_config`、`GDNAttentionMetadataBuilder`、`MambaBase.bind_kv_cache`。新增控制参考 [GPU planner][planner]与[按 cache group 绑定 tracker 的代码][trackers]：每组仅首个执行层推进 cursor，其余层共用 metadata；prefill 重置，align block 首步重锚、末步强制 flush。不能照抄正文的 `decode_step mod L`，也不能每层各推进一次。

普通 decode 与 MTP 的内存收益必须拆开：前者新增 ring，不保证省显存；后者才可能用一份 checkpoint 替代逐候选副本。记录原池与新池的实际页宽、组内 padding 和可容纳请求数，不能把 108 MiB 直接乘草稿深度当作已释放容量。配置不变但 allocator 重新对齐，也要完整登记。

计时使用真实 serving batch；冷 L2 单 kernel 只帮助判断带宽上限。另采一份包含输入整理、cursor、状态提交及相邻 stream 等待的阶段记录，避免核心 kernel 变快而整体链变慢。只复用已授权环境；取不到必要记录时交付阻塞清单，不凭其他模型的 batch 数选择用户负载。

## 最小 commit 顺序

### C1 · 非投机单层 kernel

唯一变量：原 packed recurrent kernel / 固定精度 replay，先固定一个短窗口。用该模型真实 TP shape 和相同 post-conv 输入，比较输出与**重建后的有效 state**；未 flush 的旧 checkpoint 不能直接与最新基线 state 比较。

交付 kernel、精度 manifest、多次 wrap/flush、异步行游标及 null slot 的病例。通过原数值合同且完整链存在净节省才进入 C2；错误或重算成本抵消收益则退出。回滚只关闭单层 dispatch，原状态路径可独立运行。

### C2 · 普通 decode 的 paged 生命周期

唯一变量：同一 C1 kernel 的引擎状态接线；此阶段仍不改 MTP。给 GDN page 加 ring/cursor，逐层识别 spec；PLE 维持独立、TP replicated 的 canonical 卷积格式。全局 replay 会减少原 speculative blocks，所有消费者适配前不得先缩旧池。

交付 prefill→decode、最后短 prefill、prefix materialization、block 边界、preemption、slot reuse 与 Graph 重放验证。窗口不整除 block 也必须正确；checkpoint 发布不得包含未接受尾部。失败则回退绑定/metadata，并重建该实验状态池，不能让旧代码读取新布局。

给每个恢复病例保存逻辑 token 位置、checkpoint 锚点、ring 有效区间和预期提交位置。包含同一物理 slot 被新请求复用、部分 prefill 恢复、不同请求在同一 Graph 中分别 flush/不 flush 的交错序列。复用旧页不等于复用旧游标；读取无效尾部或 padding 都算失败，不能靠重新分配全新请求掩盖。

### C3 · 现有最佳固定深度 MTP

唯一变量：固定原 K、提案器、验证器与调度，只替换 target GDN 的状态更新。参考 [PR 的 MTP 配置隔离][mtp]：attention-only drafter 必须独立禁用 replay，不能继承 target 全局开关；[worker 提交][commit]在所有层执行后统一发布接受游标，不能套用 C1 的提前推进逻辑。

交付接受 0…K 个草稿、分别拒绝首/中/尾、零草稿步、mixed prefill/decode、padding 不更新状态、QSA scratch 与 PLE 历史校验；接受长度沿用 runner 原定义，不混用 bonus 和 draft 计数。还要核对实际 page/block/ring 对齐：PR 的 K=7 曾因 448 不能整除 QSA ring 12 而启动失败；其 192→576 只是外部案例，不照搬为本机配置。

现有 K、PP 或 connector 无法保持时记 `blocked`，不降 K 或关功能制造同一 A/B。通过正确性后，至少三次交错比较 W1-best 与 candidate；回退 target dispatch 并清理不兼容状态池。

### C4 · Adaptive 后续阶段，当前 blocked

先单独修 PLE 的 device query boundaries，再修 QSA/indexer 的 varlen Graph；静态长度回归通过后，才只改变长度选择。保留完整 draft logits，不叠加 argmax 裁剪、router 校正或新 GDN kernel。能力检查仍失败就停在 C3，不绕过断言。

## 验收、退出与可复制摘要

按[协议](../../../docs/benchmark-protocol.md)报告 TTFT/TPOT、输出/总吞吐、stream gap、错误与显存；额外报 flush p95/p99、ring/page/padding、实际 state slots、接受长度和准备/复制成本。`S_incremental = throughput_candidate / throughput_W1_best`。单 kernel 变快、allocator 推算容量增加都不等于服务吞吐提升；收益须超过重复波动且质量/尾部约束通过。

```text
实现 R01：沿用当前 Qwen4Exp/H800 与 W1-best，先 profile，再按 C1→C2→C3 提交。
C1 保持原计算精度，C2 处理 per-layer state/cursor 与缓存生命周期，
C3 保持原生 MTP 的原 K，仅改 target GDN；adaptive 当前不实施。
增加独立实验开关，名称在实现 commit 中登记；本卡未提供可直接运行的上游 flag。
交付每阶段 commit、配置 diff、dtype/math-mode manifest、baseline/candidate IDs、
正确性/性能证据与回滚结果；没有 GPU 权限只交适配方案，不填性能数字。
```

`implementation_commit / profile / quality_pass / H800_gain = unknown`。

[main]: https://github.com/vllm-project/vllm/tree/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d
[pr62004]: https://github.com/vllm-project/vllm/pull/62004
[pr65657]: https://github.com/vllm-project/vllm/pull/65657
[dtype]: https://github.com/vllm-project/vllm/blob/0880cb87383330070f6a2546536ee99496306bab/vllm/model_executor/layers/mamba/mamba_utils.py
[kernel]: https://github.com/vllm-project/vllm/blob/0880cb87383330070f6a2546536ee99496306bab/vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py
[planner]: https://github.com/vllm-project/vllm/blob/0880cb87383330070f6a2546536ee99496306bab/vllm/model_executor/layers/mamba/ops/ssu_dispatch.py
[trackers]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/v1/worker/replayssm_utils.py
[mtp]: https://github.com/vllm-project/vllm/blob/682a577e7e50bfcc9786544152244ece7f2c7be2/vllm/models/qwen4_exp/nvidia/mtp.py
[commit]: https://github.com/vllm-project/vllm/blob/682a577e7e50bfcc9786544152244ece7f2c7be2/vllm/v1/worker/gpu/model_states/mamba_hybrid.py


# T015 · Qwen4Exp 动态验证预算：先解决 PLE 与 QSA 阻塞

- 状态：`blocked`；降为后端移植储备，当前不能作为直接开启的第二优先实验。
- 调研日期：2026-11-20；证据：`literature` + 源码静态核查 + `hypothesis`；没有 H800 实测。
- 阻塞范围：下述固定 vLLM SHA 的 Qwen4Exp 路径；不代表其他引擎或未来版本普遍不可能。
- Week 1 全部尝试为 `user_report`；逐项结果、当前最佳配置及 baseline ID 均为 `unknown`，不重做首轮 13 项。

## 模型与版本：通用 MTP 文档不足以证明可用

固定审计 [vLLM `12924016eb8dbaae87d5163a1b7ec2614a44a797`][engine]；[v0.34.0 官方说明][adaptive-doc]仅作为算法与接口依据。[目标配置][model-config]的架构是 `Qwen4ExpForConditionalGeneration`：48 层中有 36 层 GDN、12 层 QSA，`ple_layer_ids=[2]`，原生 MTP 有一个 full-attention 层。目标 checkpoint、各类 dtype 和原生 MTP 均不替换。

[T009](T009-native-mtp.md) 已扫描静态深度。本卡新增每步根据接受概率与实际执行成本分配验证长度，目标最大化每秒发布 token；不缩短回答，也不改目标验证规则。但源码表明当前模型不能直接启用。

## 两层阻塞，以及已支持的部分

`maybe_create_adaptive_verification_manager` 在[初始化代码][adaptive-code]中依次调用 `get_query_lens_mismatch_unsupported_backend` 和 `get_varlen_cudagraph_unsupported_backend`。即使模型直接指定 backend、绕过普通 selector，这里仍检查实际 attention groups 并抛出错误。

| 路径 | 固定 SHA 的具体证据 | 结论 |
|---|---|---|
| PLE：第一层阻塞 | `Qwen4ExpPLELayer.get_attn_backend()` 返回 `PleShortConvAttentionBackend`；它继承 `ShortConvAttentionBackend.is_ssm=True`，没有覆盖能力方法 | 基类 `supports_device_cpu_query_lens_mismatch()` 返回 `not is_ssm()`，因此为 false |
| QSA / indexer：第二层阻塞 | `Qwen4ExpQSAMetadataBuilder`、`QSAIndexerMetadataBuilder` 都是 `UNIFORM_BATCH`，没有可用的变长 Graph 上界；indexer 的 `build()` 使用 `require_uniform=True`，断言非零 decode query 长度相等 | 即使解决 PLE，当前 QSA 路径仍不满足动态验证的变长 Graph 条件 |
| GDN | `GDNAttentionBackend` 显式允许 device/CPU 长度不同；builder 提供 `get_varlen_cudagraph_max_query_len` | 不能笼统归因于“混合模型的 GDN 不支持” |

上述证据分别见 [PLE 后端][ple]、[基类能力默认值][backend]、[QSA builder][qsa]、[indexer builder][indexer]、[GDN builder][gdn]。这是源码推导的启动阻塞，尚未在用户服务器运行复现。

## 若继续移植，必须完成什么

首先处理 PLE 的短卷积状态计划、实际 device query boundaries、上一轮接受长度和请求 slot 映射；其次处理 QSA/indexer 的非等长 decode、位置与压缩组、scratch ring 和固定 Graph buffers。只把 capability 改成 true，或把断言删除，不构成实现。FULL Graph、PP/LoRA 等限制按实际 build 核对；本机卡数和并行布局未知。

兼容性修复先在**静态长度**下与原路径比对，确保修复本身没有改变模型行为。再让动态控制器接入，保留标准拒绝采样和所有原精度；暂不更换 GDN kernel、接入 ReplaySSM 或改 router。

**T014 冲突必须保留：**无 confidence head 时，在线接受率估计需要完整 draft logits，官方说明与 `use_local_argmax_reduction` 不兼容。不能为融合 argmax 删掉这些 logits，也不引入新训练 drafter/head。T014 若仅优化目标端，需另核共享 LM head 与通信路径，再单独做组合对照。

## 解锁后的最小 A/B 与收益公式

同一完成移植的 build 上，A 使用 Week 1 最佳静态 MTP，B 只改变每步验证长度选择。若 Week 1 最佳有效配置没有投机，则它仍是主性能对照，A 是额外机制对照。升级引擎时先复现静态结果，升级收益与控制器收益分开记录。

令一轮发布 token 数为 A、完整轮耗时为 T，优化 `E[A]/E[T]`，T 包括 draft、verify、状态提交和控制。无提前 EOS 的简化估计为：

```text
E[A_K] ≈ 1 + Σ(j=1..K) Π(i=1..j) p_i
S_incremental = R_adaptive / R_week1_best
              = (R_adaptive / R_static_best) × (R_static_best / R_week1_best)
```

p_i 是前序均接受的条件概率，R 是同负载累计已发布 token / 总时间。不能只报告相对无投机基线，也不能把各轮吞吐比值简单平均。逻辑 K 变小、实际仍执行全部 padding 时，不算减少验证工作。

选主要真实负载和一个低/高并发回归组，至少三次交错重复。按[统一协议](../../../docs/benchmark-protocol.md)记录 TTFT/TPOT p50/p95/p99、输出/总吞吐、stream gap、显存和错误；额外记录 K 分布、各位置接受率、实际执行 token、Graph 模式、控制成本和成本模型误差。避免新增逐 token 的 host 同步。

## 正确性、回滚与结果

覆盖拒绝首/中/末草稿、零草稿步、EOS、请求重排、preemption、slot reuse，以及上一轮接受较多而本轮只执行一个 token。QSA 压缩组、GDN/卷积、PLE 历史必须只提交接受链；检查被裁掉的尾部是否通过 MoE capacity/token dropping 影响保留 token。确定性病例比较 token、teacher-forced logits 与状态；随机采样按原协议检验分布和任务得分，不能以数学等价替代浮点正确性证据。

出错、质量或尾延迟退化、收益低于噪声时关闭控制器，恢复已验证静态路径。后续 [T017](T017-replayssm-fixed-precision.md)、GDN kernel 和 router 成本校正分别单独 A/B。[HybridSpec][hybridspec]仅保留为 router 校正灵感：一手摘要可核对，全文、公开实现、完整硬件/精度在本轮为 `unknown`，不引用其速度预测本机。

`baseline_run_id / backend_port_commit / quality_pass / incremental_speedup = unknown`。当前结论是**有明确模型后端阻塞，暂不交付可运行开关**。

[engine]: https://github.com/vllm-project/vllm/tree/12924016eb8dbaae87d5163a1b7ec2614a44a797
[model-config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json
[adaptive-doc]: https://docs.vllm.ai/en/v0.34.0/features/speculative_decoding/adaptive_verification/
[adaptive-code]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py
[ple]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/v1/attention/backends/short_conv_attn.py
[backend]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/v1/attention/backend.py
[qsa]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/common/qsa.py
[indexer]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/common/qsa_indexer.py
[gdn]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/v1/attention/backends/gdn_attn.py
[hybridspec]: https://arxiv.org/abs/2611.04055

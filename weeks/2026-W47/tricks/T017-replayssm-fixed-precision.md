# T017 · 针对 36 层 GDN 的固定精度 ReplaySSM 移植

- 状态：`conditional`；模型特定的高工程主线，先以 profile 决定是否投入实现。
- 调研日期：2026-11-20；证据：`literature` + 源码静态核查 + `hypothesis`；没有 H800 实测。
- 目标：保持现有所有 dtype，减少 GDN recurrent state 写回和投机快照成本；不是现成开关。
- Week 1 全试为 `user_report`，最佳有效配置及各项结果为 `unknown`；不要求重做首轮。

## 为什么是这个模型的候选

[官方配置][config]明确 `Qwen4ExpForConditionalGeneration` 有 36 层 GDN，16 个 key heads、48 个 value heads，key/value head dim 均为 128，`mamba_ssm_dtype=float32`。本卡固定审计 vLLM SHA `12924016eb8dbaae87d5163a1b7ec2614a44a797`，不把 Qwen3.5 的结果当目标模型实测。

[模型实现][model]的 `get_gdn_mamba_state_shape_from_config` 调用 [shape calculator][shapes]，每层每请求每 TP rank 的 recurrent shape 是 `[48/TP,128,128]`。按官方 FP32 契约计算：

```text
M_all = 36 × 48 × 128 × 128 × 4 = 108 MiB / request
M_rank = n_local_GDN × (48/TP) × 128 × 128 × 4 bytes
```

PP=1 时为 `108/TP MiB`，例如 TP4 为 27 MiB/rank；这只是合法分片下的容量推导，不是布局推荐。有 PP 时必须使用本 rank 实际 GDN 层数。108 MiB 是全 TP 聚合 recurrent payload，不含卷积、PLE、QSA KV/indexer/scratch、allocator padding 或投机副本；不能当作总显存占用。本机实际 dtype 仍须从运行时 `get_gdn_mamba_state_dtype_from_config` 与已分配 tensor 核对。

传统每步全读全写的理想 recurrent 流量约为 216 MiB/request（全 TP 聚合）。若 profile 显示其处于关键路径，减少写回有具体价值；若主要耗时在 MoE 权重或通信，应降优先级。

## 机制与已有实验的区别

[ReplaySSM 作者说明][blog]保留 recurrent checkpoint 与短输入历史，延迟完整状态写回；需要时重建，投机只提交接受前缀。它改变同一请求内部状态更新，不是 [T008](T008-hybrid-prefix-cache.md) 的跨请求缓存，不是 [T009](T009-native-mtp.md) 深度扫描，也不改变 [T015](T015-adaptive-verification.md) 的长度选择策略。

当前 [QwenGatedDeltaNetAttention][gdn-code] 的实际入口为：

| 路径 | 已有调用与移植落点 |
|---|---|
| 普通 decode | `_forward_core_decode_non_spec`：卷积更新后调用 `fused_recurrent_gated_delta_rule_packed_decode`，读写 `kv_cache[1]` |
| 普通投机路径 | `_forward_core` 调用 `fused_sigmoid_gating_delta_rule_update`，`inplace_final_state=True`，使用 spec query boundaries、state indices、accepted tokens |
| 融合投机路径 | `_forward_core_decode_spec_post_conv_fused_norm` 调用 `fused_mtp_decode_impl()`，同样使用 `kv_cache[1]` 与接受长度 |

须从运行日志/profile 确认实际 dispatch，再改相应分支；改了未执行的 kernel 没有收益。该类在审计 SHA 尚无 ReplaySSM 接线，不能直接套作者参数。

## 上游证据与明确不可照搬的部分

- [作者代码][author]固定 `73a9592e9d8a2268620e8883a82f5cec7442ed36`，入口含 `fused_recurrent_replayssm.py`、`gdn_replayssm_spec_decode.py`、`replayssm_config.py` 与 teacher-forced equivalence 测试。
- [#62004][pr-triton] 检索时 open，head `0880cb87383330070f6a2546536ee99496306bab`：Triton GDN 普通 decode，包含 H100/Qwen3.5 验证。可参考 SM90 机制，不等于支持本模型。
- [#65657][pr-qwen4] 检索时 open，head `682a577e7e50bfcc9786544152244ece7f2c7be2`：**已出现 Qwen3.8-Flash-Next 接入**，但依赖 SM100、FlashInfer 0.7.3、FP16 recurrent state。它不是 H800/FP32 可用实现，不能以改精度满足条件。
- [#55688][pr-shared] 合并 SHA `750cad1973f037fe00116cbf58cd3b3bd8701cd4` 提供共享 Mamba 生命周期；它不自动补齐目标 GDN kernel。

这是 SM90、Qwen4Exp、固定精度三项同时满足的移植研究。[论文][paper]的 FP32-state 速度包含 FP16 cached corrections，不能直接挪用。manifest 必须登记 baseline/candidate 的 checkpoint、卷积、KV/indexer、PLE、replay k/d/g/u、correction 的**形成/存储/读取/累加 dtype**。原 FP32 correction 不得新增 FP16 缓存；若需 FP32 ring 或额外重算，就测这个版本，不引用原速度。

## 状态生命周期必须一起移植

1. 在模型 `get_mamba_specs_from_config`、GDN `get_state_shape/get_state_dtype` 和 [MambaBase.bind_kv_cache][lifecycle] 中加入 ring/cursor/flush 元数据及实际 page bytes，保持类型与对齐一致。
2. PLE 的短卷积是单独 `MambaSpec`，`tp_replicated=True`；不能按 GDN 分片，也不能给所有 Mamba 层套同一 replay layout。原 GDN 卷积、PLE 历史和 QSA scratch 继续遵循各自提交规则。
3. 衔接 accepted-prefix、拒绝回退、prefix materialization、cache copy、preemption、slot reuse 与 Graph buffers。`get_mamba_num_speculative_blocks()` 在全局启用 replay 后会返回 0；未适配消费者前不能先减掉旧快照池。
4. 模型能力声明和 `validate_mamba_cached_kernel` 也要同步核对；给开关加白名单不能代替前述状态实现。

## 最小 A/B、收益与撤回

先用该模型真实 per-rank shape 验证单层普通 decode，再独立验证固定深度 MTP；初始只选一个短窗口，如 L=8。主 A 为同 build 的 Week 1 当前最佳有效配置，B 只替换状态更新实现，MTP 深度、长度控制、router、精度、调度、GPU 总预算不变。升级引擎需另记录升级差异；普通 decode 微基准只作诊断，不替代已有最佳 MTP 对照。

至少三次交错重复，测主要 batch 与低负载回归组，按[协议](../../../docs/benchmark-protocol.md)报 TTFT/TPOT、吞吐、stream gap、显存和错误。另报 DRAM bytes、非 flush/flush 时间、ring/padding 占用、抢占及 p95/p99，避免周期性 flush 拉高尾延迟。

以下为理想带宽推导，M 为原 recurrent payload，R 为新增 replay 访存，f 为最佳配置中 GDN 的关键路径占比：

```text
原流量 ≈ 2M；replay 流量 ≈ M + M/L + R
s_GDN ≤ 2 / (1 + 1/L + R/M)
S_total ≈ 1 / (1 - f + f/s_GDN)
S_incremental = throughput_candidate / throughput_week1_best
```

新增计算、FP32 ring 和重叠会使实际收益低于理想界，已有 MTP 收益不能重复计入。检查 teacher-forced logits、提交后 recurrent state、拒绝首/中/末 token、零草稿、ring wrap、flush、prefix 命中、混合长度和请求复用；按原质量合同验收，数学等价不保证逐 bit 一致。

若质量/状态错误、尾延迟恶化或收益不足，撤回 replay dispatch 并重建相关缓存；禁止让旧路径读取新布局。`baseline_run_id / implementation_commit / dtype_manifest / quality_pass / incremental_speedup = unknown`。

[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json
[model]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/model.py
[shapes]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/mamba/mamba_utils.py
[gdn-code]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
[lifecycle]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/mamba/abstract.py
[blog]: https://dao-lab.ai/blog/2026/replayssm/
[paper]: https://arxiv.org/html/2610.13778v1
[author]: https://github.com/Johnny-Liou/ReplaySSM/tree/73a9592e9d8a2268620e8883a82f5cec7442ed36
[pr-triton]: https://github.com/vllm-project/vllm/pull/62004
[pr-qwen4]: https://github.com/vllm-project/vllm/pull/65657
[pr-shared]: https://github.com/vllm-project/vllm/pull/55688

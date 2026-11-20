# T027 · Qwen4Exp 完整 logits 的按请求分片采样

- 状态：`conditional`；接口改动小，性能与 NCCL 常驻内存门槛必须先核验。
- 日期：2026-11-20；证据：`literature + hypothesis`；本机实测：无。
- 实施入口：先读 [参考 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行一个阶段，过检查点再继续。
- 基线：W1 当前最佳有效配置，保留现有采样、完整输出、reasoning、固定深度 MTP 与精度。
- 新增点：**每个请求的完整词表 logits 交给一个 TP owner 处理**；与 T014 仅返回赢家/候选、避免 logits 的融合接口不同。本卡保留 logits，不重做 T007 拓扑或 T018 GEMM+collective 融合。

## 1. 本模型当前缺的不是通用算法

固定审计 [vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`][sha]：

| 源码与符号 | 已核调用链 |
|---|---|
| [qwen4_exp/nvidia/model.py::Qwen4ExpForCausalLM][model] | compute_logits 调用原 LogitsProcessor；**仍没有 compute_logits_local** |
| [qwen3_5.py::Qwen3_5ForConditionalGeneration.compute_logits_local][wrapper] | Qwen4Exp 外层继承此包装，调用内层同名方法 |
| [sample/batch_shard.py::batch_sharded_sampling_enabled][shard] | 只检查外层 hasattr，因此此模型能过检查后在 warmup/调用时失败 |
| 同文件 BatchSharder / all_to_all_vocab_shards / gather_sampler_output | owner 排序、局部 metadata、词表 shard all-to-all、采样结果还原已有实现 |
| [gpu/model_runner.py::sample][runner] | owner 排序 hidden→局部 head→all-to-all→裁 padding→grammar→sampler/rejection_sampler→收集输出 |
| [config/vllm.py::_validate_batch_sharded_sampling][validator] | 真实配置限制必须保留，不能只见开关即声称支持 |

[PR #61018][pr]是此模型的直接修复先例，仍为 draft/open；固定 head [`3de81fb47d9efd0cb89b123dcaf1c5ad42cfb520`][patch]，不是已合并功能。最小适配在内层增加：

```python
def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)
```

继续使用同一 BF16 默认 head [248320,2560]、同一 dtype override/math、原 scale/soft-cap 与 kernel；本地 TP4/TP8 权重分别为 [62080,2560]/[31040,2560]，实际 padding 以运行时为准。**各 rank 仍计算全部所需行的本地词表 GEMM**，省的是重复全词表接收/采样，不是 head FLOPs。不得修改共享 target/MTP head 的权重或 quant_method。

## 2. 前提与上游负证据

当前 validator 要求 TP>1、max_num_seqs≥TP、max_logprobs≥0；return_sampling_mask 不支持；LoRA 若覆盖 lm_head 则阻塞。无需 lm_head 的 LoRA 不能一概说不支持，仍核实际 wrapper。保留所需功能，遇阻塞记 blocked，不删功能换速度。确认执行的是上述 GPU runner；单改模型方法不保证其他 runner 已接入。T015 的动态验证限制不因本卡自动消失。

PR 报告同款模型、TP8+EP、MTP3，在 **8×A100-40GB** 的旧 nightly/固定 KV 条件下报告约 2% 输出吞吐改善；其开启分片的实验同时列出 `NCCL_MAX_P2P_NCHANNELS=4`，未证明 control 使用同 cap，因此只能记为组合配置结果，不能单独归因于接口/分片。更新主线低并发测试未检出收益，且不是完整成对实验。其真实 TP2 测试证明 tiny head 重组和采样等价，**不覆盖整个目标模型所有采样功能**。

[#68067][memory]记录 A100 TP8 默认 P2P 缓冲约 2.7 GiB/GPU、KV 容量减少；更小 channel cap 也会损失通信速度。当前 all-to-all 已优先复用 TP PyNCCL communicator，不能再以“去掉第二套 communicator”宣称新增收益。H800、实际节点与并发均未测；不照抄 A100 的通道值。

## 3. 实施 Agent：七步

1. **C0 审计**：记录引擎 SHA、内外两层方法、当前 runner、TP 组、head dtype/stride/padding、all-gather 或 gather 分支、原 NCCL 环境和 KV/state 容量。先取实际 logits 行数 M 分布；M 含 MTP 验证/bonus 行，不等于请求数。
2. **C1 只补方法**：对照上述固定 PR 的 NVIDIA 改动最小移植；开关关闭时原路径不变。若本地已实现，记录接口 baseline-covered，继续判断分片路径是否原本启用；不重复打补丁。
3. 用 PR 的 [test_tpsp.py::test_batch_sharded_logits_match_all_gather][test]做真实 TP2 tiny 检查：有 padding、跨 shard tie、greedy 和显式 seed；先证明未补接口会失败，再证明方法使局部 shard/重组行正确。只用既有 GPU 预算，测试准备由实施 Agent 在现有授权与 GPU 预算内完成。
4. 复用 [test_gpu_batch_shard.py][testshard]的 metadata、空 owner、grammar、logprobs、slot 释放及 draft cache 测试。再对目标模型做小样本：原 top-k/top-p、penalties、grammar、logprobs，以及原 MTP 接受/拒绝/bonus/请求重排和取消。保留 owner→请求→位置→RNG 映射，不能仅比较最终文本长度。
5. **C2 单变量 A/B**：相同 adapter build、相同 NCCL 设置，A0 关闭分片，A1 只启用已核的 `enable_batch_sharded_sampling`。保持调度/Graph/所有精度/MTP 深度与采样不变。测试一档真实低 M、一档真实较高 M及混合请求，不增加总 GPU 数。
6. 独立预热后比较 free-device memory、PyTorch allocated/reserved、NCCL 增量、KV/state 容量与峰值；必须跑到真实高水位。若需 channel cap，另立 C3、固定分片开关只改 cap；再用同 cap 重做分片对照。不要在 A1 偷加 cap，或只报固定 KV 的理想微基准而隐藏服务容量损失。
7. 每点至少三次交错重复，按[协议][bench]报告完整输出/总吞吐、TTFT/TPOT 分位数、错误率、owner 行数不均衡、all-to-all+packing+sampler 时间及内存。交付 adapter/test、配置 diff、tiny/服务证据、资源统计和关闭开关的回滚方式。

## 4. 成本公式与退出

以下为推导 hypothesis，仅适用于原链全员 all-gather、P 个 TP rank、M 行均衡分给 owner、每元素 b 字节、padded vocab V 的情形：

`recv_A ≈ b*M*V*(P−1)/P`，`recv_B ≈ recv_A/P`。

实际 owner 行数不均、不同验证长度、打包、P2P 启动和输出 gather 都要单独计时；原链若只 gather 到单 rank，不能套此式。每 rank head GEMM 与本地 logits 仍存在，完整 logits 的 live memory 与 M_owner 成正比；常驻 NCCL buffer 可能远大于当步节省。

tiny 重组 logits 需一致，含 tie 的 token 结果需一致；整模型改 batch 排序可能触发原 GEMM 数值差异，应按既定数值/质量规则解释，不能把错误 owner 当随机性。保留完整 target 验证数据和 draft cache，不能用 T014 argmax 数据替代。任何功能丢失、Graph/MTP 失配、容量下降导致吞吐/尾延迟退化，关闭分片；若收益落在重复噪声内则 inconclusive。本机环境、实现 SHA、容量变化、收益及质量均 **unknown**。

[sha]: https://github.com/vllm-project/vllm/tree/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f
[model]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/nvidia/model.py
[wrapper]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/models/qwen3_5.py
[shard]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/v1/worker/gpu/sample/batch_shard.py
[runner]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/v1/worker/gpu/model_runner.py
[validator]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/config/vllm.py
[pr]: https://github.com/vllm-project/vllm/pull/61018
[patch]: https://github.com/vllm-project/vllm/blob/3de81fb47d9efd0cb89b123dcaf1c5ad42cfb520/vllm/models/qwen4_exp/nvidia/model.py
[test]: https://github.com/vllm-project/vllm/blob/3de81fb47d9efd0cb89b123dcaf1c5ad42cfb520/tests/models/qwen4_exp/test_tpsp.py
[testshard]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/tests/v1/worker/test_gpu_batch_shard.py
[memory]: https://github.com/vllm-project/vllm/issues/68067
[bench]: ../../../docs/benchmark-protocol.md


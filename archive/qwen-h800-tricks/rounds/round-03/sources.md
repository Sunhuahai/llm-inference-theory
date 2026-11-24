# Round 03 · 固定版本与一手来源

检索与源码核对日期：**2026-11-20（Asia/Shanghai）**。技术判断仅使用作者论文、官方仓库/文档和项目原始 issue/PR。源码可证明“已实现/仍缺入口”，不能代替用户 H800 上的性能、质量或支持验证。

## 1. 固定基点

| 对象 | 版本 / 读取范围 |
|---|---|
| 研究仓库更新基点 | [915023a373d3b7d2a6726df02e53804b74dfc3e8](https://github.com/Sunhuahai/qwen38-next-h800-tricks/tree/915023a373d3b7d2a6726df02e53804b74dfc3e8) |
| vLLM 主线 | [cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f](https://github.com/vllm-project/vllm/tree/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f)；本轮统一源码基点，不要求用户升级 |
| FlashTopK | [090902abad86e5bb3324519124a680122fff91a1](https://github.com/Tencent-Hunyuan/FlashTopK/tree/090902abad86e5bb3324519124a680122fff91a1) |
| TreeWY 作者实现 | [b073ed6cacfa1cf5ae23111854729e662a72a6d1](https://github.com/sneha5gsm/vllm/tree/b073ed6cacfa1cf5ae23111854729e662a72a6d1)；不假定等同论文全部实验版本 |
| TreeWY 论文 | [arXiv:2608.20961v2](https://arxiv.org/abs/2608.20961v2)，2026-10-22 修订 |
| NVIDIA hyperflare | [6555f9397bd7f39b02b836b7f51ce050a54f26a6](https://github.com/NVIDIA/hyperflare-inference/tree/6555f9397bd7f39b02b836b7f51ce050a54f26a6) |
| NVIDIA CUTLASS | [af951046cd9cac9301d8e3e90863fd8d7edc9fa9](https://github.com/NVIDIA/cutlass/tree/af951046cd9cac9301d8e3e90863fd8d7edc9fa9) |
| Qwen4Exp logits 接口 PR #61018 | draft/open；head [3de81fb47d9efd0cb89b123dcaf1c5ad42cfb520](https://github.com/vllm-project/vllm/tree/3de81fb47d9efd0cb89b123dcaf1c5ad42cfb520) |
| Expert cache PR #37190 | open/unmerged；head [c66868c8c12f10fa931e108c3a8128ccddfdcdbb](https://github.com/vllm-project/vllm/tree/c66868c8c12f10fa931e108c3a8128ccddfdcdbb) |
| 模型字段 | [Qwen 官方 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json)，滚动 main 检索；不是用户本机 revision |

所有 PR 状态按检索日期记录。实施时核用户实际 build 的路径和数值合同，不因 PR 已有代码就假定已合并或本机已启用。

## 2. 十卡的最短复核路径

| 卡片 | 当前实现 / 一手机制 | 支持边界 |
|---|---|---|
| [T021](tricks/T021-qsa-streaming-exact-topk.md) | [QSA score/top-k](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py)；[FlashTopK API](https://github.com/Tencent-Hunyuan/FlashTopK/blob/090902abad86e5bb3324519124a680122fff91a1/docs/api.md)；[现成 DSA fused wrapper](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/kernels/attention/dsa/fused_mqa_topk.py) | SM100/32-head 或连续 KV 接口均非本模型 H800 四头分页即用实现 |
| [T022](tricks/T022-qsa-query-tile-reuse.md) | [QSA attention](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/nvidia/ops/qsa.py)；[NVIDIA 11-17 文章](https://developer.nvidia.com/blog/co-designing-sparse-attention-for-efficient-long-context-inference/)；[cuDNN QSA](https://docs.nvidia.com/cudnn/operations/sparse_attention/qsa) | union 方案为本仓库推导；不改 block/精度，不把 SM100 训练接口当 H800 inference 后端 |
| [T023](tricks/T023-treewy-branch-verification.md) | [TreeWY v2](https://arxiv.org/abs/2608.20961v2)；[作者 RFC #54080](https://github.com/vllm-project/vllm/issues/54080)；[tree_wy_triton](https://github.com/sneha5gsm/vllm/blob/b073ed6cacfa1cf5ae23111854729e662a72a6d1/vllm/third_party/flash_linear_attention/ops/tree_wy_triton.py)；[采样参考](https://github.com/sneha5gsm/vllm/blob/b073ed6cacfa1cf5ae23111854729e662a72a6d1/vllm/v1/spec_decode/tree_spec.py) | Qwen3.5 外部实验；默认 BF16 dot 与本项目合同不符；宽树有吞吐负结果 |
| [T024](tricks/T024-mtp-history-drafts.md) | [Qwen3.8 MTP 设计](https://github.com/NVIDIA/hyperflare-inference/blob/6555f9397bd7f39b02b836b7f51ce050a54f26a6/docs/mtp-design.md)；[逐步 q 接线](https://github.com/NVIDIA/hyperflare-inference/blob/6555f9397bd7f39b02b836b7f51ce050a54f26a6/ignition/recipes/qwen38/speculative_model.py)；[vLLM proposer](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py) | 参考 runtime 为 SM120/121、示例 FP4-Local；只能迁移机制，首版仍执行 MTP forward |
| [T025](tricks/T025-router-top10-fusion.md) | [GateLinear](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/router/gate_linear.py)；[MoERunner](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/runner/moe_runner.py)；[topk CUDA](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/csrc/moe/topk_softmax_kernels.cu) | 当前 top-k 已融合；跨 gate 边融合是新假设，不能改变 BF16 logits、完整路由和 EPLB |
| [T026](tricks/T026-shared-expert-mlp-island.md) | [shared MLP](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/models/qwen3_moe.py)；[activation CUDA](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/csrc/activation_kernels.cu)；[shared overlap](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/runner/shared_experts.py) | 保留 SiLU 中间 BF16 舍入；既有 overlap 与小 GEMM 分派不是新增收益 |
| [T027](tricks/T027-full-logits-batch-sharding.md) | [PR #61018](https://github.com/vllm-project/vllm/pull/61018)；[固定模型补丁](https://github.com/vllm-project/vllm/blob/3de81fb47d9efd0cb89b123dcaf1c5ad42cfb520/vllm/models/qwen4_exp/nvidia/model.py)；[batch_shard](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/v1/worker/gpu/sample/batch_shard.py)；[#68067](https://github.com/vllm-project/vllm/issues/68067) | PR 未合并；外部 A100 旧结果混有 cap 变化，不能单归因；常驻通信内存可能抵消收益 |
| [T028](tricks/T028-persistent-expert-workqueue.md) | [Triton MoE](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/fused_moe.py)；[SM90 grouped scheduler](https://github.com/NVIDIA/cutlass/blob/af951046cd9cac9301d8e3e90863fd8d7edc9fa9/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp) | 借调度机制，保留原 block128 算术；普通 grid 本来就被硬件调度，不保证 persistent 更快 |
| [T029](tricks/T029-heterogeneous-state-pages.md) | [platform 对齐](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/platforms/interface.py)；[Mamba spec](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/mamba/abstract.py)；[packed planner](https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/v1/core/kv_cache_planning.py) | 原 QSA packing 已有；新实验只去剩余 GDN/PLE 尾 padding，必须验证整池总账 |
| [T030](tricks/T030-exact-expert-weight-cache.md) | [RFC #38256](https://github.com/vllm-project/vllm/issues/38256)；[PR #37190](https://github.com/vllm-project/vllm/pull/37190)；[固定 provider](https://github.com/vllm-project/vllm/blob/c66868c8c12f10fa931e108c3a8128ccddfdcdbb/vllm/model_executor/layers/fused_moe/expert_weight_provider.py)；[并行校验](https://github.com/vllm-project/vllm/blob/c66868c8c12f10fa931e108c3a8128ccddfdcdbb/vllm/model_executor/layers/fused_moe/routed_experts.py) | 原权重/scale 同驻留；拒绝会改变舍入的 expert split，EP/DP/SP 支持及 Graph 合同有阻塞 |

## 3. 文档与代码不一致时采用什么

1. **Expert cache PR 正文**仍描述 per-tensor/eager 限制；固定 head 的 [fp8.py](https://github.com/vllm-project/vllm/blob/c66868c8c12f10fa931e108c3a8128ccddfdcdbb/vllm/model_executor/layers/quantization/fp8.py) 已接 block-scale buffer，[config/vllm.py](https://github.com/vllm-project/vllm/blob/c66868c8c12f10fa931e108c3a8128ccddfdcdbb/vllm/config/vllm.py) 允许 piecewise，但会把 FULL 降级。采用实际代码限制；不据标题推断支持。
2. **TreeWY 文件顶部的精度说明**不能覆盖 `dot_bf16=True` 的实际分支。逐项读 operands、accumulator、stash 与 commit，不只查 state storage dtype。
3. **CUTLASS Hopper 示例**顶部说明与实际 FP8 dtype 校验不同；[固定示例](https://github.com/NVIDIA/cutlass/blob/af951046cd9cac9301d8e3e90863fd8d7edc9fa9/examples/python/CuTeDSL/cute/hopper/kernel/grouped_gemm/grouped_gemm.py) 可证明存在 persistent 机制，不能证明本模型 block128 scale 合同已满足。
4. **本模型支持检查**不能止于外层 `hasattr`。T027 必须沿 wrapper 调到实际 language_model；通用测试和同模型 tiny 测试分别陈述覆盖范围。

T025/T026 使用 SM90 cluster/DSM 的实现机制参考 [CUDA 13.1.1 官方归档](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。这不要求本机升级工具链，也不证明具体融合占用或性能；参与 CTA 的生命周期和容量由实施时核验。

## 4. 如何引用与更新

每张卡已放完整实现入口、必要函数和来源链接；研究设计与收益公式标 hypothesis，别的硬件/模型数字只作带条件的外部证据。论文 v2 与代码分支 SHA 分别锁定，不能自动视为同一实验版本。滚动厂商文档标检索日期；实际部署以运行 manifest 为准。

没有本轮 H800 benchmark。后来收到用户或实施 Agent 的结果时追加真实 run_id、原始附件和代码 commit；没有证据时保留 unknown，不覆盖历史负结论。

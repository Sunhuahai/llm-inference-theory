# T025 · Qwen4Exp 路由 GEMM → 精确 top-10 跨边融合

- 状态：`conditional`；难度高，先过路由数值与关键路径门槛。
- 日期：2026-11-20；证据：`literature + hypothesis`；本机实测：无。
- 实施入口：先读 [参考 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行一个阶段，过检查点再继续。
- 基线：W1 当前最佳有效配置；仅 T001–T013 已由用户确认尝试，逐项结果 unknown。
- 新增点：消除 **2560→512 路由投影与已融合 softmax/top-10 之间**的输出物化/启动边界。不是 T004 的 QSA GEMM 调参、T010 的通信或 R03 的专家分段。

## 1. 固定源码与真实计算合同

本卡审计 [vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`][sha]。下列路径均相对此 SHA；本机不同版本须核对同一调用链，不为实验自动升级。

| 文件、符号 | 已核事实 |
|---|---|
| [models/qwen4_exp/nvidia/model.py::Qwen4ExpSparseMoeBlock][model] | 继承 Qwen3Next 的门控/路由实现 |
| [model_executor/models/qwen3_next.py::Qwen3NextSparseMoeBlock][next] | gate 为 `GateLinear(2560,512)`，top-k=10；没有传入 `out_dtype` 或 `force_fp32_compute`；norm_topk_prob 缺省 True |
| [layers/fused_moe/router/gate_linear.py::GateLinear.forward][gate] | BF16→FP32 的 ll_bf16 快路需要 out_dtype=FP32；不能因为该 helper 存在就称本模型已走 FP32 gate |
| [layers/fused_moe/runner/moe_runner.py::_forward_impl/_maybe_dispatch/_apply_quant_method][runner] | gate 在 dispatch 前，路由在 dispatch 后；modular 与 monolithic 分支不同 |
| [router/fused_topk_router.py::fused_topk][topk]、[csrc/moe/topk_softmax_kernels.cu::topkGating][cuda] | E=512 已有 fused softmax/top-k，并含 PDL；输出权重 FP32，索引类型随后端 |
| [router/base_router.py::BaseRouter._select_experts][base] | 保留 logical-ID capture、EPLB logical→physical 映射和索引类型转换 |

[官方 config][config]把 gate 排除于 FP8 转换，默认模型 dtype=BF16。**默认链是 BF16 权重/hidden、原 GEMM 的累加合同、BF16 logits，再在 top-k 内转 FP32**；本机 override、编译器和实际 kernel 仍须记录。不得把 BF16 logits 改成 FP32，不能用“更准确”解释路由改变。

首版完整计算 512 个分数，复用原 full-row softmax、expf、归约次序和 top-10 后 renorm。即使数学上等价，也不改成“先选 raw top-10，再对十个数 softmax”。保持相同 tie（较小 expert ID 优先）、NaN/Inf→0、padding ID=-1、routed scale、输出索引 dtype。实际若有 grouped/bias/custom routing，首版回退，不删其语义。

## 2. 前置分支：不跨过未适配的通信

仅在当前最佳配置本来满足下列条件时进入：H800；实际独立 gate + `FusedTopKRouter`；E=512/k=10；原 BF16 路由链；modular MoE；`do_naive_dispatch_combine=False`；PCP 不在 gate/router 间 gather；无 FSE 合并 gate；能访问原权重且无 gate LoRA。其他分支先记 blocked/fallback，**不关闭 SP、EP、EPLB、MTP 或改拓扑来制造适用性**。EPLB 若开启，仍必须经过 BaseRouter 后处理。

少量 trace 只记 M（含 Graph padding）、kernel 名、dtype/stride、gate+route 的关键路径时间及后端，不导出原始请求。若这段已合并、主要被其他流遮蔽，或占比低于测量分辨率，停止。

## 3. 实施 Agent：按顺序最多七步

1. **C0 审计**：在实施仓库执行 `git rev-parse HEAD`，用 `rg -n 'GateLinear|_maybe_dispatch|_compute_routing' vllm/` 定位上述链；保存实际 gate weight=[512,2560]、logits dtype、累加/math 设置、router 类和分支证据。
2. **C1 只接接口**：新增仅 Qwen4Exp 可选的 router 子类与默认关闭的本地开关。参考分支仍调用原 gate→原 fused_topk；通过 BaseRouter 的同一 capture/EPLB 后处理返回。runner 只有命中完整 predicate 才跳过重复 gate；未命中原样回退。不要用全局 stash 传递结果。
3. 为该接口增加小型测试：M=0/1/4/16、E=512/k=10，原输入、全等行、第 10/11 名接近 BF16 舍入边界、NaN/Inf、padding。复用 [已有 top-k 测试][test]的零 bias/tie 参照；要求 IDs、顺序、padding、capture/EPLB 映射一致，权重满足预先固定数值合同。
4. **C2 只替换内部 op**：为实际高频小 M 实现 SM90 tile/cluster kernel。按 expert 维分块计算，先按原 dtype 舍入，再把整行 512 值交给原路由归约；可用共享内存/DSM，禁止重复计算每块全 gate、启发式裁专家或换 fast-exp。使用 DSM 时按 [CUDA 13.1.1 说明][dsm]核对本机工具链、cluster 容量与全部 CTA 的起止同步；不直接复制其他 GPU 的 cluster 大小。K 累加/舍入无法满足合同即停止，不放宽容差。
5. 为实现增加 eager/Graph replay、padding→非 padding、不同 batch 次序与权重 reload 检查；静态 buffer/指针在 capture 前分配，跨 replay 无残留。抽取实际中间 tensor 在原位做差分，只保存脱敏误差统计。
6. **A0=原链；A1=参考 adapter；A2=融合 op**。先证 A0/A1 无行为改变；性能只比较 A1/A2，唯一变量为跨边融合。选择真实小 M 和一个原大 M 回退点，保持所有权重、dtype/math、Graph、并行、采样及 MTP 不变。
7. 按[协议][bench]做至少三次交错重复；交付默认关闭的 patch、predicate、测试、原始统计、fallback 比例和 rollback commit。用完整输出服务回归，不把 route 微基准当最终收益。

## 4. 收益判断、验收和退出

理论上只省 logits 写/读与启动边界，**不省 512×2560 GEMM 工作量**。本仓库推导 hypothesis：若原路由段占步时比例 f，局部变成原来的 r，新增同步/占用损失为总步时的 ε，则 `S=1/(1-f+f*r+ε)`。必须实测 DSM/cluster 同步、寄存器和对专家并发的影响；少一个 kernel 不保证更快。

报告 TPOT/TTFT p50/p95/p99、输出/总吞吐、失败率、显存、gate+route 时间和 routed-ID 差异。任何真实行路由 ID 改变、舍入/math 偷换、EPLB/capture 丢失、Graph 错误、总收益落入噪声或尾延迟退化，立即恢复原 router/gate。未满足分支记 blocked，已有实现记 baseline-covered；本机 baseline/candidate ID、实现 SHA、收益及质量均 **unknown**。

[sha]: https://github.com/vllm-project/vllm/tree/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f
[model]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/nvidia/model.py
[next]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/models/qwen3_next.py
[gate]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/router/gate_linear.py
[runner]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/runner/moe_runner.py
[topk]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py
[cuda]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/csrc/moe/topk_softmax_kernels.cu
[base]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/router/base_router.py
[test]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/tests/kernels/moe/test_fused_topk.py
[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json
[bench]: ../../../docs/benchmark-protocol.md
[dsm]: https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/02-basics/writing-cuda-kernels.html


# T026 · Qwen4Exp BF16 shared-expert MLP 小 M 执行岛

- 状态：`conditional`；激进 kernel 实验，收益依赖 shared 分支是否在关键路径。
- 日期：2026-11-20；证据：`literature + hypothesis`；本机实测：无。
- 实施入口：先读 [参考 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行一个阶段，过检查点再继续。
- 基线：W1 当前最佳有效配置，保留既有 shared-stream/Graph/MTP 与全部精度、math。
- 新增点：跨 **shared FC1→SwiGLU→FC2** 的局部执行岛；不是 T010 已有 shared/routed 重叠、T004 单 GEMM 调参、T016 HC 或 R03 routed-expert chunking。

## 1. 为什么只研究这个 shared expert

审计 [vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`][sha]。Qwen4ExpSparseMoeBlock 继承 [Qwen3NextSparseMoeBlock][next]，独立 shared_expert 实际是 [Qwen3MoeMLP.forward][mlp]：

`gate_up_proj → maybe_fused_act_quant → down_proj → sigmoid(expert_gate(x)) × out`。

[官方 config][config]里 hidden=2560、shared intermediate=640、激活 SiLU，shared 的 gate/up/down 及单标量 gate 均排除 FP8。因此本题为原 BF16 路径；不是把 routed FP8 专家改成 BF16，也没有可照搬的 block128 shared scales。

| 既有合法布局（不在本卡切换） | 本地 FC1 权重 | 本地 FC2 权重 |
|---|---|---|
| 普通 TP4 | [320,2560] | [2560,160] |
| 普通 TP8 | [160,2560] | [2560,80] |
| MoE SP/disable_tp | [1280,2560] | [2560,640] |

这里 I 是实际本地 intermediate；必须从权重核对。shared FP8/Quark、LoRA、不同 I 或特殊 scale 路径未适配就回退。目标的 `reduce_results=False`：融合只返回本地 down 输出，保留原 scalar gate、shared+routed 相加及最终 collective 次序，不能把复制的 shared 输出再求和。

## 2. 已内建的路径与数值陷阱

[shared_experts.py::SharedExperts][stream]已含 aux-stream 和 MK 内部 overlap；[utils.py::resolve_layer_fused_shared_expert][fse]是 AITER 门槛，不能当作 NVIDIA 已融合。实际若已有完整 MLP fusion 则 baseline-covered，不拆掉优化。当前 [SM90 GEMM 表][gemm]已覆盖 TP4 FC1 的 M=1/2/4/8/16（来源为 H200 筛选），A0 必须保留其真实分派。

[maybe_fused_act_quant][actquant]在没有 input quant key 时调用原激活；不能为本卡创建 FP8 activation。尤其 [activation_kernels.cu][actcuda]的 CUDA `silu_kernel/packed_silu_kernel`先把 SiLU 结果转回输入 dtype，随后与 up 相乘，再转回输出 dtype。原 CUDA BF16 合同示意如下，`rB` 为 BF16 舍入：

`g,u=rB(FC1(x)); s=rB(g/(1+expf(-g))); z=rB(float(s)*float(u)); y=rB(FC2(z))`。

不能把 s 保持 FP32 到乘法，也不能把 FC1 累加器直接喂激活。实际若走 compile-native，先记录并复现该分派的舍入/运算合同。FC1/FC2 的累加精度、指令 math 模式和允许的归约误差必须沿用基线；禁用近似激活替换。首版 scalar sigmoid gate 留在原位置，不同时融合。

## 3. 实施 Agent：七步与两个独立阶段

1. **C0 定位**：在实际引擎记录 SHA、目标层前缀、M/stride、两组权重形状和 dtype、`get_input_quant_key(down_proj)`、SiluAndMul 实际分派、shared stream/order。用少量 trace 量 shared 完成到主流等待的时间，不以 kernel 累计时间当关键路径。
2. **C1 接入参考 op**：只给 Qwen4Exp 的独立 BF16 shared MLP 加默认关闭 wrapper；内部仍顺序调用原 FC1/activation/FC2，gate 与返回值不变。不得全局替换 Qwen3MoeMLP。不满足 predicate 直接原路径；先证明 wrapper 输出相同。
3. **Tiny 数值检查**：按当前 I，只测 M=1/4/16 和一个较大回退点；覆盖零值、正负饱和、BF16 舍入边界与实际层 tensor。逐阶段核对 g/u、s/z、y；激活舍入需一致，GEMM 遵守原数值门槛，不以放宽容差通过。最终检查 sigmoid gate 后结果。
4. **C2 真融合**：实现一个 SM90 局部执行岛；FC1 按 I 分块，保留原舍入后把 z 放共享内存/DSM，再由分布在输出 H 上的 CTA 做 FC2。共享的只是 M×I 激活，不宣称所有权重常驻片上；禁止每个 FC2 输出块重算整个 FC1。按 [CUDA 13.1.1 DSM 规则][dsm]核对工具链和 cluster 起止同步；先写清 tile/cluster 所有权、临时字节和权重重读次数，再写 kernel。
5. 对同一实际 M 集合检查 Graph replay、batch/padding 切换和权重 reload；buffer 在 capture 前分配。保留 aux stream/MK overlap 接口及事件；若 cluster 占用挤掉 routed kernel，记录干扰，不同时改 stream 阈值。首版不顺带融合 scalar gate 或 all-reduce。
6. **A0=最佳原 MLP；A1=顺序 wrapper；A2=执行岛**。A0/A1 先证明接入无变更；A1/A2 只改三算子的执行边界。每点至少三次交错，主场景用真实高频小 M，回归用混合 prefill/decode；TP/EP/SP、外层 batch、量化和 MTP 均固定。
7. 交付默认关闭 patch、形状 predicate、逐阶段误差、Graph/服务回归及 rollback commit。按照[统一协议][bench]记录 TPOT/TTFT 分位数、完整输出/总吞吐、错误率、显存、fallback 比例及 shared/routed 并发 trace。

## 4. 不把局部收益重复计入服务收益

本仓库推导 hypothesis：理想两流完全重叠时，局部步时近似 `max(T_routed,T_shared)`。实际改善应比较：

`Δ=max(T_routed,A,T_shared,A)−max(T_routed,B,T_shared,B)`。

融合可能使 shared 更快，却因资源争用使 routed 更慢；没有可见等待时，shared 微基准加速也可能毫无端到端收益。片上暂存的最低 z 大小为 `2*M*I` 字节，另计 FC1 tile、累加器、同步与对齐；它不是整岛总显存预算。

任何 dtype/math/SiLU 舍入变化、shared 重复相加、MTP/Graph 异常即回滚 wrapper。若权重重读/cluster 同步抵消收益、总收益在波动内或尾延迟退化，则停止，不通过关闭既有 overlap 制造成功。本机 baseline/candidate ID、实际 I、kernel 分派、实现 SHA、收益和质量均 **unknown**；没有现成同模型 H800 全 MLP 融合实测可引用。

[sha]: https://github.com/vllm-project/vllm/tree/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f
[next]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/models/qwen3_next.py
[mlp]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/models/qwen3_moe.py
[stream]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/runner/shared_experts.py
[fse]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/utils.py
[gemm]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py
[actquant]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fusion/fused_act_quant.py
[actcuda]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/csrc/activation_kernels.cu
[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json
[bench]: ../../../docs/benchmark-protocol.md
[dsm]: https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/02-basics/writing-cuda-kernels.html


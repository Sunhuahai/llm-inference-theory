# T028 · 640 宽专家的常驻 CTA 工作队列

实施前先读 [optimize-qwen-h800 skill](../../../skills/optimize-qwen-h800/SKILL.md)，一次只执行本卡的一个阶段，通过检查点后再继续。

- 状态：`conditional`；日期：2026-11-20（Asia/Shanghai）。
- 证据：`literature + hypothesis`；H800 实测：无；优先级：条件储备。
- 目标：降低本地 routed expert GEMM 的调度/尾波次成本；关注 TPOT、输出吞吐。
- 固定：checkpoint、全部 dtype/scale/math、top-10、专家归属、TP/EP/DP、Graph、总卡数与外层 batch。

## 新点与源码边界

本卡只改变**一个本地 GEMM 内不同输出 tile 分配给 CTA 的方式**。T010 的后端选择/EPLB、T018 的 collective 融合、T019/R03 的 token 分段与工作区都不属于本卡。若实际 backend 已有等价 persistent grouped GEMM，记 `baseline-covered`，不能换成 Triton 再宣称取得新增收益。

固定 vLLM `cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f`：[TritonExperts.apply][experts] 调用 [invoke_fused_moe_triton_kernel / fused_moe_kernel][kernel]；当前 grid 为 `ceil(EM/BM)×ceil(N/BN)`，每个 `tl.program_id(0)` 执行一个 tile，已含 grouped ordering、padding 和 remote-expert 跳过。拟保留其算术主体，用有限常驻 CTA 按 `tile_id += worker_count` 领取后续 tile；首版采用静态步进，不增加 atomic 抢队列。硬件本来就调度普通 grid，常驻版本**不保证**改善负载均衡。

[NVIDIA CUTLASS 的 SM90 grouped scheduler][scheduler]提供 `PersistentTileSchedulerSm90Group` 与 `advance_to_next_work`，固定 SHA `af951046cd9cac9301d8e3e90863fd8d7edc9fa9`。其 [Hopper CuTe grouped 示例][demo]可借鉴队列和 descriptor 生命周期：顶部说明仍写 FP16/BF16，但实际 dtype 校验已包含 FP8，不能仅凭说明判支持范围；源码也未给出本模型 block128 scale 的等价消费合同，不能直接替换。本卡不采用 SM100 的硬件 Cluster Launch Control，也没有 H800/Qwen4Exp 外部端到端收益证据。

## 模型算子与进入条件

[官方 config][config]给出 H=2560、I=640、512 experts、top-10。以合法完整专家分片为例：FC1 `M_e×2560` 乘 `W13[1280,2560]`，FC2 `M_e×640` 乘 `W2[2560,640]`；原 block128 scale 分别 `[10,20]`、`[20,5]`。EP8 仅作形状例子，每 rank 64 个完整专家，不代表用户卡数。plain TP 的 160/80 宽、scale refine/padding 必须另按真实 loader 核对；首版只接受原完整 block128 的已运行形状。

只有 profile 显示该 kernel 的 tile 调度/尾部留下可测空间才进入。小 M 下可能根本不足一波，persistent 会更慢；低 occupancy 若来自寄存器或单 tile 太大，不能靠队列解决。实际框架不是此 vLLM 路径时先标 `blocked`，不要照抄路径修改别的引擎。

## 实施步骤（最多六步）

1. **C0 审计**：记录实际 class、FC1/FC2 的 M/N/K、tile 配置、FP8/scale 指针与 stride、TMA/SWAP_AB、寄存器、有效/空 tile 数；复用已有 profile。若已有 persistent 或该边不占关键路径，退出。
2. **C1 最小 commit**：只复制 FC1 的当前 kernel 为实验分支，抽取可被单 tile 与 persistent 共同调用的算术主体；新增内部默认关闭的调度选择，不虚构现成 CLI。grid 的 worker_count 静态确定，先固定一个值；FC2 保持原路径。
3. 保留每个 tile 的 K 遍历、FP32 accumulator、逐块 scale 乘法、router weight 与最终 cast 次序，`SPLIT_K=1`；禁止 Stream-K、重排 K、fast-accum 开关和 scale 合并。原 early-return 改为“本 tile 结束、继续下一 tile”，尤其不能让 remote `-1` 或尾部 tile 提前杀死整个 worker。
4. 审计新循环的 TMA descriptor、scratch allocator、输出唯一 owner 和 Graph 生命周期；保持同一捕获形状、固定地址，禁止 host 读取动态 tile 数。预分配若新增 scratch 必须计入峰值，不能偷占其他池预算。
5. 单层回放真实 top-10：覆盖专家空/热倾斜、零本地 token、M 尾行和原 Graph padding；逐 tile 对照输出、NaN、越界/竞态。只有算术及编译器精度审计通过才比较速度。
6. **C2 接线**：唯一变量仍为 FC1 调度，按[协议](../../../docs/benchmark-protocol.md)至少三次交错 A/B，主测 long-decode，回归 prefill/mixed；通过后才另建 FC2 对照，不能合并首项收益。

## 收益边界、退出与交付

令 `T=Σe ceil(m_e/BM)×ceil(N/BN)` 为有效 tile 数，`S` 为可用 SM 数，`q` 为资源约束下每 SM 并发 CTA。`ceil(T/(S q))` 只是理想波次数，真实队列还包含空 tile、不同成本与调度开销。仅当减少的尾部/设置成本大于新增循环与失去并发的成本才有收益；若 GEMM 占服务时间 f、局部加速 s，服务加速上限为 `1/(1-f+f/s)`。

报 kernel p50/p95、尾波次、SM 活跃度、寄存器/occupancy、HBM 字节、额外显存及 TTFT/TPOT/输出吞吐。任何改变 math、漏/重算 top-10、Graph 退化、服务收益落在噪声内或尾延迟越界，均退出；回滚单一分派开关及实验 commit，恢复原 grid，释放新 scratch。交付环境/shape manifest、源码 diff、正确性与 A/B 原始记录、明确 verdict；implementation commit、基线 ID、收益目前均为 `unknown`。

[kernel]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/fused_moe.py
[experts]: https://github.com/vllm-project/vllm/blob/cc23ce153ec82cdda7046fb0d8369f8cd4dfba4f/vllm/model_executor/layers/fused_moe/experts/triton_moe.py
[scheduler]: https://github.com/NVIDIA/cutlass/blob/af951046cd9cac9301d8e3e90863fd8d7edc9fa9/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp
[demo]: https://github.com/NVIDIA/cutlass/blob/af951046cd9cac9301d8e3e90863fd8d7edc9fa9/examples/python/CuTeDSL/cute/hopper/kernel/grouped_gemm/grouped_gemm.py
[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json

# T018 · Qwen4Exp 输出投影与 collective 的 tile 级融合

> 深入实施入口：[R03 任务卡](../tasks/R03-moe-pipeline.md)。新增核查区分内部 chunk 完成边和真正 tile-ready 接口；现成 SM90 block-FP8 MegaMoE 也有 FC1 scale 重缩放，不能仅凭名称判断符合原精度。

- 状态：`conditional`
- 调研日期：2026-11-20（Asia/Shanghai）
- 目标指标：TPOT、输出吞吐量、通信尾延迟。
- 优先级：储备；不增加本轮立即执行的三项任务。
- 证据来源：`literature` + `hypothesis`；本轮 H800 实测：无。

## 模型与真正要改的边

研究对象是 [Qwen/Qwen3.8-Flash-Next-FP8 的 Qwen4Exp 配置][config]：H=2560、48 层，其中 36 层 GDN、12 层 QSA；512 个 routed experts、top-10、专家中间维 640。官方配置将 attention/GDN 投影和 shared expert 排除于 FP8 转换，因此首入口是**现有 BF16 输出投影**，不转换 routed 权重。实际 checkpoint revision、dtype 与部署 SHA 仍须核对。

以下映射来自 vLLM **`12924016eb8dbaae87d5163a1b7ec2614a44a797`**。p 是 attention TP 度，M 是本次真实执行行数，含 graph padding 或投机验证行，不能直接当请求数。[模型层][model]、[QSA][qsa]、[GDN][gdn]定义了这些边。

| 算子边 | 本地 GEMM 的 M/N/K | dtype、scale 与 collective |
| --- | --- | --- |
| 36×`linear_attn.out_proj` | M / 2560 / (48×128/p) | BF16，无 FP8 scale；通常接 TP all-reduce |
| 12×`self_attn.o_proj` | M / 2560 / (24×256/p) | BF16，无 FP8 scale；通常接 TP all-reduce |
| `shared_expert.down_proj` | M / 2560 / (640/p) | BF16；`reduce_results=False`，不能假设紧接独立 all-reduce |
| EP8 routed FC1 / FC2 | 各专家路由行数 / 1280 / 2560；各专家路由行数 / 2560 / 640 | 原 FP8 block128；dispatch/combine 或模型 late reduction，非默认首入口 |

前两行 TP4 的 K=1536，TP8 的 K=768。逻辑输入为 `[M,K]`，权重存储 `[2560,K]`，常规 contiguous stride 分别为 `(K,1)`；数学右操作数 `W.T` 为 `(1,K)`，输出 `[M,2560]`。这只是未打包布局，实施时打印实际 stride/packing。开启 HC-SP 的既有基线在模型层把这两条边改接 `sp_reduce_scatter`；本实验沿用它，禁止额外开关 SP 来制造收益。

两种输出投影虽然矩阵形状相同，生产者不同。GDN 输入来自原门控归一化后的价值分量，QSA 输入来自原稀疏注意力输出；本卡从这些输入已经就绪的位置开始，不改变递归状态更新、稀疏检索预算或注意力选择。记录每层实际发生次数，单独列出 MTP 验证调用，不能把主模型四十八次调用机械乘到所有请求阶段。shared 分支的宽度虽然也为六百四十，其通信依赖与注意力输出不同，不能因形状相似就合并结论。

[shared expert 实现][shared]与 [MoE runner][runner]表明：shared/routed 均未归约时，先合并再 late all-reduce；combine 已归约 routed 时，可能单独归约 shared；SP 又有自己的归约。因此不能直接把 shared down GEMM 与“下一次 AR”硬拼接。后续 HC 的依赖也必须保持。

## 新点、基线覆盖与精度边界

[T010](T010-moe-communication.md)研究后端、普通 overlap 和负载；本卡固定同一数学运算、拓扑、dtype 与 GEMM 计算路径，将**整 tensor 就绪屏障改成 tile 就绪依赖**。GEMM 产出 tile 后立即归约，目标仅是隐藏尚暴露的通信；下游需完整归约时仍等待完整结果。

对一块输出区域 J，数学关系仍为 `Y[J]=Σ_i X_i W_iᵀ[J]`，只是允许第一个本地部分完成时就启动通信。每 rank 的完整输出 payload 为 `5120M` 字节；若现有算法恰为 ring all-reduce，理想传输主项为 `2(p−1)/p×5120M`，TP4/TP8 分别是 `7680M/8960M` 字节。这是流量推导，不能换算成固定时延；真实算法、链路和同步另测。单行 decode 的完整 payload 只有五 KiB，常受启动成本支配，进一步切小可能更慢。因此不能把每个 tile 简单变成一次独立主机 NCCL 调用；需证明设备侧通知与原 collective 接口可表达所需依赖，且没有新的串行启动链。

当前 [Qwen4Exp low-latency GEMM][skinny]已有 SM90 专用表，注释说明基于 **H200** hot-cache/L2-flush 与 graph replay 测量；TP4 `(N,K)=(2560,1536)` 的 M=1、2、4 已有 skinny kernel。只切到该 kernel 属于基线核对，不能算本卡创新，也不能把 H200 数字写成 H800 收益。若部署已有 tile collective，同样记 `baseline-covered`。

[FLUX][flux]、[COMET][comet]提供细粒度通信计算依赖的机制依据。[Triton-distributed GEMM-AR][gemmar]固定 SHA 为 `451e7d084736b1021fb7ea404f0c620b9d785dad`。其 [8×H800 dense demo][h800]仅证明另一模型上的工程可行性，包含部分形状退化，不能替代上述 Qwen4Exp 算子映射。[FlashComm SM90 fused dispatch kernel][sm90]只接受 FP16/BF16；SM100 性能表和普通 FP8 row/column-scale GEMM 也不证明支持本模型 block128。

EP8 时 routed 的有效 TP=1，每卡 64 个完整专家；未打包 `W13=[64,1280,2560]`、`W2=[64,2560,640]`，FP32 scale 网格分别 `[64,10,20]`、`[64,20,5]`。动态 activation scale 的逻辑组宽 128，FC1/FC2 每行分别 20/5 组；实际转置/padding 另查。Triton 要求权重末维 stride=1。该路径必须保留原 FP8 值、scale、累加与通信精度，不能照搬 BF16 FlashComm。

**纯 TP 的合法性不能只看 640 可整除。** [FP8 oracle][oracle]对 TP4 的 160 宽分片可把原 scale 无损重复为 32 网格；TP8 的 80 宽分片只得到 gcd=16，不能走此 refine。当前显式 backend 还可能走完整 128 块 padding：[loader][loader]会将 local 80 补到 128，原 640 宽仅占前五份，其余填零，总分配宽度为原来的 1.6 倍。是否选择该分支及 kernel 支持须看日志；recipe 的 plain TP8 限制不能升级为永恒结论，源码可加载也不等于 H800 高效。布局变化需另建基线。另 [native MegaMoE][mega]会将 block FP8 重量化为 MXFP8，违反本研究固定制品约束，不作为捷径。

## 触发条件与最小 A/B

先在现有最佳配置确认前两条投影存在可见的 `GEMM → collective` 等待，记录最慢 rank。纯 TP1 无此入口；等待主要来自别的慢 rank、可隐藏部分极少或已有同粒度实现时停止。

1. 导出合法的实际 M/N/K、stride、进程组、reduce dtype、graph 路径及模块名；先只改一层的一条边。
2. 同一改写 GEMM 保留“整 tensor 屏障”和“tile 屏障”两个模式，固定 tile/GEMM 配置隔离依赖粒度；同时对比原最佳实现，防止新基线变慢放大比值。
3. 固定模型、总 GPU 数、SP/EP/TP、request batch、输入/输出规则、MTP、sampling、缓存和 graph；至少三次交错重复。以真实低并发 decode 为主，高负载 decode、长 prefill 作回归，按[测量协议](../../../docs/benchmark-protocol.md)执行。

实施记录须把“计算核更快”和“等待更短”拆开：先比较原计算核与改写核的独立计算时间，再比较相同改写核的两种屏障模式，最后检查完整请求。若通信占用更多计算单元使矩阵乘变慢，即使时间线上两者重叠面积增大，也不代表节省了关键路径。对每个常见行数分别保留结果，不能将最适合的大预填充形状加速比外推到单行解码。真实模型调度存在其他进程组或状态等待时，记录其位置而不是把全部空白都归因于这条 collective。

进入开发的材料应包含脱敏形状清单、当前线性方法类名、一次完整依赖链以及单边可回退补丁。模型配置写着排除量化只是首要证据，仍要用加载后的权重和输入张量核实；若部署使用另一种八比特制品导致这些投影精度不同，停止迁移，重新建模。多卡数量也不应直接用几个并行参数相乘推断，应保存实际进程组成员，避免在不同资源预算间比较。

局部完全串行时的空间为

\[
t_0=t_g+t_c+t_s,\qquad t_1\geq\max(t'_g,t'_c)+t_{\rm sync},\qquad
S_{\rm request}=\frac1{1-f+f/s}.
\]

撇号包含通信抢占 SM、细 tile 和寄存器压力引起的退化；f 是该路径占实际关键路径的比例。已有重叠只能算剩余等待。没有 profile 不填百分比。报告每 rank 的 GEMM/通信/屏障时间、HBM/NVLink 流量、资源占用及服务 TTFT/TPOT 分位数、输出/总吞吐、峰值显存和错误率。

## 正确性、退出与回滚

检验发布/消费的内存可见性、跨 rank fence、graph 多次 replay、buffer 重用、取消请求及尾块；保持 accumulation/reduce 精度。验证层输出误差、完整生成质量及原业务功能；浮点求和顺序变化须记录。不能只看 kernel 微基准或要求随机采样逐字一致。

同步开销抵消收益、合法 dtype/scale 路径缺失或端到端退化则停止；死锁、越界、状态污染或质量回退立即恢复原 kernel 与 graph。实施 commit、环境、baseline/candidate IDs、实测收益和证据文件均为 `unknown`。本轮仅保留该单边适配设计。

[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json
[model]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/model.py
[qsa]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/qsa.py
[gdn]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
[shared]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/models/qwen3_next.py
[runner]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/fused_moe/runner/moe_runner.py
[skinny]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py
[oracle]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/fused_moe/oracle/fp8.py
[loader]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/fused_moe/routed_experts.py
[mega]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/mega_moe.py
[flux]: https://arxiv.org/abs/2406.06858
[comet]: https://arxiv.org/abs/2502.19811
[gemmar]: https://github.com/ByteDance-Seed/Triton-distributed/blob/451e7d084736b1021fb7ea404f0c620b9d785dad/python/triton_dist/kernels/nvidia/gemm_allreduce.py
[h800]: https://github.com/ByteDance-Seed/Triton-distributed/blob/451e7d084736b1021fb7ea404f0c620b9d785dad/docs/getting-started/e2e/e2e_dense.md
[sm90]: https://github.com/ByteDance-Seed/Triton-distributed/blob/451e7d084736b1021fb7ea404f0c620b9d785dad/FlashComm/python/flash_comm/ep_overlap/kernels/cutedsl_dispatch_group_gemm_sm90.py

# T016 · Qwen4Exp HC 小 M 投影与门控融合

- 状态：**conditional**；第二轮主线候选，按本机占比决定实施顺序。
- 调研日期：2026-11-20（Asia/Shanghai）。
- 目标指标：TPOT、输出吞吐量；重点为低/中并发 decode 的 HC 小矩阵链。
- 证据：**literature + hypothesis**；本轮无本机实测。
- 基线：**Week 1 当前最佳有效配置**。用户报告首轮已全部尝试，各项效果 unknown；保留已有 Graph、HC sequence parallel、MTP 和采样，不重做 T001–T013。

## 为什么针对这个模型

[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json)为 H=2560、HC=4、lowrank=320、48 层。固定审计 [vLLM SHA 12924016eb8dbaae87d5163a1b7ec2614a44a797](https://github.com/vllm-project/vllm/tree/12924016eb8dbaae87d5163a1b7ec2614a44a797)，该模型不是普通 dense Qwen：每层 attention 前、MLP 前各有一个 GatedResidual，末尾还有 final mixer，完整 target forward 共 **2×48+1=97 个逻辑 HC mixing 调用点**。这不是每卡 97 次实际 launch 的保证；PP 只执行本地层，HC SP 改变每次的本地 M，MTP/编译融合还会改变调用次数。

该版本已有 HC up+gate_mix 的 SM90 融合 kernel，但 **MIN_FUSED_UP_M=512**；小于 512 时仍先把 gate 写回，再读入做 sigmoid 和四路混合。目标是补上这条明确的小 M 路径，不重复 T005 的 GR/PLE 序列并行，也不把已经存在的 PDL 当成新优化。

## 已核源码、张量和已有覆盖

以下文件均使用上述固定 SHA；本机运行 SHA、分支与 dispatch 命中仍须核对。

| 源码入口 | 对目标模型的具体结论 |
|---|---|
| [model.py::Qwen4ExpDecoderLayer / Qwen4ExpModel](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/model.py) | 每层两个 BF16 HC，加末尾 use_combine=False mixer；不是修改 FP8 MoE GEMM |
| [hyperconnection.py::GatedResidual._up_gate_mix](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/hyperconnection.py) | lowrank [M,320] × W_up [10240,320]；分离路径产 gate [M,10240]，再输出 [M,2560] |
| [ops/hc.py::hc_up_gate_mix / _hc_up_gate_mix_sm90_kernel](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/ops/hc.py) | SM90、HC=4、BF16、W_up=(10240,320) 有实现；现有 tile 为 BM=64、BN=128、BK=32，num_warps=4、num_stages=4 |
| 同文件 MIN_FUSED_UP_M | 阈值 512；源码“B300 从约 M=32 胜出”注释不是 H800 结果，不能直接照搬 |
| hyperconnection.py::_down_and_inject | 普通 HC 合并 down+inject：W=[336,10240]，有效 320+4 列、12 列 padding；final mixer 为 W=[320,10240]，仍独立 GEMM→hc_silu |
| [low_latency_gemm.py::QWEN4_EXP_SM90_GEMM_PLANS](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py) | final-down 在 M=1/2/4/8 已有 H200 测得 skinny 配置；W_up=(10240,320) 不在该表，不能拿未优化的 final-down 作基线 |

上述 HC 权重以 params_dtype=bfloat16、quant_config=None 创建，down 禁用 TP 切权重，up 为 ReplicatedLinear；因此 GEMM 的 N/K 不随 TP 缩小，HC SP 主要改变 M。这里**没有 FP8 block-scale 张量**；“保持 scale”指保留 HC 除以 4、norm affine、sigmoid 和舍入边界，不能改成 FP8 或直接复用 FP8 fast-accum 示例。

PDL 去重证据：hc.py 的 norm/SiLU/gate_mix/combine_norm 已含 gdc_wait 和 gdc_launch_dependents；[skinny_gemm.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm.py)启用架构支持的 PDL；[hc_down_silu.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py)已做普通 HC 的 BF16 down+SiLU 融合、M≤48 分派、M=1 权重预取。单纯“开启 PDL”在这些路径属于 baseline-covered。

## 数值契约和适配门槛

现有 SM90 up 融合使用 FP32 dot accumulator，**先 round gate 至 BF16，再转 FP32 做 sigmoid**；四路按原顺序累加后除以 4，输出 BF16。新 kernel 必须保留这些边界，不能把 sigmoid 直接作用在未舍入的 FP32 GEMM 结果上。最终 down 也须计算 silu(float32(bf16(GEMM))/4)，保持原输出 dtype。改变 reduction 顺序仍可能产生误差，按仓库协议逐算子验收，不把同 dtype 当作逐 bit 一致。

现有融合要求关闭 batch-invariant 路径，且 up 仍是 ReplicatedLinear；LoRA 包装不得被绕过。若用户需要 batch-invariant 或对应 LoRA，先保留原路，不能关闭要求换速度。本机实际 dtype、精度覆盖项、量化排除表、Graph token count 和 HC SP local M 全部记录。

## 三组分开的最小对照

**A0：当前有效小 M 基线。** 记录各 HC 的 local M 分布、GEMM→gate_mix 时间、相邻 kernel 重叠，以及 HC 占整个 decode 的比例。真实输入、权重、TP/SP、Graph、精度和 sampling 固定；不升级引擎与 kernel 同时测。

**A1：仅改变现有融合的分派范围。** 不写新算法，选择真实出现的 M 桶，把现有 SM90 fused kernel 接入白名单；逐桶与 A0 对比。M=1/2/4/8/16/32/64/128/256 仅为候选桶，不能用全局 batch 代替 local M。BM=64 可能在很小 M 浪费计算，失败不代表所有小 M 融合无效。A1 通过的桶才保留，不把阈值直接全局降为 1。

**A2：新 H800 小 M kernel，独立测。** 保持 A1 的白名单和其他设置，只替换 up→gate_mix 实现；按 M 选择 FMA/warp reduction 或较小 BM tensor-core tile，将匹配的四路 gate 列与 xn 同时消费，避免完整 gate 落地。不得同时修改 norm、down、SP 和跨层状态。分别报告 A2/A1 与 A2/A0；A1 退化时仍可在隔离微基准研究 A2，但不得把退化路径部署为新基线。

**B：final mixer down→SiLU，独立第二变体。** A 之外单独比较原 skinny/F.linear+hc_silu 与 dedicated BF16 epilogue；N=320、K=10240、无 injection。普通 down helper把 hc 同时用于注入列数和除数，不能简单传 hc=0，或让 320 行权重被当成 324 行。实现须拆分“零 injection 列”和“除以 HC=4”。target 只有末尾一处；单层 MTP 每步另有末尾一处，空间通常小于 A，不把两项收益混记。

## 测量、收益公式和退出

至少三次交错重复，主场景为真实低/中并发长 decode，回归包含 prefill、混合流量、MTP 接受/拒绝和请求取消。微基准轮换真实各层权重；单个 HC 反复热缓存不能代表 97 个不同权重。保留现有 Graph 和已调优 kernel；报告 TPOT/TTFT 分位数、输出/总吞吐、错误、显存和 HC 分段时间。

以下为本仓库推导 hypothesis：一次 up 融合理论上省去 BF16 gate 的写回和重读，即 2×M×10240×2=40960M 字节，但每个 W_up 仍有 6,553,600 字节；小 M 下权重读与 launch 更可能主导，不能按节省激活字节直接承诺比例。无 PP 时 97 处只是收益放大机会，不能把单核热缓存节省乘 97 当成实测。

令 f 为实际可替换 HC 链占 decode 的比例、r 为该链加速、ε 为新增开销比例，整体推导为
\[
S=1/(1-f+f/r+\varepsilon).
\]
本机 f/r/ε 与端到端收益均 unknown。

按[统一协议](../../../docs/benchmark-protocol.md)校验中间 gate rounding、混合输出、长 decode 状态和质量；保留 target 的多流 [T,10240] MTP buffer，以及原 SP/PP 生命周期。A1/A2/B 使用独立开关与请求兼容回退。若数值不通过、该链占比太低、HBM/寄存器竞争抵销收益，或改善落在波动内，记录 blocked/regressed/inconclusive 并停止扩展。全模型 persistent/MegaOps 重写留储备，本轮不启动。

## 结果与下一步

本机 baseline/candidate ID、实现 commit、各 M 桶表现、质量和收益：**unknown**。下一步先核实当前构建是否已有相同小 M 分派；若有记 baseline-covered，否则交实施 Agent 做 A0/A1，再决定是否值得写 A2。B 独立立项，不与 A 捆绑。


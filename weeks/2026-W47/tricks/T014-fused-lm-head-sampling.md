# T014 · Qwen4Exp LM-head 与精确采样融合

- 状态：**conditional**；第二轮**储备项**，优先级低于已定位的 HC/GDN/PLE 主线。
- 调研日期：2026-11-20（Asia/Shanghai）。
- 目标指标：TPOT、输出吞吐量；先证明输出端有实际占比及兼容请求。
- 证据：**literature + hypothesis**；本轮无本机实测。
- 基线：**Week 1 当前最佳有效配置**。用户报告首轮均已尝试，逐项结果 unknown；不要求重做 T001–T013。

## 模型专属调用链与 dtype

[官方 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json)为 hidden_size=2560、vocab_size=248320、text dtype=bfloat16，lm_head 在 modules_to_not_convert 内。固定审计 [vLLM SHA 12924016eb8dbaae87d5163a1b7ec2614a44a797](https://github.com/vllm-project/vllm/tree/12924016eb8dbaae87d5163a1b7ec2614a44a797)后可确认：Qwen4Exp 的 target/MTP 都构造未传 quant_config 的 ParallelLMHead。默认合同为 BF16 head，**不是 FP8 head 改成 BF16**；本机权重、hidden、head_dtype override、累加/输出 dtype 仍待实查。

完整 head 权重形状为 [248320,2560]，635,699,200 个元素，BF16 约 1.27 GB。词表 TP 切分后，TP4 的 N=62080、TP8 的 N=31040，K=2560；M 是实际取出的 logits 行数，开启 MTP 验证后不等于请求数。实际 padding、LoRA 和分片索引从运行时记录。

| 固定 SHA 的源码/符号 | 已核执行路径及约束 |
|---|---|
| [model.py::Qwen4ExpModel.forward](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/model.py) | final hyper_connection_mixer 产生 [T,2560] sample_hidden 和 [T,10240] multi_hidden；HC SP+MTP 时打包 12800 维一起 gather，后者写入 _mtp_hidden_buffer |
| 同文件 Qwen4ExpForCausalLM.compute_logits | 调用 logits_processor(lm_head, hidden_states)，未调用局部 top-k/argmax，也未传 skip_gather |
| [logits_processor.py::_apply_head / _get_logits](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/logits_processor.py) | 先物化本地 [M,V/TP] logits，TP>1 再 gather 或平台指定 all-gather；head_dtype override 可改变输出路径，必须保留 |
| 同文件 get_top_tokens/get_top_k_tokens | 已能归约候选、减少通信，但仍先 _apply_head 物化本地 logits；函数存在不等于 Qwen4Exp 已接入 |
| [low_latency_gemm.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py) | SM90 的 [62080,2560]、M=1/2 已有 H200 测得 skinny GEMM；不匹配的 M/shape 回落。应对比实际分派，不能拿普通 F.linear 弱基线 |
| [mtp.py::Qwen4ExpMTP.compute_logits](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/models/qwen4_exp/nvidia/mtp.py) | 同样物化 logits；该类未实现 get_top_tokens |
| [eagle/utils.py::load_eagle_model / share_target_lm_head](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/v1/worker/gpu/spec_decode/eagle/utils.py) | mtp/speculator.py 经此加载；该类无 has_own_lm_head 声明，未设置新 draft quant 时 draft.lm_head 直接别名为 target head |

Qwen4ExpForConditionalGeneration 继承的 compute_logits 包装最终进入上述 language_model。外层继承的 compute_logits_local 包装也不能证明支持 batch-sharded sampling：此 SHA 的 Qwen4ExpForCausalLM 内层没有对应实现，须核对实际 gate 与版本，不能把它当作已覆盖收益。

## 哪些请求仍需要 logits

[model_runner.py::sample](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/v1/worker/gpu/model_runner.py)先算 sample logits，再应用 grammar，之后调用普通 sampler 或 rejection_sampler。[rejection_sampler.py::_verify](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py)使用处理后的 target logits 和可选 draft logits 验证；**target 的输出 token 不足以替代现有验证接口**。

这是现有接口限制，不是“所有投机验证数学上都需要完整 logits”。[rejection_sampler_utils.py](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py)的纯 greedy 判定只需每个验证位置的 target argmax 与 draft token；原配置本就满足条件时可另研究精确 argmax 接口。仍要覆盖全部验证位置、bonus token、mask/tie 和所需 logprobs，不能把原随机采样改成 greedy 来换速度。

[speculator.py::sample_draft](https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/v1/worker/gpu/spec_decode/speculator.py)在普通 greedy draft 路径也是 compute_logits 后 argmax。use_local_argmax_reduction 要求模型实现 get_top_tokens，Qwen4ExpMTP 在此 SHA 会被接口校验拒绝。probabilistic draft 或无 confidence head 的 adaptive verification 需要完整 draft logits，不得套用只返回赢家的路径；T015 当前模型适配限制另行记录。

共享权重不意味着 target 与 draft 具有同一接口要求。修改共享 lm_head.quant_method 会同时影响双方，必须用调用级 dispatch 隔离；不能删除 final mixer 的多流 buffer、SP gather 或 MTP 状态。原 MTP 已启用时保持它，不为制造融合收益而关闭。

## 机制、证据和功能范围

[FlashSampling v4](https://arxiv.org/html/2603.15854v4)（2026-11-12）将 head GEMM 与采样归约融合，避免完整 logits 写回/重读。kernel 实验含 H100，服务实验为 B200 上普通 Qwen3/Llama；不是本模型 H800 实测。

固定 [官方代码 SHA 834fab55b638a32e13e0a0bc525ee5aeebfb0cc9](https://github.com/FlashSampling/FlashSampling/tree/834fab55b638a32e13e0a0bc525ee5aeebfb0cc9)：core.py::fused_mm_sample_triton 为融合入口；tl_fused_mm_topk.py 是每 tile top-k 再 merge，**不同于论文附录 N 的 group-max 筛选/重算**。[作者 vLLM PR #13](https://github.com/tomasruizt/vllm/pull/13)仍 open，head 7a74973e4dc727df979f2a5ec9fff64ac5319467，作者 fork 的开关不能冒称主线现成参数。

| 请求功能 | 本轮允许的范围 |
|---|---|
| greedy | 只处理本来 greedy 的分支，保持 padding mask、tie-breaking |
| temperature categorical | 保持原分布、温度和 RNG 独立性；不得取消已有 top-k/top-p |
| top-k=20→top-p=.95 | 在论文附录 N 单 GPU k≤28 范围；公开实现与附录不同，TP 合并另需验证 |
| pure top-p / min-p | 新版论文回退标准路径；不更改请求 |
| logprobs / full logits | 需完整归一化及要求的输出；首轮回退 |
| penalties、bias、grammar、混合采样参数 | 必须先完成原 logits 处理；公开 scalar 参数不能替代逐请求值，不支持即回退 |
| MTP target verification / probabilistic draft | 首轮保留现有 logits 接口；不把只返回 sampled token 的融合接入 |

数学分布等价不保证换 RNG 后同 seed 逐 token 一致。仍需检查低精度舍入、过滤边界、tie、batch 重排和 TP rank 随机独立性。所有现有 dtype、quantization、scale 与累加合同固定。

## 两个独立、可执行的后续分支

**A：先补 Qwen4ExpMTP 的局部 argmax 接口，保留原 GEMM。** 仅在当前 draft 原本 greedy、无 draft-logits cache/acceptance-estimator 需求时，给该模型增加调用 logits_processor.get_top_tokens(self.lm_head, hidden_states) 的方法并启用对应分派。A0=现有 full gather→argmax；A1=同 head GEMM→局部 argmax→候选归约。唯一变量是 draft 结果归约路径，不改变 target sampling、MTP 深度或质量要求。该分支减少全词表通信，**仍物化本地 logits**，不能把收益称为 FlashSampling 融合。

**B：head+采样融合，另设对照。** 只选择真实存在且无需完整 target verification/logprobs 的兼容请求；若实际流量全部需要它们，B 暂缓。B0=已调优 head+sampler；B1=保持同一 TP 通信策略的 fused head/sampling。若实现不可避免同时替换通信，先用捕获的真实 tensor 独立测 kernel，再明确报告组合路径，不冒称单变量融合。随后才另测候选归约，并确保所有 penalty/mask 在选候选前应用。

A/B 不联动修改共享权重，不增加量化，不改变输出长度、reasoning、sampling 或已有 MTP。保留请求级 fallback。实施前登记 local M/N/K、实际调用次数、dtype、TP/SP/PP、engine SHA、head 占比与兼容请求比例；没有兼容性证明，不接生产路径。

## 测量、收益与退出

至少三次交错重复，使用真实低/中并发 decode 和混合请求；报告 TPOT/TTFT 分位数、输出/总吞吐、失败率、显存、head GEMM/通信/sampling 时间，并记录 fallback 覆盖率。TP4 M=1/2 必须保留当前 skinny GEMM，head 已有 PDL 不能重复计为新增。

以下为本仓库推导 hypothesis：f 为实际可替换输出段占 decode 比例，r 为局部加速，ε 为额外开销，则
\[
S=1/(1-f+f/r+\varepsilon).
\]
例如 f=.12、r=2、ε=0 只有约 1.064× 整体加速；这是算式示例，不是本机预测。若大部分流量回退，f 应按真实流量重估，不只报兼容微基准。

按[统一协议](../../../docs/benchmark-protocol.md)核验数值、分布、质量、vocab padding/tie、MTP 接受/拒绝和状态回滚。已有路径覆盖记 baseline-covered；接口不支持或需要变精度记 blocked；head 占比低、fallback 高、GEMM 效率下降或收益落在波动内则停止扩展，恢复原 dispatch。

## 结果与下一步

本机 baseline/candidate ID、实际 head dtype、是否共享、兼容请求比例、实现 commit、收益和质量：**unknown**。先核验 A 的 greedy draft 条件及通信占比；B 保持储备。不得以论文速度或通用 helper 存在宣称本模型已经支持。

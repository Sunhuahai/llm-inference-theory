# R02 · PLE 静态计算缓存实施任务

> **辅助执行：** 先读 [optimize-qwen-h800](../../../skills/optimize-qwen-h800/SKILL.md)，从本工作流选择一个阶段和对应 T 编号。每次只做一个变量，不将整条工作流一起叠加测试。

- 状态：**conditional**；关联 [T020](../tricks/T020-ple-exact-cache.md)，不是新增 trick。
- 基线：W1-best；固定 Qwen3.8-Flash-Next-FP8、H800 总预算、全部实际 dtype、采样、MTP 与输出要求。结果 **unknown**，本轮无 GPU 实测。
- 源码锁定 vLLM **aefea5c8e00c47032cda6d0b75704f6cd0a7c66d**。下列 Qwen PLE、model_state、model_runner 文件与前次 12924016 审计的 Git blob 相同；本机版本另登记。

## 1. 新核查与启动门槛

[model.py][model]按 layer_idx+1 判断 PLE：配置 ple_layer_ids=[2] 对应 **layers.1.ple、ple_dense_layer_id=0**。它读取 16 行×160 元素，原 FP8 表使用全局 scale；[ple_layer.py][layer]在反量化后做 replicated 2560→12800 投影，再进入依赖 hidden 的 gate 和有历史状态的 short-conv。只缓存原始行或投影 [k,v]，不得缓存后两者。

[官方 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json)把 layers.1.ple.key_proj/value_proj 排除 FP8；实施前仍须核实合并 kv_proj 的 mapper、quant_method、实际权重/输入/输出 dtype。原反量化先把 FP8 值和全局 scale 各自转为 output_dtype，再相乘；不能改成 FP32 先乘后转，二者舍入边界不同。

**行命中不等于 tuple 命中。** 例如末尾三元为 (a,b,c) 与 (d,b,c)，末尾二元 (b,c) 相同，可重复 8 个二元头而不重复完整 16-ID 向量。各头相关且可能 hash 碰撞，不能用平均行命中率或其 16 次方估算投影命中。

新增反证：[上游 RFC #66467][rfc]作者报告 1 GB direct-mapped **host** row cache 只改善合成 prompt，真实文本 decode 重复行接近零。环境是 GB300/TP1、NVFP4 主体但 FP8 PLE；真实文本样本规模和 trace 未公开。它不证明 H800 的 owner HBM cache 无效，却要求**独立 trace 先通过复用与成本门槛**，不能预设缓存有收益。

## 2. C0：先交 trace 与预算，不改执行路径

从已有运行补采按序记录：脱敏请求/会话键、step/slot generation、实际生成位置、prefill/decode/验证标记、accepted/rejected 数、local M、ETP owner、16-ID 的顺序保持映射、重复距离、真实 host bytes、暴露的预取等待、投影耗时和 state slot 容量。原 token/完整键留实施环境；公开仅盐化标识和聚合统计。

按会话或时间切分**校准 trace**与独立**评估 trace**，登记 hash。只用校准段选择热行/热 tuple；评估段不得回流选键。分别统计原始行命中、全 tuple 精确命中和成本加权命中，分冷启动、稳定期及 MTP 验证行；不能只展示重复完整 prompt 的暖场景。

每 rank 预先给出 HBM 上限 B，计入 keys、索引、padding、固定 scratch 和复制成本；不默认借用 KV/GDN 池。原行 payload 为 160b；投影为 12800b_out，若 BF16 为 25600 字节，另含 16×8 字节完整键。C0 先离线回放 **0、B/4、B** 三点及真实低并发 decode、混合负载两场景；B 由当前余量决定。C0 只给命中和成本上界，实现后才在相同预算上测 probe/miss 的真实代价，不能将离线预测写成加速实测。容量不够就缩 B 或停止，不增卡。

## 3. 分阶段 commit 与唯一变量

| Commit | 改动与对照 |
|---|---|
| C0 trace/离线预算 | 仅观测、生成只读目录与两份 trace 报告；原路径仍输出 |
| C1 owner 原行 cache | W1-best↔固定目录 HBM lookup；只换读取来源，miss 原 UVA，ETP/SP、预取、投影不变 |
| C2 投影 memoization | 关闭 C1，W1-best↔完整 tuple 的 [k,v] 复用；先验证 early-probe/拼回，再证明 miss 路径真实少算 |
| C3 组合（可选） | 仅 C1/C2 各自通过后，相对较优单项测组合，不把收益相乘 |

C1 在 [common/ngram_embedding.py][common]的 DP IDs gather 后、实际 UVA _lookup 前 probe；FP8 cache 原字节不反量化，scale 保持原格式及应用次序。**仅词表 owner 返回非零值**，保留原 int8-view reduce/reduce-scatter，不让多 rank 命中重复相加。

C2 必须在 [nvidia/ngram_embedding.py::start_prefetch][ngram]发出读取之前判定 tuple。只在 ple_layer.forward 查询已经晚于 UVA。参与同一 ETP 查询的 ranks 必须对跳过行达成一致；首版可使用一致的只读 tuple 目录，缺目录/版本不一致全体回退。投影副本的预算与词表 owner cache 分开记，保持原 collective 的参与、顺序和 shape；省 lookup 不等于省通信。投影值只能通过原 dequantize+kv_proj 数值路径生成，不能用 CPU FP32 重算再转 BF16 代替；目录在 capture 前准备并冻结，填充耗时和临时显存计入预算。

## 4. Graph、生成位置与 MTP 正确性

probe 使用本步完成 combine_sampled_and_draft_tokens 后的 input_ids；[model_state.py][state]从 num_computed_tokens 和 all_token_ids 取得前两个已提交 token，原 ple_ngram_ids 处理 query_start_loc、EOS 和 chunk 边界。沿用这条依赖链，在前一层计算期间预取即可，**不猜尚未生成的下一 token，也不复用上一 step 的 probe mask**。

MTP target 验证行可包含候选 token；下一步 context 必须在 [model_runner.postprocess_sampled][runner]按接受/拒绝更新后重建。拒绝分支的纯静态条目仍是合法函数值，不能把其请求状态提交到 cache。gate/conv 继续使用原 num_accepted_tokens 和状态回滚。请求取消、slot 复用、DP 重排需校验 step/slot generation；side stream 读完前不得覆盖 IDs/mask/output，保留 wait/record_stream 生命周期，新增提前复用时用独立固定 buffer 和完成事件。

Graph 内不用 nonzero→CPU count→动态 tensor/临时分配。预分配最大容量 miss_indices/count/输出，GPU compaction 后以固定 grid、运行时 count 跳过无效投影 tile，再 scatter 回原 token 次序；所有地址和外部 shape 固定。**若仍执行全 M GEMM，只能计查表收益，不能声称省投影。**更换 miss GEMM 的 backend/形状需独立数值核验，保留原计算精度与 fallback。

缓存 namespace 包含 checkpoint、PLE 表/层、hash multipliers/offsets、scale、投影权重/adapter、dtype/量化与 GEMM 数值路径版本；行键为完整 global row ID，投影键为有序 16-ID 向量。短 hash 仅定位，完整比较才命中。权重 reload、owner 拓扑或版本变化失效目录；跨 M 桶复用先证明数值等价，否则把桶加入 namespace。

## 5. 成本、验收和退出

本仓库推导 hypothesis：
\[
h_{\rm cost}=\frac{\sum_i hit_i c_i}{\sum_i c_i},\qquad
G(B)=\sum_s[C_s^0-C_s^{miss}(B)-P_s-S_s-J_s]-F(B).
\]
c_i 为该命中真正避免的暴露成本；row 与 tuple 分开计。C_s^{miss} 用实际 miss GEMM 形状与剩余 lookup 测量，P/S/J 为 probe、拼回、竞争，F 为填充摊销；已隐藏的 UVA、未省掉的 collective 不计收益，重叠时间不重复相加。

交付独立 commits、开关/回滚说明、目录 manifest、校准/评估 hash、预算明细、真实 bytes/算量证据及[协议](../../../docs/benchmark-protocol.md)结果 CSV。原行逐字节一致；投影/最终 logits、业务质量与 MTP/conv 状态通过既定门槛。覆盖全 hit/miss、碰撞、EOS、边界、空/填充行、跨 owner、取消和拒绝。

每点至少三次交错重复；报告 TTFT/TPOT、输出吞吐、尾延迟、OOM/错误、cache MiB 与可运行 state slots。独立评估无有效命中、G≤噪声、挤压容量或正确性不通过即停；关闭 dispatch、等待 side stream 完成、恢复原 lookup/GEMM 和 pool，再释放 cache。**H800 收益、最优 B、实现 commit 均 unknown。**

## 可复制给实施 Agent

> 复用 W1-best，按 R02 先交校准/独立评估 trace 与成本预算；仅有复用和暴露等待时做 C1。C2 单独证明 probe 先于 UVA、完整 tuple 键、Graph miss 真少算、MTP accepted context 正确。全程固定精度/采样/预算，保留 owner 不变量和 fallback；无收益就提交负结果，不扩成 T021。

## 一手来源

[model]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/models/qwen4_exp/nvidia/model.py
[layer]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/models/qwen4_exp/nvidia/ple_layer.py
[common]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/models/qwen4_exp/common/ngram_embedding.py
[ngram]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/models/qwen4_exp/nvidia/ngram_embedding.py
[state]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/models/qwen4_exp/nvidia/model_state.py
[runner]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/v1/worker/gpu/model_runner.py
[rfc]: https://github.com/vllm-project/vllm/issues/66467

新 EngramConfig 的 RDMA/NVLink peer pool 仍只支持 engram_layer_ids，[校验明确排除 Qwen 的 ple_layer_ids](https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/config/engram.py)；不得把它当本任务已实现的缓存。


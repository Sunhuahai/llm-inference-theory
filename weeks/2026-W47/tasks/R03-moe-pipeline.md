# R03 · Qwen4Exp top-10 窄专家的有界执行任务

- 状态：`conditional`；MoE 条件储备，HC 短期实验另行推进。
- 日期：2026-11-20；证据：`literature` + `hypothesis`；本轮无 GPU 实测。
- 承接：[T018](../tricks/T018-tile-collective-fusion.md)、[T019](../tricks/T019-bounded-ep-workspace.md)，不新增 trick 编号。
- 目标：固定原模型、全部 dtype/scale、专家 kernel、总 GPU、request batch 与 prefill chunk，验证内部 token 分段的真实空间收益及尾延迟代价；不转 MXFP8，不接受静默量化 fallback，不承诺加速。

## 1. 本次新增核查与否决条件

当前上游固定 **vLLM `aefea5c8e00c47032cda6d0b75704f6cd0a7c66d`**（11 月 20 日 14:42，Asia/Shanghai）；旧卡保留初筛快照。对象为 H=2560、512 专家、top-10、I=640 的 Qwen4Exp。首版只接已核对的 `Fp8MoEMethod → FusedMoEKernelModularImpl → TritonExperts` 原 block128 路径；实际框架不同先映射类和调用边，不为匹配文档升级后混淆基线。

[模型配置检查][modelconfig]明确拒绝 PLE/QSA 下的 DBO 或 `ubatch_size>1`。[已合并 #63483][dbo]的通用纯 TP DBO 不能当本模型现成功能，不能删除检查绕过。这里只考虑 MoE 调用内部的分段，QSA、PLE、GDN 调度边界保持原样。

[Triton 源码][triton]已有 SiLU+Mul+block128 quant 融合及非本地 EP 路由跳过，命中条件时记 `baseline-covered`；其 Sorted-A TMA 还要求 `w2.size(1)≥4096`，本模型 H=2560 不满足。[#62603][fi-pr]虽已合并、测过 Qwen3.8 专家形状的 H200 合成用例，但 [SM90 权重适配][fi-code]用 `max(s_gate,s_up)` 合并 scale 并重缩放 FP8 行，可能新增舍入；排除直接替换。只借鉴全 rank 一致容量桶和空 rank 参与，不能挪用 GLM 的端到端收益。

## 2. 先确定真正执行的分叉

| 实际路径 | 最小可研究改动 | 本阶段不能声称 |
| --- | --- | --- |
| `MoEPrepareAndFinalizeNoDPEPModular` + `TritonExperts` | 一次 `_prepare` 后按行切专家计算，完整输出后按原规则 finalize/shared/late AR | 节省不存在的 dispatch 接收缓冲 |
| 真正 dispatcher，且原专家为 `TritonExperts` | 先完整 dispatch 一次，仅切专家计算，最后原 combine 一次 | 已减少 transport 或 recv 分配 |
| Monolithic、BatchedExperts、native MegaMoE 或其他类 | 首版走原路径并记录原因 | 可套同一个二维切片循环 |

[并行配置][parallel]要求 EP 且 DP>1、PCP>1 或 MoE-SP 才启用普通 all2all；TEP8 本身不足以判断。[MoERunner][runner]负责 shared/routed 合并和晚归约，不能在 chunk 循环中重新调用整个 runner，否则会重复 shared、路由或 collective。shared 对每个原 token 只执行一次；所有十条专家贡献完整保留。

## 3. 分阶段最小 commit

### C0：入口与分配审计，无行为变化

交付实际类名、量化键、权重/scale hash、M/N/K/stride、原 kernel 配置、并行组和调用图；沿用 W1-best。记录 input、A1 scales、两块 workspace、A2/scale、完整输出、shared、通信 handle 和 graph 池的分配及别名生命周期。`workspace_shapes` 在本模型给出两份 `[R,10,2560]` BF16，即 `102400R` 字节，但 R 是 prepare 后容量，不是请求数。

[modular kernel][modular]目前 `_allocate_buffers` 的两个 M 都传 `M_full`；参数名含 chunk 不表示已有循环。若真实 build 已分段并释放相同高水位，直接标覆盖；没有显存或关键路径瓶颈则停止。

### C1：只分段专家计算，保留通信次数

单层、单开关起步：一次 prepare 后，循环处理完整 token 行及其十个 top-k 槽，把结果写入独立完整输出的对应 slice；只调用一次原 finalize。固定原专家代码与合法调优配置，M/grid 改变的效率损失计入结果，不能悄悄换 kernel。

[量化实现][quant]按每行连续 128 列求 scale，所以原 block128 路径沿 M 切分不改变归约域：A1 每行 20 组，FC2 输入每条路由 5 组。优先切已有 A1/scale，保持列主序 scale 的真实 stride，禁止用错误 `.view` 重解释。比较有效行的量化值和 scale；纯 TP refine、tensor-wide scale、LoRA、post-GEMM2 hook 或布局不兼容时，首版原路回退，不能关闭业务功能。

dispatcher 分支还须遵守 [PaddedStandard 契约/#54109][padded]：尾部可能未初始化，保留 `topk=-1`、有效长度和设备端 prefix 元数据。按捕获容量固定循环次数，不以 `.item()` 读取真实接收数；需重定位的索引必须显式转换。只切 hidden 而忽略 scale、映射或 padding 判为不正确。

### C2：只有真实接收峰值仍受限，才研究通信分段

另立对照，不与 C1 合并归因。每轮源端发送上限不足以保证安全，须证明对所有目的 rank 都有 `Σ_s recv(s→r,j)≤C_r`，包含对齐和多份缓冲。同 token 同 rank 的输入去重保持原语义，十条专家执行仍完整；多轮重复发送的字节必须另计。所有 rank 使用一致轮数与 collective 顺序，空 rank 也参与。无法在原 backend 中表达这些边界，保留 C1 并停止 C2，不换拓扑或增加卡数。

原 GEMM 未暴露 tile-ready 信号，**不能同时承诺真正 tile 融合与 kernel 完全不变**。最多再单独评估“完整 chunk 就绪事件→原 collective”这一条边；它增加通信次数和缓冲存活，须独立 A/B。若原 shared/late AR 顺序无法保持，就不实现该分支。

## 4. Graph、高水位与容量真实性

[WorkspaceManager][workspace]只增长并在捕获前后锁定。单层切块后，其余层或旧 graph 仍可保留最大分配，因此单层验证只证明算法和局部存活量，不能宣称进程释放显存。正确性通过后再覆盖同类层，按相同 graph bucket 重新 profile/capture；输出不能别名到下一 chunk 会覆盖的 scratch，跨流消费者完成前禁止复用。

每个 graph 的循环次数、地址和尾块规则固定，预分配合法容量；不能降低 graph 覆盖来制造收益。保留旧图供即时 fallback 可能保留旧高水位，必须计入。若需重启/重捕获才能释放或回滚，提前写清，不能在 replay 中解锁、增长工作区。首次方案采用串行 chunk，避免双缓冲同时抬高峰值。

## 5. A/B、通过、退出与交付

C1 唯一变量为内部行容量及串行分段；C2 另以 C1 为锚点只改通信分段。请求 batch、状态池、模型/精度、图桶、采样、MTP、输出规则、负载和总卡数均固定。按[测量协议](../../../docs/benchmark-protocol.md)至少三次交错重复，主测真实大 M 场景，回归小 batch decode、空尾块、极端专家偏斜和多次 replay。

记录 `ΔM=peak_old−peak_new`，以及规划、额外 launch、小 GEMM、通信次数与等待；完整模型报告 TTFT/TPOT 分位数、输出/总吞吐、错误率和质量。不用 allocator reserved 下降代替可用容量，不先扩大状态池。收益须超过重复波动；token-expert 计数、scale、层输出与完整生成质量均通过。

精度语义改变、路由缺失、越界/死锁立即退出；省内存被额外缓冲抵消或尾延迟退化则降级。开关恢复原调用链，必要时重启并重捕获原图。交付 C0/C1/C2 各自 commit、baseline/candidate IDs、原始分配/trace、质量与负结果、覆盖/阻塞原因及回滚记录；未执行项填 `unknown`。本卡没有启动 GPU 作业。

[modelconfig]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/model_executor/models/config.py
[dbo]: https://github.com/vllm-project/vllm/pull/63483
[triton]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/model_executor/layers/fused_moe/experts/triton_moe.py
[fi-pr]: https://github.com/vllm-project/vllm/pull/62603
[fi-code]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/model_executor/layers/fused_moe/flashinfer_moe_ep.py
[parallel]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/model_executor/layers/fused_moe/config.py
[runner]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/model_executor/layers/fused_moe/runner/moe_runner.py
[modular]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/model_executor/layers/fused_moe/modular_kernel.py
[quant]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/model_executor/layers/fused_moe/utils.py
[padded]: https://github.com/vllm-project/vllm/pull/54109
[workspace]: https://github.com/vllm-project/vllm/blob/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d/vllm/v1/worker/workspace.py

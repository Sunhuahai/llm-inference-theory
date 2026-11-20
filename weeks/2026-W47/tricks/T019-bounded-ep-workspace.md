# T019 · Qwen4Exp top-10 专家 workspace 与固定容量 EP 路由

> 深入实施入口：[R03 任务卡](../tasks/R03-moe-pipeline.md)。优先核查 NoDPEP 下完整 prepare 不变的专家内 token 分段；Qwen4Exp 的 DBO 限制、尾行有效性与跨层/Graph 高水位另列为必须满足的条件。

- 状态：`conditional`
- 调研日期：2026-11-20（Asia/Shanghai）
- 目标指标：真实 workspace 峰值、KV/SSM 可用容量、相同负载尾延迟。
- 优先级：储备；不增加本轮立即执行的三项任务。
- 证据来源：`literature` + `hypothesis`；本轮 H800 实测：无。

## 具体模型与适用门槛

针对 [Qwen/Qwen3.8-Flash-Next-FP8][config]：H=2560、512 个专家、top-10、routed/shared intermediate=640。routed 保持原 FP8 block128，shared 保持现有 BF16；不改变路由、权重、激活、通信、KV 或 SSM 精度。精确模块形状与 TP 分片边界见 [T018](T018-tile-collective-fusion.md)。实际制品和部署配置仍为待核对项。

固定审查 vLLM **`12924016eb8dbaae87d5163a1b7ec2614a44a797`**。[并行配置源码][parallel]的 `use_all2all_kernels` 要求 EP 且满足 DP>1、PCP>1 或 MoE-SP。普通 Triton 的 TEP8、DP=PCP=1、无 MoE-SP 路径不自动获得 dispatch/combine：各 rank 已有输入，本地专家后归约。先记录实际 prepare/finalize class；若是 `NoDPEP`，**接收缓冲分支不适用**，只能单独审计本地 top-10 workspace，不得暗开 SP/DP 把它变成 T005/T007。

本卡改变中间缓冲容量、生命周期和路由计划。[T001](T001-prefill-scheduling.md)的 request batch/prefill chunk 必须固定；[T010](T010-moe-communication.md)的后端切换、普通 overlap、负载均衡也固定。新点只可能是已有实现尚未覆盖的**有界接收及专家计算分段**。

## top-10 不等于通信量乘十

vLLM 的 [DeepEP HT][ht]用 `is_token_in_rank` dispatch。[当前安装脚本][install]固定 DeepEP `a56d6156febcd9976e55adc85b5155bfac9f28f8`，其 [ElasticBuffer][elastic]明确同 token 同目标 rank 去重，即使多个所选专家位于该 rank，也只接收一份输入。专家计算仍须执行全部十条路由；combine 将结果回到原 token，不能去掉专家贡献。

令 U 为所有实际 dispatcher 的不同输入行总数，R_r 为 rank r 的去重接收行数，P 为 EP 度，则 `R_r≤U`，全组接收主项 `ΣR_r≤min(10,P)U`；padding、元数据和 transport 副本另计。不能把 TP 副本当不同请求，也不能用 `10U/P` 代替真实接收量。EP8 每卡 64 专家，但同 rank 多专家重用输入与本地 expert-slot 展开是两种存储语义。

按该模型尺寸可做以下**源码容量推导，非本机测量**：

| 具体路径 | 逻辑存储主项与例子 |
| --- | --- |
| 非展开接收，FP8+FP32 block scales | 每有效行 `2560+20×4=2640 B`；R=8192 时 20.625 MiB，未含路由/输出/transport |
| 非展开接收，原通信为 BF16 | 每行 5120 B；R=8192 时 40 MiB；不能为了省一半改通信精度 |
| legacy LL worst-case 接收 | `[E/P,Pm_cap,2560]`；P=8、m_cap=128 时为 `[64,1024,2560]`，单份 FP8 payload 160 MiB，加原 FP32 scales 5 MiB；BF16 则 320 MiB |
| standard Triton 专家 workspace | 两份 `[R,10,2560]` BF16，共 `102400R B`；R=128 为 12.5 MiB，R=8192 为 800 MiB |

LL 形状来自 [legacy API][legacy]；这是 per-expert 最坏容量，不是有效接收行数，原实现还有双缓冲及输出，必须按别名和生命周期审计，不能机械相加。该 pin 的 [LL kernel][llkernel]支持 top-k≤11，包含本模型 top-10；旧版≤9 的实现不能照搬。vLLM 的 [LL adapter][ll]支持 H=2560，并明确无效尾行不得参与 scale 归约。若实际 scale 为另一原生布局，按该布局重算，禁止改 scale 格式凑数字。

Triton 一行来自 [workspace_shapes][triton]：FC1 N=1280，激活后 640，均小于 K=2560，因此两块都按 2560 宽预留，包含非本地 expert-slot 的容量。[modular kernel][modular]目前传 `M_full` 给两个 M 参数，并复用中间/输出；实际 R 是 prepare 后行数，不是 request batch。这个 top-10×H 的分配比笼统“MoE 通信太大”更值得先核对。不同专家类不能套用这条公式。

## 必须扣除的已有优化与外部证据

[DeepEP manager][manager]的 HT transport 本来就按固定 MB 建池；[DeepEP v2 adapter][v2]已有 non-expand dispatch、按 DP/SP token 上界选二次幂容量 bucket、无逐 forward CPU recv-count 同步，并可重叠 shared FFN 与 combine。modular kernel 也已有 buffer alias/复用，且 workspace manager 会保留进程高水位。实际 build 包含这些就逐项标 `baseline-covered`；仅启用这些功能属于基线核对，不能重报新收益。

剩余研究入口是让 `prepare → experts → finalize` 的真实存活范围受接收容量 C 约束，并解除对应完整 workspace 的高水位预留；只改一个分配参数或只降低 allocator reserved 无效。若基线已经分段并释放相同高水位，停止本卡。

最小实现宜按原 token 行分段，每一段完整保留该行的十个专家选择，再调用原专家内核。若改成按专家分段，同一 token 可能多次发送到同一 rank，还需要跨段累加，必须把新增流量和输出存活时间计入；不能继续套用原去重接收字节。shared expert 对每个原 token 只计算一次，不能随专家路由块重复执行；其结果仍在原有归约位置合并。规划记录至少包含原 token、目标专家、目标 rank、接收槽位和有效长度，确保重排只改变执行顺序。

即使两块专家临时张量可收缩，完整层输出也可能必须保留。沿用当前复用结构时，简单容量下界为 `2×2560×[max(10C,R)+10C]` 字节，尚未计接收、通信和元数据；第一项保留完整输出或较大中间块。它解释了为什么把 C 趋近零也不能把显存趋近零。若实现分离了输出与临时空间，则按新的真实生命周期重新计算，不把此下界写成确定可达容量。模型下一层需要的完整结果、四路残差与递归状态均不属于可丢弃 workspace。

[FlashComm EPChunkPlanner][planner]及[测试][tests]提供固定容量规划参考，固定 Triton-distributed SHA `451e7d084736b1021fb7ea404f0c620b9d785dad`。[官方文档][flashdoc]的 4×SM90、M=8192、chunk=1024 示例通信时延接近原路径；它是通信微基准，未证明本模型或 H800 服务收益。只借鉴原精度 data movement，不引入 MXFP8 通信量化，也不换为 BF16 专家 GEMM。

## 两阶段最小 A/B

**A：同工作量验证机制。** 固定 request batch、prefill chunk、状态池、原专家 kernel/调优配置、全部 dtype、总 GPU 数及并行布局、MTP、graph bucket、缓存和输出规则。只改变内部容量/分段执行；分段会改变专家 GEMM 的 M，必须计入其效率损失。容量以接收端约束，不能只均分发送端；保留每条 token-expert 路由，按原规则归并。不得跨 chunk 重算原本跨块的 scale。

先导出每 rank 的输入、接收、专家两块 workspace、combine 输出、transport/graph 池峰值与别名图，再扫少量合法 C。至少三次交错重复，覆盖真实偏斜、空专家/尾块、低并发 decode 与长 prefill；遵守[测量协议](../../../docs/benchmark-protocol.md)。正式计时关闭重 profiler，解释性 trace 另采集。

同批请求下先审计最忙接收端，而不是平均卡：多个发送者同时命中一个 rank 时，单个发送块合法并不保证总接收块合法。必须给出每轮全部发送者共同满足的容量约束，以及所有 rank 一致的结束条件。保留空专家、空发送者和尾块的同步参与；禁止丢弃拥塞 token 或减小 top-k。若规划需要每轮把计数搬回主机，单列这部分等待；若用缓存计划，说明路由变化时如何失效。旧图持有大缓冲时，应重新捕获同一组形状进行对照，而不是降低图覆盖范围来省显存。

\[
\Delta M=M_{\rm peak,old}-M_{\rm peak,new},\quad
T_{\rm new}=T_{\rm plan}+\sum_j(T_{d,j}+T_{e,j}+T_{c,j})-T_{\rm overlap}.
\]

显存必须按同一时刻所有真实存活分配计算，包含额外映射、双缓冲、scale 和图池；旧完整缓冲仍被图持有时不能宣称释放。更小的 C 可能增加 launch、主机同步或小 GEMM 成本，所以省内存不必然改善 TPOT。报告 TTFT/TPOT 分位数、吞吐、规划/通信/专家时间、错误/OOM 和实际可用容量。

**B：独立验证容量价值。** 只有 A 证明真实释放且尾延迟/质量可接受后，才另建 baseline ID 扩大 KV/SSM pool；混合状态不能按纯 KV 模型估算。粗界 `ΔB≤floor(ΔM/M_request)`，实际取决于分页、SSM、索引与 graph。复用相同外部到达 trace，另列最大可行并发扫描；不能先改 batch，再把吞吐变化归因于 A。

## 正确性、退出与回滚

验证每条原 token-expert 路由恰好执行一次、rank 去重不丢专家贡献、top-10 权重对齐、无溢出丢弃、跨块 combine 正确；覆盖无效尾行、非整除 C、取消、graph replay 和 buffer 重用。保持累计/reduce 精度并验证最终生成质量。

无真实 workspace 瓶颈、仅 reserved 下降或基线已覆盖则停止；小 GEMM/规划导致尾延迟退化记 `regressed`，精度语义无法保持记 `blocked`。恢复原 planner、缓冲/状态池与 graph，保留负结果。环境、baseline/candidate IDs、实施 commit、实际字节、质量和收益均为 `unknown`。

[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/raw/main/config.json
[parallel]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/fused_moe/config.py
[ht]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py
[install]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/tools/ep_kernels/install_python_libraries.sh
[elastic]: https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/deep_ep/buffers/elastic.py
[legacy]: https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/deep_ep/buffers/legacy.py
[llkernel]: https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/csrc/kernels/legacy/internode_ll.cu
[ll]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py
[triton]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/fused_moe/experts/triton_moe.py
[modular]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/fused_moe/modular_kernel.py
[manager]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/distributed/device_communicators/all2all.py
[v2]: https://github.com/vllm-project/vllm/blob/12924016eb8dbaae87d5163a1b7ec2614a44a797/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py
[planner]: https://github.com/ByteDance-Seed/Triton-distributed/blob/451e7d084736b1021fb7ea404f0c620b9d785dad/FlashComm/python/flash_comm/ep/chunk_plan.py
[tests]: https://github.com/ByteDance-Seed/Triton-distributed/blob/451e7d084736b1021fb7ea404f0c620b9d785dad/FlashComm/tests/ep/test_ep_chunked.py
[flashdoc]: https://triton-distributed.readthedocs.io/en/latest/flashcomm/index.html

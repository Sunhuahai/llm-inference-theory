# 第二轮调研 · Qwen3.8-Flash-Next-FP8 的 H800 专项改造

> **2026-11-20 深入更新：**用户要求选取改动最大的方向并写任务卡，当前派发使用 [R01–R03](tasks/README.md)及[更新后的任务书](agent-tasks-round-02.md)。下文第 1–5 节保留初筛判断，第 6 节记录新证据与任务顺序调整。

日期：2026-11-20（Asia/Shanghai）。用户已反馈 Week 1 全部尝试，并允许本轮采用更激进的测试。随后明确要求针对这个具体模型研究。本轮仍归档在实际 ISO 周 W47，称“第二轮”，不提前创建 W48。

**初筛推荐三条：T016 的 HC 小 M 融合、T017 的 GDN 状态写回改造、T020 的 PLE 精确缓存。** 它们直接对应这个模型的四路残差、36 层 recurrent state 和大型 n-gram 查表路径。当前更大改造的阶段安排见第 6 节。研究状态与用户口述完成状态分开；尚未取得首轮逐项原始结果，本轮没有 H800 实测。

实施入口：[第二轮任务书](agent-tasks-round-02.md)。
证据入口：[模型源码审计](model-audit-round-02.md)与[固定版本来源](sources-round-02.md)。
首轮完成反馈：[2026-11-20](feedback/2026-11-20.md)。

## 1. 本轮依据的是哪一个模型

按仓库既有目标固定 `Qwen/Qwen3.8-Flash-Next-FP8`，架构为 `Qwen4Exp`。上游 vLLM 固定到 `12924016eb8dbaae87d5163a1b7ec2614a44a797`；所有实际部署参数仍从首轮环境读取。

模型是 48 层、36 GDN + 12 QSA，hidden 2560，4 路 residual，HC bottleneck 320；MoE 为 512 个 routed experts 取 10 个，另有 shared expert，expert width 640；PLE 在指定层使用 16 个 n-gram heads、160 维行；词表 248320。原 FP8 制品有大量 BF16 排除模块和 FP32 state，不是所有运算都在 FP8 上进行。完整依据见模型审计。

这些结构决定了三个具体问题：

1. 每层 attention 与 MLP 前后的四路 HC 是否还在小 M 下物化可以融合的门控中间量？
2. 36 层 GDN 是否每步或每个投机候选都读写/复制大块状态？
3. PLE 的精确 n-gram 行及静态投影是否被重复计算，而当前优化只隐藏了传输？

## 2. 三个优先实验

| 顺序 | 卡片 | 实际改变什么 | 为什么超出首轮 | 进入实验的证据 |
|---|---|---|---|---|
| 1 | [T016 HC 小 M 融合](tricks/T016-hc-small-m-fusion.md) | 小 M 的 HC up-projection + 门控混合；最终 mixer 剩余分离路径 | T005 做序列并行；本轮处理形状阈值与 GPU 中间结果 | 真实 trace 落入未融合小 M 分支，HC 占时可见 |
| 2 | [T017 GDN 固定精度 ReplaySSM](tricks/T017-replayssm-fixed-precision.md) | checkpoint + replay buffer，延迟状态写回/只提交接受链 | T009 扫 MTP 深度；本轮改每次验证的状态数据流 | recurrent state I/O 或 rollback 占时显著，SM90 固定精度移植可行 |
| 3 | [T020 PLE 精确缓存](tricks/T020-ple-exact-cache.md) | owner 内缓存原始行；独立探索静态 `[k,v]` 投影复用 | T006 改放置/预取，T008 复用全前缀；本轮减少局部查表/投影 | 不同完整 prompt 间仍有 n-gram 重复，且 PLE 等待暴露在关键路径 |

顺序是工程可验证性判断，不是本机收益排序。先复用首轮 profile；某条触发条件不成立，就记录原因并进入下一条，不强迫跑满大矩阵。

### T016：把四路 HC 的小矩阵做成适合 H800 的路径

模型的四路状态宽度为 `4×2560=10240`，bottleneck 为 320。当前代码已经做了部分 norm、down+SiLU、combine，以及 PDL。因此“再开一次 PDL”被排除。

新的可检验点是小 M 上投影/门控的分离路径：先只改已有融合 kernel 的适用阈值，原数学与参数不动；若原 fused kernel 在 H800 的小 M 上不适合，再做小形状 kernel。最终输出 mixer 的 down→SiLU 边另立变体，不把两项同时改。

这条最适合先从单层捕获的真实 tensor 开始，改动边界清楚。重点看完整 HC 链的时间、中间张量写回和端到端 TPOT，不能只测其中一次 GEMM。

### T017：让 36 层 GDN 少搬完整状态

按 config 的 FP32 recurrent matrix 估算，36 层的聚合状态为 **108 MiB/request**，未含其他历史与投机副本。标准读写若都落到 HBM，单步主项约 216 MiB/request；实际分片、缓存命中和 kernel 行为另测。

ReplaySSM 的可借鉴部分是“保留 checkpoint 与短输入记录，延迟写回、拒绝时回退记录指针”。本轮不采用其状态降精度部分。现成 Qwen4Exp FlashInfer PR 要求 SM100 和 FP16 state；H800 需要沿 Triton/SM90 路线接线，并保持原 state、correction、buffer 和累加精度。

因此这是三项中工程量最大的主线。先在真实 GDN shape 做 teacher forcing 和接受链状态校验，再接完整 QSA/PLE/卷积生命周期，最后测服务吞吐。原论文的 B300/其他模型数字不作本机预测。

### T020：利用 PLE 静态路径，不重复做已知查表

PLE 的 row IDs 由局部 token 历史产生，即使两个请求没有相同长前缀，也可能命中同一行。第一版缓存每个 owner 自己负责的原始权重行，miss 仍走当前 pinned-host/UVA 路径，预取和 ETP 归并保持原状。

第二版更激进：在源码中 `embedding → dequantize → kv_proj` 只依赖精确 IDs 和权重，可缓存其 `[k,v]` 输出；后面的 hidden-dependent gate 和 short-conv 仍正常运行。此版有更大的 cache entry 和 miss 小 GEMM 风险，必须独立对照。

这条是源码导出的研究假设。没有真实 n-gram 命中与暴露等待，就没有收益承诺。先做低成本 trace 回放，避免为一个已被完全隐藏的 lookup 花 GPU 内存。

## 3. 已调研，但不列入本批三项的方向

| 卡片/方向 | 当前结论 | 模型相关原因 |
|---|---|---|
| [T014 LM head + exact sampling](tricks/T014-fused-lm-head-sampling.md) | 储备，`conditional` | 248320 大词表确实相关，但 target/MTP 共享 head，现有验证接口接收 logits；原本纯 greedy 的每位置 argmax 可另作接口改造，不能取消实际采样/logprobs 要求 |
| [T015 adaptive verification](tricks/T015-adaptive-verification.md) | 当前 vLLM 路径 `blocked`，需适配 | 通用 MTP 文档不够；PLE、QSA、indexer 的 device query-length / varlen Graph 能力检查有缺口 |
| [T018 tile collective fusion](tricks/T018-tile-collective-fusion.md) | 储备，`conditional` | 可先针对 BF16 QSA output projection 的实际 collective；MoE FP8 block128 不能照搬 BF16/MXFP8 示例 |
| [T019 bounded EP workspace](tricks/T019-bounded-ep-workspace.md) | 储备，`conditional` | 当前部分 DeepEP 路径已有容量分桶和 rank 去重，必须先找仍存在的高水位 workspace |
| 整模型 megakernel / A-F disaggregation | 暂缓 | 缺 Qwen4Exp + H800 的直接完整实现，部分最快路径依赖 SM100，且可能损害小 batch |
| TreeWY 小树验证 | 暂缓 | QSA compression ring、PLE n-gram 与 GDN 状态都需分支历史；增加接受长度不保证净吞吐 |
| CacheBlend / 跨层 index 直接复用 | 不进入主线 | 前者当前专用 recipe 不支持；后者尚无保持本模型 QSA 行为的精确实现 |

上游最新实现的“已包含”只说明不要重复写同一功能。用户本机是否已包含，必须结合首轮实际 commit 判断。

## 4. 收益模型：只预测首轮之后的增量

令 `T_W1` 为当前最佳有效配置的同一段 decode 时间，`f` 为待改路径在关键路径中的比例，`s` 为同形状局部加速，`ε` 为新增成本占 `T_W1` 的比例。本项目的简化估算：

$$
S_{\mathrm{incremental}}=\frac{1}{1-f+f/s+\varepsilon}.
$$

它假定其他部分不变且没有重复计算 overlap 的收益；服务排队与动态 batch 要另做实测。

| 假设情景，仅用于理解上限 | 路径占比 f | 局部速度 s | 零新增成本时整体速度 |
|---|---:|---:|---:|
| 较小局部瓶颈 | 0.10 | 2 | 1.053× |
| 明显局部瓶颈 | 0.30 | 2 | 1.176× |
| 主导局部瓶颈 | 0.50 | 2 | 1.333× |

这三行是算例，不是 H800 预测值。

GDN 的写回理论先用 payload 与 replay-buffer 流量估局部上限；PLE 用成本加权命中和实际暴露等待；HC 用真实小 M 占比。三条都受工程新增成本约束，不能把各论文倍数或三个局部收益直接相乘。进一步计算沿用 [现有理论预测器](../../theory/predictor/README.md)，缺参数保留 unknown，不填造假的实测行。

## 5. 本轮的验证重点

采用现有 [Benchmark 协议](../../docs/benchmark-protocol.md)，但基线更新为 **首轮已有最佳有效组合**。不重做初始 13 项。保持 checkpoint、所有实际 dtype、路由 top-k、采样、thinking、任务要求、输出规则与总 H800 数量。

每项先单模块、再完整路径、再同 trace 的三次交错重复。报 TTFT/TPOT 分位数、输出/总 token 吞吐、失败与显存。必须把首轮效果、第二轮增量、兼容性适配和不支持的负结论分别记录。

“可以激进”在这里体现在允许改模型专属执行代码、状态表达与缓存路径。仍须保持既定精度、全部专家贡献、采样约束和状态历史。

## 6. 深入更新：将三个大改方向拆成任务卡

用户随后要求聚焦改动最大的方向，并继续调研、写进本周文档。本次在研究仓库 `be556f7` 之上补核 vLLM `aefea5c8e00c47032cda6d0b75704f6cd0a7c66d` 与相关 PR head；当前入口改为 [R01–R03](tasks/README.md)。

| 方向 | 相比初筛新增的实质证据 | 当前决定 |
|---|---|---|
| GDN / MTP | PR 正文的 host cursor 已落后于 GPU planner 代码；每 cache group 游标只推进一次。ring dtype 外还需审计 tf32x3，target/draft 配置与接受游标单独提交 | [R01](tasks/R01-gdn-state-mtp.md)为主线；单层→生命周期→固定 MTP，adaptive 后置且仍 blocked |
| PLE | 真实文本 host cache 有公开负结果；行命中不等于完整 tuple 命中。实际模块是 layers.1.ple；early probe、舍入与 Graph miss-count 有精确入口 | [R02](tasks/R02-ple-memoization.md)先做可否决的 trace/预算分析，不预设收益 |
| MoE | Qwen4Exp 不能直接启用通用 DBO。SM90 block-FP8 MegaMoE 也可能重缩放 FC1；原 Triton 则存在完整 prepare 后只切专家计算的候选入口 | [R03](tasks/R03-moe-pipeline.md)为条件储备；先内部串行 token 分段，再判断通信分段 |

依据见[任务卡固定源码引用](tasks/README.md)和[来源增补](sources-round-02.md)。普通 decode 的 replay 不保证省显存，PLE 高行命中不保证能跳投影，MoE 单层缩 workspace 也不保证释放进程高水位；三卡都要求记录新增成本和完整模型证据。

首轮最佳配置不变。HC 小 M 的 T016 仍可作短期独立实验；本次按大改方向重排研究资源，没有把旧候选判为失败，没有新增 H800 实测。各阶段列出唯一变量、最小 commit、交付、通过与退出、回滚。


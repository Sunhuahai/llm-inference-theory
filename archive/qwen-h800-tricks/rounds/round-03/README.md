# Round 03 · 10 个新 trick 与参考 skill

更新：**2026-11-20（Asia/Shanghai）**。日期索引仍是 [2026-W47](../../weeks/2026-W47/README.md)，实验批次已推进到 **Round 03**。

**T001–T013 已按用户反馈全部移入 [Round 01 归档](../../archive/round-01/README.md)。本轮新增 T021–T030，共 10 项。** T014–T020 与 R01–R03 保留为第二轮研究；用户没有报告它们全部尝试，不替用户补结论。

这 10 项与旧卡按“改哪个执行对象、数据流或协议”排重，不把现成开关换名字计数。全部是针对 Qwen3.8-Flash-Next-FP8 / Qwen4Exp / H800 的研究候选；本次未运行 GPU，收益和可采用性等待实际环境验证。

## 先交给实施模型这两个入口

1. [optimize-qwen-h800 参考 skill](../../skills/optimize-qwen-h800/SKILL.md)：按 S0–S5 推进，一次一个变量，有结构检查脚本、精度合同、验证分支和交付模板。
2. [本轮可复制任务书](agent-tasks.md)：填写实际实施仓库与一个 T 编号，沿用已有最佳有效配置；能力较弱的模型也不需要自己发明实验流程。

## 新增十项

| ID | 新增机制 | 先看什么证据 | 工程范围 | 与旧卡的区别 |
|---|---|---|---|---|
| [T021](tricks/T021-qsa-streaming-exact-topk.md) | QSA 分数计算与精确 top-k 流式融合 | score 矩阵物化、读写与 selection 占比 | 新 H800 分页索引 kernel | T003/T004 是 prepare / 投影，不是 score→selection |
| [T022](tricks/T022-qsa-query-tile-reuse.md) | 多 query 共享选中 KV block 的读取 | selected 集合重叠与 union 密度 | attention 内核与 tile 分组 | 当前 GQA、split-K 已有，新增跨 query 取数复用 |
| [T023](tricks/T023-treewy-branch-verification.md) | 固定树的 GDN/QSA/PLE 分支验证 | 固定精度、祖先状态与 Graph 支持 | 大型状态与 verifier 改造 | T009/T015/T017 处理链深度、预算或 replay；本卡是分支树 |
| [T024](tricks/T024-mtp-history-drafts.md) | 原生 MTP 与历史复制 proposal 组合 | 真实历史命中和正确 proposal 分布 | proposer 与接受路径适配 | 固定 K；新增 proposal 来源，不重复 n-gram 开关 |
| [T025](tricks/T025-router-top10-fusion.md) | Router GEMM→softmax/top-10 执行边界融合 | gate+router 耗时、dispatch 分界 | 路由专用融合 kernel | 当前 fused top-k 已有；不重复 MoE 通信调优 |
| [T026](tricks/T026-shared-expert-mlp-island.md) | BF16 shared expert 的 MLP 内部融合 | 小 M、launch 与中间流量 | FC1→SwiGLU→FC2 内核 | shared stream 已重叠；本卡改分支内部计算 |
| [T027](tricks/T027-full-logits-batch-sharding.md) | 完整 logits 按 token/batch 分配 owner | TP、多 logits 行、NCCL 额外内存 | 模型接口与采样通信 | T014 减少/融合 head 输出；本卡保留完整 logits |
| [T028](tricks/T028-persistent-expert-workqueue.md) | Routed expert GEMM 常驻 CTA 工作队列 | tile 不均、空转与窄 I=640 形状 | 本地专家 GEMM scheduler | 保留路由/EP/scale；不重复 T019 token 分段 |
| [T029](tricks/T029-heterogeneous-state-pages.md) | GDN/PLE 状态页按实际大小打包 | 页尾 padding 的实际容量账本 | 模型 cache 规格与 allocator | 固定 prefix/checkpoint/pool 策略；不重复 T008 |
| [T030](tricks/T030-exact-expert-weight-cache.md) | 原 FP8 权重与 scale 的逐专家驻留缓存 | 每步 unique experts、miss 流量与容量压力 | 专家驻留与 Graph/进程组适配 | 固定专家归属；不是 PLE cache 或 EPLB |

所有卡片都给出锁定 SHA 的源码入口、一手依据、实际前置条件、唯一变量、分步实施、正确性、退出和回滚。T023 的完整模型树路径目前有明确支持阻塞；T030 的现成移植对 EP/DP/SP 等有硬限制，不能直接拿来启动当前多卡服务。

## 优先核对三项，不同时动十项

以下是研究侧的条件顺序，不是未经测量的收益排行榜：

1. **T027：先核接口和通信账本。** 有同款模型的公开修复与完整 logits 正确性对照，入口最明确；只有实际 TP/行数与内存条件满足才做 A/B。
2. **T021：索引 score→top-k 有实质占比时优先。** 新增点与前两轮清楚分离，但 H800 + QSA 分页形状需要专门实现，不能直接套 Blackwell 或其他 indexer 的 kernel。
3. **T029：显存容量或抢占受限时优先。** 先做 CPU 可完成的逐层物理页账本；若没有可消除的 padding，立即结束此卡，不写 allocator。

用户 profile 若明确指出 shared expert 或 router，可以用 T026/T025 替换相应项。T022/T024/T028 是有门槛的内核/算法候选；T023/T030 先完成可否决的可行性阶段。

## 研究与记录

- [研究判断、排重和停止规则](research.md)
- [固定版本与一手来源](sources.md)
- [参考 skill](../../skills/optimize-qwen-h800/SKILL.md) · [任务书](agent-tasks.md)
- [本周反馈](../../weeks/2026-W47/feedback/2026-11-20.md) · [结果入口](../../weeks/2026-W47/results/README.md)

沿用已有最佳有效配置作为本轮 baseline；记录实际 build、checkpoint、dtype、负载与 GPU 总预算。不同日期产生的结果放对应 ISO 周，原 trick ID 和研究卡片保持稳定。

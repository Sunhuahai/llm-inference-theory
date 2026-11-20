# 本周大改方向任务卡

> **批次说明：** 此页保留 Round 02 研究；当前新增十卡与辅助执行 skill 见 [Round 03](../../../rounds/round-03/README.md)。用户仅明确 T001–T013 已尝试，未替本页补完成结论。

更新：2026-11-20（Asia/Shanghai）。研究仓库基线：`be556f74a34d32daab28754183c1a1dff2823c2b`。
用户要求进一步调研，并将三个大改方向写成可交给实施 Agent 的任务卡。R01–R03 是已有 trick 的分阶段实施组合，不新增重复的 T 编号。

## 任务顺序

| 任务卡 | 对应已有研究 | 当前任务 | 状态 / 开发门槛 |
|---|---|---|---|
| [R01 GDN 状态重构与原生 MTP](R01-gdn-state-mtp.md) | [T017](../tricks/T017-replayssm-fixed-precision.md)、[T015](../tricks/T015-adaptive-verification.md) | 主线：先固定精度状态更新，再生命周期和固定深度 MTP | `conditional`；动态验证阶段仍 `blocked` |
| [R02 PLE 静态计算缓存](R02-ple-memoization.md) | [T020](../tricks/T020-ple-exact-cache.md) | 先采真实访问 trace，判断行/完整 tuple 的命中与可节省等待 | `conditional`；通过收益门槛后才开发 cache |
| [R03 MoE 内部分段与流水化](R03-moe-pipeline.md) | [T018](../tricks/T018-tile-collective-fusion.md)、[T019](../tricks/T019-bounded-ep-workspace.md) | 条件储备：先查实际后端、容量和依赖，再选一条边实现 | `conditional`；不盲目启用 DBO 或更换专家 kernel |

三卡均已完成研究设计，**尚无本轮 GPU 实测**。`conditional` 不表示部署环境已核对。实现项目、负责人、运行 commit、H800 数量和各项结果仍从用户现有环境填写，未知保持 `unknown`。

[第二轮总任务书](../agent-tasks-round-02.md)提供可复制的派发提示词；[初筛综述](../research-round-02.md)与[模型审计](../model-audit-round-02.md)保留原研究过程。HC 小 M 的 [T016](../tricks/T016-hc-small-m-fusion.md)仍可作为短期独立实验，不与这三条大改捆成一次对照。

## 本次深入调研改变了什么

### R01：按真实代码移植，不能照抄 PR 参数和旧正文

上游普通 GDN ReplaySSM 与 Qwen4Exp FlashInfer 路线仍有硬件、精度、runner 和状态生命周期差异。尤其是 ring/correction 的 dtype、cursor 推进者、prefix block 边界强制 flush，以及 target/draft 的区别，必须从固定代码核对。

[PR #62004](https://github.com/vllm-project/vllm/pull/62004)的正文与 head 代码存在演进差异；任务卡按源码处理 GPU planner，不把旧 host cursor 描述当实施方法，也不把正文验证范围等同于所有 runner 的硬性拒绝。普通 decode 是诊断入口，最终性能锚点仍为 W1-best；ReplaySSM 和动态验证分开消融。

### R02：加入公开负结果，先证实命中

[上游 RFC #66467](https://github.com/vllm-project/vllm/issues/66467)报告：曾尝试 1 GB direct-mapped host row cache，真实文本 decode 重复行接近零，收益仅见于合成 prompt。该来源的主评测环境为 GB300、TP1、NVFP4 主模型但 FP8 PLE；缓存层与本方案的 H800 owner-HBM cache 不同，真实文本子实验的 trace/样本规模未公开。

这条证据不能证明本机方案无效，却足以撤掉“命中应该很高”的默认假设。R02 首阶段只做独立 trace 与成本分析；**行命中和完整 16-ID tuple 命中分别报告**。后者不能由前者推出，也不能用重复完整 prompt 的 warm-only 测量代替业务效果。

### R03：现成流水参数和同名后端都要逐项核对

目标 Qwen4Exp 配置对 PLE/QSA 与 DBO/ubatching 的组合有限制；R03 研究层调用内部的分段，保留原边界，不以移除能力检查代替实现。

[已合并 PR #62603](https://github.com/vllm-project/vllm/pull/62603)提供 SM90 block-FP8 MegaMoE，说明“所有 MegaMoE 都只支持 SM100/MXFP8”并不成立。但该适配会为 FC1 gate/up 交错重设共同 scale、重缩放 FP8 行，可能产生额外舍入，不能直接作为本仓库固定原权重/scale 的方案。它的 Qwen 专家几何测试也不等于目标模型 H800 服务实测。可借鉴的是全 rank 一致的容量桶、idle-rank 参与和明确的 workspace 生命周期。

## 共用实验合同

- **比较锚点**：首轮当前最佳有效配置 W1-best；模块微基准仅用于解释机制。需要换引擎/runner/backend时先建立独立对照，不混入目标改造收益。
- **唯一变量**：每个阶段只改变一项。普通 decode、固定深度 MTP、动态验证、缓存层次、内部 chunk、外部 batch 和状态池扩大分别比较。
- **固定边界**：同 checkpoint 与所有实际 dtype/scale/累加精度，保持原 top-10、采样、reasoning、EOS/输出规则、任务质量和总 GPU 数。
- **少量测点**：复用现有 profile；一个主要场景加一个针对性回归场景即可初筛。只有解释具体风险或满足原协议时扩展。
- **真实成本**：计入 probe/规划、ring、flush、miss compaction、通信争用、graph 和 cache 占用；只把关键路径上实际减少的等待计为收益。
- **证据**：沿用[Benchmark 协议](../../../docs/benchmark-protocol.md)与[结果表头](../../../templates/result.csv)，至少三次交错重复；状态和证据来源分开记录。

三项可能竞争同一 HBM 容量及带宽。先单项、再组合，不把局部倍数相乘。若某路径占当前关键路径的比例为 f，即使该路径完全消失，其他部分不变时的理想整体上界也只有 `1/(1-f)`；这是筛选上限，排队与实际调度需另测。

## 每个阶段必须交付

1. `baseline_run_id`、`candidate_run_id`、实现 commit、独立开关和完整配置差异；拟议开关明确标注为自定义。
2. 模型 revision、引擎/依赖 SHA、真实进程组、张量 shape、dtype/scale 清单及实际 dispatch。
3. 针对改造机制的正确性证据与结果；不能只附随机生成样例。
4. 完整请求的 TTFT/TPOT、输出/总 token 吞吐、错误率、显存与机制 profile；尚未运行写 `unknown`。
5. 保留/退出决定及回滚方法；无收益和不支持也归档，不删除原负结果。

状态使用仓库已有词表。缺环境或实现时不标 `ready/validated`；正确性失败停止该分支；无可见空间、成本抵消或结果落在波动内时分别记录原因。未获运行资源时交付可审核的适配设计，不启动未授权作业。

来源锁定与本次增补见[来源索引第 5 节](../sources-round-02.md#5-大改任务卡的增补证据2026-11-20)。


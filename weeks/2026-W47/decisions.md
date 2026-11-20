# 决策记录

## 2026-11-16 · 初始化

- **已知**：用户要求公开仓库、每周文件夹、每晚反馈驱动研究，其他 Agent 实施；固定现有 8-bit 制品，关注 TTFT/TPOT/吞吐。
- **模型判断**：官方名称匹配 Qwen/Qwen3.8-Flash-Next-FP8，本机制品待核对。
- **决策**：建立 12 个候选和 P0→P1→P2 任务顺序；优先可解释的单变量实验。
- **证据**：[来源索引](sources.md)；目前没有用户 H800 日志。
- **状态**：调研完成；无 validated trick；环境/基线待补。
- **下一步**：P0 完成后选择 T001/T002 或 profile 指向的模型专属卡片；没有反馈时不盲目增加清单。

后续按日期追加：反馈来源 → 可比性 → 观察 → 决策 → 下个最小实验 → 证据链接。性能好坏都保留。

## 2026-11-17 · Prefix / 上下文天花板研究（未收到新实测）

- **来源**：用户要求建立预测模型，特别研究 KV/Prefix cache 命中率、上下文压缩与优化空间；一手上游文档检索，没有用户 H800 trace 或 GPU 测量。
- **判断**：请求命中率、token 命中率、可节省 Prefill 的成本加权命中率必须分开；Qwen3.8-Flash-Next 是 QSA+GDN 混合状态，不能把传统纯 KV 复用模型照搬。
- **新增**：[理论天花板与实验任务](../../theory/cache-context-ceilings.md)、按序精确键 LRU 情景预测器及合成算例、新候选 [T013](tricks/T013-context-compression.md)；T008 补充版本与 state caveats。**合成数字全部非 H800 实测**。
- **优先级**：P0 环境与真实性能基线不变；只有业务 trace 显示重复前缀或可去除冗余，才将 T008/T013 放入下一轮，不为了增加 trick 而启动 GPU 作业。
- **最小下一步**：①脱敏 trace / 缓存状态预算 / 冷热 Prefill；②固定原问题集合的压缩配对质量样本；③按原协议重放单变量 A/B，再校准饱和吞吐。

## 2026-11-20 · 首轮已尝试，转向具体模型的第二轮改造

- **反馈来源**：[user_report](feedback/2026-11-20.md)。用户明确 Week 1 全部尝试，测试允许激进，并要求针对这个具体模型。逐项效果未知；不批量标 validated，不再安排从 T001/T002 起步。
- **证据基点**：研究仓库 main `eab2d2fdce3f22ee596c9b11ce6e50845fa16de8`；vLLM `12924016eb8dbaae87d5163a1b7ec2614a44a797`；[来源与版本](sources-round-02.md)。本机 engine/拓扑/dtype 仍从已有运行记录提取。
- **新结论**：HC 已有 PDL，但 up+gate_mix 的小 M 分派留下具体入口；36 层 GDN 状态有可推导的大读写主项；PLE 可缓存的静态对象与动态 gate/conv 边界明确。T015 则被 PLE、QSA/indexer 能力检查阻塞，通用 MTP 文档不足以证明可用。
- **决策**：本批只推荐 [T016](tricks/T016-hc-small-m-fusion.md)、[T017](tricks/T017-replayssm-fixed-precision.md)、[T020](tricks/T020-ple-exact-cache.md)。T014/T018/T019 保留条件储备；T015 记当前上游路径 blocked；未核验完整目标实现的整模型 megakernel、A/F 分离和树验证暂缓。
- **精度边界**：ReplaySSM 现成 Qwen4Exp PR 的 SM100/FP16 路径不适用 H800 固定精度；FP32-state 论文数字里的 FP16 corrections 不直接采用。MegaMoE block-FP8→MXFP8 转换排除。激进范围是执行代码、状态数据流与精确缓存。
- **最小下一步**：复用 W1-best 和已有 profile，按 HC 小 M / GDN 状态 / PLE 等待的触发条件选项；每项独立 A/B，缺触发条件则跳到下一项。[完整任务书](agent-tasks-round-02.md)已给出实现入口、正确性、机制指标与退出条件。
- **状态**：第二轮研究完成；无本轮 agent_measured H800 结果，未启动 GPU 作业。按实际日期继续归档 W47，不因“第二轮”提前建 W48。

## 2026-11-20 · 深入三个大改方向，落地阶段任务卡

- **用户要求**：选择改动最大的方向，再进一步调研并写入本周文档。范围为 GDN 状态重构＋MTP、PLE 静态计算缓存、MoE 内部分段与流水化。
- **基点**：研究仓库 `be556f74a34d32daab28754183c1a1dff2823c2b`；上游 `aefea5c8e00c47032cda6d0b75704f6cd0a7c66d`，PR head 与初筛分开锁定。
- **新证据**：GPU planner 与 PR 正文存在演进差异，ring/math-mode 和接受游标要一起核对；PLE 有外部真实文本低复用负结果；MoE 的 DBO/尾行/Graph 高水位和 FC1 scale 转换均有具体限制。[来源增补](sources-round-02.md)。
- **决策**：新增 [R01](tasks/R01-gdn-state-mtp.md)、[R02](tasks/R02-ple-memoization.md)、[R03](tasks/R03-moe-pipeline.md)，不新增重复 T 编号。R01 主线分阶段，adaptive 仍 blocked；R02 先过 trace/预算门槛才开发；R03 条件储备，先只分段原专家计算。
- **原候选处理**：T016 保留短期独立实验；T017/T020/T018/T019 增加任务入口，保留初筛证据。用户完成首轮的报告不变，没有逐项收益就不改 validated 状态。
- **交付与状态**：更新总任务书、周目录、首页、来源和反馈。设计完成，未启动 GPU；实施 commit/质量/收益均 unknown，每阶段有唯一变量、交付、通过与退出、回滚。


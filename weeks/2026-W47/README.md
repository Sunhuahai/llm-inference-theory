# 2026-W47 · 首轮反馈与 Qwen4Exp 第二轮专项

日期：**2026-11-16—2026-11-22**（Asia/Shanghai）。首轮调研：2026-11-16；第二轮更新：2026-11-20。

**用户已反馈首轮全部尝试，当前进入第二轮。** 逐项成功/失败、运行 commit 和原始指标尚未归档；本轮研究没有运行 H800 实验。实际部署与首轮最佳配置由已有运行记录补齐，不从官方示例猜测。

## 当前派发：三个大改方向

| 顺序 | 卡片 | 状态 | 触发条件 |
|---|---|---|---|
| 1 | [R01 GDN 状态重构与原生 MTP](tasks/R01-gdn-state-mtp.md) | conditional | 保持原 state/ring/math 精度，按单层、生命周期、固定 MTP 分阶段 |
| 2 | [R02 PLE 静态计算缓存](tasks/R02-ple-memoization.md) | conditional | 先过真实 trace 的命中/等待/容量门槛，再写 cache |
| 3 | [R03 MoE 内部分段与流水化](tasks/R03-moe-pipeline.md) | conditional | 实际后端和高水位存在空间，原路由、kernel、scale 合同可保持 |

直接交付：[三张任务卡与新增发现](tasks/README.md)、[第二轮总任务书](agent-tasks-round-02.md)。判断依据：[第二轮研究](research-round-02.md)、[具体模型源码审计](model-audit-round-02.md)、[固定版本来源](sources-round-02.md)。

R01 是主研究线；R02 当前先做可否决的 trace 分析；R03 是条件储备。HC 小 M 的 T016 保留为短期独立实验。当前任务卡落实已有 trick，不重新启动 Week 1。

| 储备 | 状态 | 为什么不列入本批 |
|---|---|---|
| [T016 HC 小 M 融合](tricks/T016-hc-small-m-fusion.md) | conditional | 初筛优先的小改造，仍可独立快速验证；不与三条大改合并计收益 |
| [T014 head + sampling](tricks/T014-fused-lm-head-sampling.md) | conditional | 必须保留目标 MTP 验证所需 logits；部分 draft 接口可独立适配 |
| [T015 adaptive verification](tricks/T015-adaptive-verification.md) | blocked | 当前审计 vLLM 的 PLE 与 QSA/indexer 有后端能力缺口 |
| [T018 tile collective fusion](tricks/T018-tile-collective-fusion.md) | conditional | 已整理进 R03；原 kernel 无 tile-ready 接口时先研究 chunk 完成边 |
| [T019 bounded expert workspace](tricks/T019-bounded-ep-workspace.md) | conditional | 已整理进 R03；真实释放高水位是开发门槛，不重复已有分桶 |

所有第二轮收益都相对首轮当前最佳有效配置；不把上游基准当成本机结果。第二轮仍发生在 W47，未提前创建 W48。

## 首轮任务历史（保留，已尝试）

1. **P0 环境与基线**：确认模型、H800/拓扑、实际 dtype、版本，查已有融合路径；产出可比较的 baseline。
2. **P1 调度与 Graph**：依据 baseline 分别做 T001 或 T002 单变量试验；优先短/长请求混合时的尾延迟。
3. **P2 模型专属瓶颈**：若 QSA/GR/PLE 耗时明显，优先 T003–T006；业务重复前缀多则优先 T008。先有证据再选择，不一次性改全套。

**新增研究入口（2026-11-17）**：[前缀缓存命中、有限状态容量与上下文压缩的收益上界](../../theory/cache-context-ceilings.md)，附可运行合成回放与质量 gate；T008/T013 的真实收益均待实测。仅在真实长前缀重复或业务冗余充分时提升它们的优先级。

历史任务入口：[首轮任务书](agent-tasks.md)。历史研究判断：[research.md](research.md)。以上初始次序不再作为默认下一步。

## 首轮登记快照

下表状态是初始研究时的登记；整体尝试完成另以 `user_report` 记录。未得到逐项可复查结果前，不臆造成功/失败，不把研究状态当作仍未执行。

| ID | 卡片 | 状态 | 决定先后顺序的条件 |
|---|---|---|---|
| T001 | [Prefill 调度](tricks/T001-prefill-scheduling.md) | researched | 混合请求或 TTFT/TPOT 冲突 |
| T002 | [CUDA Graph](tricks/T002-cuda-graphs.md) | researched | launch/CPU 间隙、短 batch decode |
| T003 | [QSA prepare 融合](tricks/T003-qsa-prepare.md) | researched | 先查当前 build 是否覆盖 |
| T004 | [QSA 投影融合](tricks/T004-qsa-projection.md) | researched | 小 GEMM 与实际 SM90/TP 形状 |
| T005 | [GR/PLE sequence parallel](tricks/T005-gr-ple-sequence-parallel.md) | researched | 支持的多卡路径，长 prefill |
| T006 | [PLE 预取与 offload](tricks/T006-ple-prefetch.md) | researched | host 取表/PCIe/NUMA 等待 |
| T007 | [并行与拓扑](tricks/T007-parallel-topology.md) | conditional | 合法多卡配置，固定总预算 |
| T008 | [混合前缀缓存](tricks/T008-hybrid-prefix-cache.md) | researched | 多轮 continuation 或前缀分叉 |
| T009 | [原生 MTP](tricks/T009-native-mtp.md) | researched | 接受率、验证开销与质量门槛 |
| T010 | [MoE 通信](tricks/T010-moe-communication.md) | conditional | 专家不均衡或 collectives 占比高 |
| T011 | [Host runtime](tricks/T011-host-runtime.md) | researched | CPU/分词/主机调度瓶颈 |
| T012 | [Prefill/decode 分离](tricks/T012-prefill-decode-disaggregation.md) | conditional | 多机规模、真实流量、混合状态传输支持 |
| T013 | [上下文压缩](tricks/T013-context-compression.md) | conditional | 业务确有冗余、压缩成本与严格配对质量门槛 |

## 记录入口

- [反馈说明](feedback/README.md)
- [首轮已尝试与本轮范围反馈](feedback/2026-11-20.md)
- [结果说明](results/README.md)
- [决策记录](decisions.md)
- [一手来源索引](sources.md)
- [第二轮来源与版本](sources-round-02.md)

首周无 carry-over。每晚更新只补新增反馈和明确下一步；尚未运行的候选不写收益数字。

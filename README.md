# Qwen3.8-Flash-Next · H800 推理优化笔记

围绕 **固定 8-bit checkpoint** 的 **TTFT、TPOT、输出吞吐量与总 token 吞吐量**，每周积累可验证、可交给实施 Agent 的优化方案。

目标模型暂按官方 [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) 建档。用户机器上的实际模型 ID、revision、H800 数量及推理框架仍待登记；不能把该模型替换成旧 `Qwen3-Next-80B-A3B`。不改变现有权重、激活、KV/indexer cache、SSM state 或 embedding 精度。

**当前状态（2026-11-20）：用户已反馈 Week 1 全部尝试；第二轮已进一步整理为三张大改方向任务卡。** 主线为 GDN 状态重构，PLE 先过真实 trace 收益门槛，MoE 内部分段与流水化为条件储备。首轮逐项性能结果尚未归档，本轮没有启动 GPU 实验。

## 从这里开始

| 目的 | 入口 |
|---|---|
| 看本周结论、优先级和进度 | [2026-W47 周目录](weeks/2026-W47/README.md) |
| 看当前三张大改方向任务卡 | [GDN / PLE / MoE 任务索引](weeks/2026-W47/tasks/README.md) |
| 回看第二轮初筛与深入增补 | [模型专属研究](weeks/2026-W47/research-round-02.md) |
| 直接交给其他 Agent 执行 | [第二轮任务书](weeks/2026-W47/agent-tasks-round-02.md) |
| 查看本模型的真实源码边界 | [Qwen4Exp 固定版本审计](weeks/2026-W47/model-audit-round-02.md) |
| 了解模型与支持边界 | [模型核对](docs/model-and-support.md) |
| 统一性能测量方法 | [Benchmark 协议](docs/benchmark-protocol.md) |
| 在实施前预测优化收益与上限 | [理论预测方法](theory/optimization-benefit-prediction.md) |
| 研究 Prefix/KV 命中率、上下文压缩与理论天花板 | [专项推导与可运行 trace 模型](theory/cache-context-ceilings.md) |
| 查看三页高密度优化综述 | [可编辑 draw.io 与 SVG](theory/diagrams/README.md) |
| 用模型预测 TTFT / TPOT / 吞吐 | [可运行预测器、情景与测试](theory/predictor/README.md) |
| 填写机器与基线 | [环境清单](environment.yaml) |
| 每晚反馈 | [简短反馈模板](templates/feedback.md) |
| 了解每天、每周怎样更新 | [反馈闭环](docs/feedback-loop.md) |

## 分工

- **研究 Agent**：查一手来源、解释机制、审核适用条件，写最小对照实验；根据反馈决定下一步。
- **实施 Agent**：在用户指定的推理项目里实现、测试和回滚；提交配置差异、代码 commit、真实结果。
- **用户**：每晚反馈效果、现象和预算；不要求先整理完整报告，缺失信息保留 `unknown`。

本仓库存研究文档、任务书、脱敏结果和决策记录。实施代码所在仓库由执行时指定，不在此预造一套推理框架。

## 当前优先：三个大改方向

沿用首轮最佳有效配置和实际精度，只衡量新增收益；不从最初调度与 Graph 开关重做。

| 顺序 | 方向 | 本轮真正新增的内容 |
|---|---|---|
| 1 | [R01 GDN 状态重构与 MTP](weeks/2026-W47/tasks/R01-gdn-state-mtp.md) | 固定精度单层 → 状态生命周期 → 固定深度 MTP；动态验证后置 |
| 2 | [R02 PLE 静态计算缓存](weeks/2026-W47/tasks/R02-ple-memoization.md) | 先用真实 trace 核验行/tuple 命中与收益；通过才实现缓存 |
| 3 | [R03 MoE 内部分段与流水化](weeks/2026-W47/tasks/R03-moe-pipeline.md) | 条件储备；固定原 kernel、scale 和外层调用，先核实际工作区与后端 |

R01–R03 将已有 T017/T015、T020、T018/T019 拆成实施阶段，不重复创建 trick。三卡均为 `conditional`；R01 动态验证子阶段仍 `blocked`。HC 小 M 的 [T016](weeks/2026-W47/tricks/T016-hc-small-m-fusion.md)保留为短期独立实验。新核查的负结果、精度陷阱和来源见[任务索引](weeks/2026-W47/tasks/README.md)与[来源增补](weeks/2026-W47/sources-round-02.md)。

## 首轮候选历史

下表保留 2026-11-16/17 的初始研究顺序，**不是当前待执行清单**。[11 月 20 日反馈](weeks/2026-W47/feedback/2026-11-20.md)说明用户已全部尝试；逐项实施与性能证据补到对应卡片，不批量标为 `validated`。

| ID | 方向 | 重点指标 | 初始顺序 |
|---|---|---|---|
| T001 | Chunked prefill 与 batch token 预算 | TTFT / TPOT / 吞吐权衡 | 先试 |
| T002 | CUDA Graph 模式、覆盖形状和预热 | TPOT、短请求 TTFT | 先试 |
| T003 | QSA prepare 融合 | TPOT、launch 开销 | 先核对是否已有 |
| T004 | QSA 投影融合与 SM90 小 GEMM | TPOT、输出吞吐量 | 先核对是否已有 |
| T005 | GR/PLE sequence parallel | 长输入 TTFT、吞吐量 | 条件满足后试 |
| T006 | PLE offload、异步预取与 NUMA | TTFT / TPOT 尾延迟 | 根据 profile 优先 |
| T007 | 合法 TP / EP / DP 与 H800 拓扑 | 延迟、单位卡吞吐 | 多卡条件满足后试 |
| T008 | 混合模型 prefix checkpoints / 状态池 | 多轮 TTFT、goodput | 业务有前缀复用时试 |
| T009 | 原生 MTP 深度与负载 | TPOT、输出吞吐量 | 正确性门槛后试 |
| T010 | MoE 通信与专家负载 | TPOT、吞吐量 | 通信占比高时试 |
| T011 | CPU、分词与运行时调度 | 短请求 TTFT、尾延迟 | CPU 瓶颈时试 |
| T012 | Prefill / decode 分离 | 混合负载尾延迟 | 后置条件候选 |
| T013 | 上下文结构去重 / 查询相关压缩 | 长输入 TTFT、质量与缓存交互 | 需要业务质量基准后试 |

机制与依据见各卡片；该排序是本仓库的研究判断，不是性能排行榜。实际基线已包含的功能记为 `baseline-covered`，不重复实现。

## 指标约定

TTFT 和 TPOT 报 p50/p95/p99；吞吐量同时报输出 tokens/s 与总输入+输出 tokens/s，并记录 GPU 预算、请求失败和实际输出长度。有业务 SLO 后增加 goodput。先做单变量 A/B，收益超过重复波动且质量通过，才讨论组合。完整公式与 streaming 多 token chunk 的处理见 [Benchmark 协议](docs/benchmark-protocol.md)。

## 按周组织

使用 **Asia/Shanghai 的 ISO 周**：周一到周日。首周为 `weeks/2026-W47/`（2026-11-16—2026-11-22）。新周第一次维护时创建目录，保留上一周证据；跨周继续的 trick 沿用原 ID 并链接旧卡片，避免复制成“新发现”。

每周有 `README.md`、`research.md`、`agent-tasks.md`、`feedback/`、`results/`、`decisions.md` 和新候选的 `tricks/`。所有实测结论必须能追溯到结果文件或实施 commit。

维护规范见 [AGENTS.md](AGENTS.md)。

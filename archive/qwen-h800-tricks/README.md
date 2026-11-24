# Qwen3.8-Flash-Next · H800 推理优化笔记

围绕固定模型与全部实际精度，研究 **TTFT、TPOT、输出吞吐量与总 token 吞吐量**，把可核对的方案交给实施 Agent。

**当前：Round 03（2026-11-20）。T001–T013 已按用户反馈归档，新增 T021–T030 共 10 项，并提供逐阶段执行的参考 skill。** 日期仍在 2026-W47；实验轮次独立递增。归档表示“用户已尝试”，不表示全部失败或已验证成功。

目标研究对象为 [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) / `Qwen4Exp` / **H800**。实际部署身份、revision、引擎、卡数和 dtype 以用户已有运行记录为准；[环境清单](environment.yaml) 的未知项不从官方示例补猜。本轮仅完成研究与任务设计，没有运行 H800 实验。

## 当前入口

| 目的 | 入口 |
|---|---|
| 看本轮十个新增 trick、条件和优先顺序 | [Round 03](rounds/round-03/README.md) |
| 让实施模型按步骤工作 | [optimize-qwen-h800 参考 skill](skills/optimize-qwen-h800/SKILL.md) |
| 直接复制交接任务 | [Round 03 任务书](rounds/round-03/agent-tasks.md) |
| 查排重、工程边界与负证据 | [研究判断](rounds/round-03/research.md) · [固定来源](rounds/round-03/sources.md) |
| 查首轮 T001–T013 原文 | [Round 01 归档](archive/round-01/README.md) |
| 继续第二轮 T014–T020 / R01–R03 | [第二轮研究](weeks/2026-W47/research-round-02.md) · [三张大改任务卡](weeks/2026-W47/tasks/README.md) |
| 查本周反馈、结果和决策 | [2026-W47 日期目录](weeks/2026-W47/README.md) |
| 核对模型与数值边界 | [模型与支持](docs/model-and-support.md) · [第二轮源码审计](weeks/2026-W47/model-audit-round-02.md) |
| 按统一口径测量 | [Benchmark 协议](docs/benchmark-protocol.md) |
| 学习推理优化理论与实践方向 | [理论框架图解导读](theory/inference-optimization-framework.md) · [完整理论总纲](theory/llm-inference-optimization-theory.md) |
| 调研 AI 自动优化推理 | [Agentic Inference Optimization 专题](theory/ai-driven-inference-optimization.md) |
| 估算收益与天花板 | [优化预测方法](theory/optimization-benefit-prediction.md) · [可运行预测器](theory/predictor/README.md) |
| 研究 prefix/上下文上界 | [专项理论与 trace 模型](theory/cache-context-ceilings.md) |
| 看优化结构图 | [draw.io 与 SVG](theory/diagrams/README.md) |
| 提交反馈与维护 | [反馈模板](templates/feedback.md) · [反馈闭环](docs/feedback-loop.md) |

## Round 03：先核对三个可否决的入口

1. **T027 完整 logits 分片**：核对本模型接口、实际 TP/采样行数与 NCCL 额外内存；有明确公开修复入口，收益仍取决于负载。
2. **T021 QSA score→精确 top-k 融合**：只在索引打分/选择是实际瓶颈时实现 H800 分页 kernel，保持原索引语义。
3. **T029 状态页物理打包**：先用 CPU 做 raw/padded 页账本，确认总池容量会下降，再动 allocator。

这三个是条件顺序，不是承诺收益。其余七项和逐卡触发条件见 [完整十卡表](rounds/round-03/README.md)。首轮最佳有效配置继续作基线；不从最初调度/Graph 开关重新试起。第二轮未被用户报告为全部完成，原研究继续保留。

## 参考 skill 怎样使用

让执行模型先读 `skills/optimize-qwen-h800/SKILL.md`，再指定一张任务卡和实际推理仓库。Skill 提供 S0–S5 固定流程、条件分支、精度与混合状态合同、计划模板、结构检查脚本和证据交付格式。支持显式 skill 调用的工具可用 `$optimize-qwen-h800`；其余工具直接读取文件。

[任务书](rounds/round-03/agent-tasks.md) 已给出完整可复制提示词。每次只实施一个变量，缺少前提时完成可做的分析并准确报告阻塞。

## 分工与证据

- **研究 Agent**：核对一手来源和当前代码，解释机制、适用条件、负证据，写任务卡。
- **实施 Agent**：在用户指定的推理仓库实现、验证、测试与回滚，提交真实代码和运行证据。
- **用户反馈**：可先给现象；缺失字段保留 unknown，不要求先整理完整报告。

本仓库存研究、任务书、参考 skill 与脱敏结果。状态与来源分开记录；`user_report / attempted` 不自动升级为 `validated`。具体度量遵循统一协议：TTFT/TPOT 报尾分位，吞吐附实际输出长度、错误与总 GPU 预算，原始证据可定位。

## 实验批次与日期

- `rounds/round-03/` 表示第三批研究，不依赖周号递增。
- `archive/round-01/` 保存已尝试的首轮内容；旧链接保留跳转。
- `weeks/YYYY-Www/` 按 **Asia/Shanghai 的 ISO 周**保存日期、反馈、结果和决策。2026-11-20 属于 2026-W47（11 月 16–22 日）。

跨周继续同一 trick 时沿用 ID；不复制成“新发现”。维护规范见 [AGENTS.md](AGENTS.md)。

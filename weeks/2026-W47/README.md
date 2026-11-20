# 2026-W47 · 日期索引 / 当前 Round 03

日期：**2026-11-16—2026-11-22（Asia/Shanghai）**。最新更新：2026-11-20。

**实验已推进到 Round 03；W47 只是日期目录。** 用户明确 T001–T013 均已尝试，已移入 [Round 01 归档](../../archive/round-01/README.md)。本轮新增 10 个独立 trick（T021–T030）与一个辅助实施模型的参考 skill；本次未运行 GPU。

## 当前交付

| 内容 | 入口 |
|---|---|
| 10 个新 trick 与优先条件 | [Round 03 十卡目录](../../rounds/round-03/README.md) |
| 逐阶段执行参考 skill | [optimize-qwen-h800](../../skills/optimize-qwen-h800/SKILL.md) |
| 可直接复制的任务书 | [Round 03 任务书](../../rounds/round-03/agent-tasks.md) |
| 本轮排重、机制、反证 | [研究判断](../../rounds/round-03/research.md) |
| 锁定版本与一手来源 | [来源索引](../../rounds/round-03/sources.md) |

先核对 T027 的接口/通信、T021 的 score→top-k 瓶颈、T029 的物理 padding；每次由 skill 带着模型执行一张卡的一个阶段。全部是条件候选，不将上游数据当作用户 H800 收益。

## 历史批次

| 批次 | 内容 | 当前含义 |
|---|---|---|
| Round 01 | [T001–T013 归档](../../archive/round-01/README.md) | user_report / attempted；逐项结论 unknown，历史完整保留 |
| Round 02 初筛 | [T014–T020](research-round-02.md) | 新机制研究，用户未报告全部尝试 |
| Round 02 深入 | [R01–R03](tasks/README.md) · [任务书](agent-tasks-round-02.md) | GDN 状态、PLE 静态计算、MoE 分段三条既有工作流 |
| Round 03 | [T021–T030](../../rounds/round-03/README.md) | 当前新增十卡与执行 skill |

[首轮研究跳转](research.md) 与 [首轮任务书跳转](agent-tasks.md) 保留旧链接。第二轮的 [模型审计](model-audit-round-02.md)、[来源](sources-round-02.md) 仍保留各自 SHA 和判断日期，不能把旧版本支持结论无条件套在新 build 上。

## 本周记录

- [2026-11-20 反馈](feedback/2026-11-20.md)：含明确归档范围与参考 skill 要求。
- [决策记录](decisions.md)：按时间追加，旧决策保留。
- [结果入口](results/README.md)：用户已实验，仓库尚未收到逐项原始运行附件；研究 Agent 未伪造数据。
- [反馈说明](feedback/README.md) · [首轮一手来源](sources.md)。

新周按实际日期创建目录。实验轮次可以在同一周推进，已有任务和结果不因轮次更新而重置。

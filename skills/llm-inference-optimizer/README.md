# LLM Inference Optimizer Skill

**Self-evolving theory-guided inference optimization.** 面向已有具体项目、源码、优化记录和可比较实验的 Agent。通用理论、用户自己的演化理论库、项目轮次记录相互连接，而不是一个只产生优化 Tricks 的 prompt。

[图解介绍与完整工作流程](INTRODUCTION.md) · [三页可编辑流程图](diagrams/skill-workflow.drawio) · **[六层理论模型库](knowledge/structured/INDEX.md)** · [理论图谱](knowledge/structured/ATLAS.md) · [结构化使用说明](references/structured-theory.md)

## 安装

可独立复制本目录，也可保留 Git 工作区并创建符号链接。包含 [SKILL.md](SKILL.md)、[33 个结构化模型单元和 37 篇原始文档](knowledge/INDEX.md)、参考文档、模板、Python 3.10+ 标准库工具及测试。Codex 用户级安装示例（先在本机已授权的目录 clone 仓库，确认目标路径不存在；不覆盖其他安装）：

```bash
git clone https://github.com/Sunhuahai/llm-inference-theory.git ~/repos/llm-inference-theory
mkdir -p ~/.agents/skills
ln -s ~/repos/llm-inference-theory/skills/llm-inference-optimizer ~/.agents/skills/llm-inference-optimizer
```

已有 checkout 则使用实际路径；其他兼容 Agent 按其 Skill 发现目录安装。本地用户库默认 ~/.local/share/llm-inference-optimizer，不在代码仓库中。若跨项目不允许共享数据，请为不同保密域设置独立 LLM_OPTIMIZER_HOME 或 --home。Skill 本身不会访问网络、启动 GPU 或推送。

进入具体项目，明确调用：

```text
$llm-inference-optimizer
复盘本项目到目前为止的全部优化尝试及结果，根据当前理论知识库给出下一轮优先方向、理论预测和验证计划。本次只读分析。
```

若希望实际实施，明确授权候选、GPU/Agent 预算、质量不变量及范围。每个候选均需事前冻结与事后比较。对于已有实验无事前预测的历史记录，按回顾性证据使用，不补造冻结记录。

## 完整生命周期

| 命令（均位于 scripts/skill.py） | 功能 |
| --- | --- |
| init / resolve | 初始化私有理论库 / 找到当前版本和结构化索引 |
| validate-library / catalog QUERY | 验证结构、查找相关模型 |
| lookup MODEL_ID / impact MODEL_ID | 查询模型契约、先修及下游影响 |
| calculate MODEL_ID INPUT_JSON | 单位检查后的窄解析参考计算 |
| migrate VERSION REVIEW_REF | 保留旧版本和证据，显式迁移旧 Markdown 库 |
| freeze ROUND_DIR PREDICTION_JSON | 校验并冻结预测及理论版本，目录不可复用 |
| compare ROUND_DIR MEASUREMENT_JSON | 生成结构化与 Markdown 对比、收据 |
| record ROUND_DIR THEORY_NOTE | 校验并接收本地实测/失败/未决记录；拒绝 simulated |
| summary / module-summary | 按版本、context、指标及模型 revision/参数集统计实测预测误差 |
| calibrate METRIC TRAINING_ID... | 同工作点稳健乘性拟合，未验证的候选 |
| propose PROPOSAL_JSON | 创建完整候选理论文档快照与留出计划 |
| promote VERSION REVIEW_REF HOLDOUT_ID... | 在全部预登记留出与真实审阅授权通过后升级 |
| rollback ANCESTOR_VERSION REVIEW_REF | 显式回滚 active 指针，保留版本和证据 |

各候选的项目目录和用户知识库分离。参见 [预测合同](references/prediction-comparison.md) 与 [理论演化规范](references/theory-evolution.md)。独立审阅、实际实验、真实日志与数据保密由使用者和 Agent 负责，程序不会自动判断科学因果或授予权限。

## CPU 合成演示（不会进入实际知识库）

在本目录运行；临时目录与个人数据隔离：

```bash
SKILL_DIR="$(pwd -P)"
DEMO_HOME="$(mktemp -d)"
python3 "$SKILL_DIR/scripts/skill.py" --home "$DEMO_HOME/library" init
python3 "$SKILL_DIR/scripts/skill.py" --home "$DEMO_HOME/library" resolve
python3 "$SKILL_DIR/scripts/skill.py" --home "$DEMO_HOME/library" validate-library
python3 "$SKILL_DIR/scripts/skill.py" --home "$DEMO_HOME/library" catalog roofline
python3 "$SKILL_DIR/scripts/skill.py" --home "$DEMO_HOME/library" calculate FND-ROOFLINE "$SKILL_DIR/assets/roofline-input.json"
python3 "$SKILL_DIR/scripts/skill.py" --home "$DEMO_HOME/library" freeze "$DEMO_HOME/round-001" "$SKILL_DIR/assets/prediction.json"
python3 "$SKILL_DIR/scripts/skill.py" --home "$DEMO_HOME/library" compare "$DEMO_HOME/round-001" "$SKILL_DIR/assets/measurement.json"
cat "$DEMO_HOME/round-001/result/comparison.md"
python3 -m unittest discover -s tests -v
```

例子预测 TPOT 从 10 到 8 ms/token，合成实际值 8.6：预测收益 20%，实际 14%，偏差 -6 pp；HBM、kernel、未知吞吐同时进入比较。模板的 simulated 标签禁止入用户知识库。JSON 示例使用当前 seed ID，新 seed 发布时须同步更新示例。

## 模板和阅读路径

[项目审计](references/project-audit.md) · [知识路由](references/knowledge-router.md) · [瓶颈诊断](references/diagnosis.md) · [候选决策](references/decision-making.md) · [逐轮比较](references/prediction-comparison.md) · [评价](references/evaluation.md) · [理论演化](references/theory-evolution.md)

[项目概况](assets/project-profile.json) · [历史证据台账](assets/evidence-ledger.json) · [实验计划](assets/experiment-plan.json) · [复盘报告](assets/optimization-report.md) · [预测输入](assets/prediction.json) · [测量输入](assets/measurement.json) · [理论更新](assets/theory-update.md) · [下一轮计划](assets/next-plan.md) · [候选理论](assets/proposal.json)

## 边界和维护

完整的是 Skill 工作流、知识快照、可追溯记录、确定性计算、版本治理与最小参数拟合；尚没有真实 GPU adapter、自动 profiler 或通用模型训练器。理论修订由 Agent 提出、使用者授权的独立实验检验。程序的最小数值门槛不是科学充分条件。程序只在本地工作；私有实验数据、路径、密钥不得上传公共仓库。默认用户路径不在 Git 仓库内；不要把用户库或项目私有日志纳入公开提交。

升级 Skill 不覆盖用户演化文档。初始知识快照见 manifest，使用原仓库作为发布来源；重新生成时固定 source commit、保留目录结构、链接未打包内容到该 commit，校验快照目录与 manifest 一致。当前 v2 提供显式 migrate 命令，为同源旧版建立新子版本；不自动合并本地理论语义或覆盖原始记录。已有 active 的使用者先执行 resolve，按照[迁移说明](references/structured-theory.md)审查后再迁移。用户库建议定期独立备份，保留历史版本和原始证据。

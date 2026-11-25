# 用户级理论模型演化与版本治理

## 三层知识

- **Seed**：随 Skill 分发的 [37 篇初始文档](../knowledge/INDEX.md)，固定上游 commit；不可原地修改。上游是通用理论权威来源，本地是有适用域的演化副本。
- **User**：默认 ~/.local/share/llm-inference-optimizer（或 LLM_OPTIMIZER_HOME / --home）。init 只在首次建立库，不覆盖已存在 active。每次 resolve 从 active.json 指向的版本读取，版本有完整 documents、INDEX 和 manifest。读取新增文档的域、来源、限制与验证状态。用户之间物理隔离目录；不默认同步，不把私有证据提交公共仓库。
- **Project**：每个具体项目保留完整日志、配置、历史台账、轮次报告；用户库接收本地授权的结构化记录及引用，不复制原始大 trace。不同项目可以共用 user 目录但用不同 project ID；context ID 区分工作点。对于有保密要求的项目使用独立 --home，禁止跨保密域复制任何数据。

```text
user-home/
  active.json
  versions/<version>/
    INDEX.md
    manifest.json
    documents/{theory,models,optimizer,experiments,docs}/...
    proposal.json               # 候选才有
  evidence/<project>/<round>/
    prediction.json freeze.json measurement.json
    comparison.json receipt.json theory-update.md
  promotions/<version>.json
  rollbacks/<timestamp>.json
```

store 采用本地单写者目录锁和原子重命名。不要编辑 active.json、manifest、版本内文档或记录来绕过验证。SHA-256 检查帮助识别意外篡改，不提供加密、访问控制或可信时间戳；用户负责本机权限、备份、原始证据保存。在共享文件系统使用前验证锁和原子性；残留锁仅在确认无进程写入后移除。中途崩溃若已存在晋升 receipt 但 active 未切换，审查两者再人工恢复，禁止覆盖记录。

## 每轮的知识更新判断

读 [预测对比](prediction-comparison.md)、[校准](../knowledge/initial/models/calibration/README.md)、[经验记忆](../knowledge/initial/optimizer/memory/README.md)。

理论更新不是每次都改公式。填写 theory-update.md：事前假设、可比实测、机制验证、偏差来源（未辨识则不下定论）、保留/修订理由、证据、适用域与下一轮可证伪预测。record 接纳真实失败及未决作为知识线索，模拟不入库；报告类不参与实测统计。历史无预测的项目资料仍可用于形成假设，不伪造事前误差。

类型分为 calibration（有效速率/参数），correction（原模型遗漏项），extension（增加新机制），empirical（条件经验规律），revision（假设/推导变更）。经验拟合不得修改严格物理界的逻辑结论。新参数或残差模型必须写清单位、适用硬件、模型结构、负载、输入范围和失败边界；不同 context 不直接复用数值。

## 最小可运行学习

CLI summary 分组输出点预测 MAE、bias、正收益分类、区间覆盖与界满足率。校准例：

```bash
python3 "$SKILL_DIR/scripts/skill.py" calibrate tpot project-a/round-001 project-a/round-002
```

仅对同 active theory version、同 context、同指标定义下至少两个可比实测 search 点，计算 factor=median(actual/predicted)。这给出候选的乘性修正，保留原模型，必须检查新预测是否满足物理边界。不是自动学习所有模型参数，不保证泛化，也不应强加到物理界。更复杂拟合可以由 Agent 用授权数据设计并记录训练集、特征、目标、方法、参数、域和不确定性；只提交候选文档，不自动改 active。物理结构加残差的模型要保持已知约束，跨域重新校准。

## 候选、验证与晋升

1. 基于实测证据写 [proposal.json](../assets/proposal.json)：version 新 ID、parent 当前 active、scope、category、rationale、training（证据 ID）、changes（相对 documents 的 Markdown 路径 -> 完整文档内容）、targets（metric ID -> 最小可比留出数），validation_rounds 与 validation_contexts 在验证前一次性固定。新文档应带关联路径、数学式/单位、假设、失效范围、证据引用；替换文档保留原章节仍成立内容。训练和验证 ID 不重叠，验证 ID 尚无记录。不要选已知道结果的轮次。
2. propose 创建完整候选副本与版本 manifest，生成 INDEX。不修改 active。候选、拟合及其验证协议冻结。若需改动，另开候选并使用新留出集。
3. 逐个 holdout 按 [预测合同](prediction-comparison.md) 事前冻结 active 点与 shadow 点。必须是相同指标、同 baseline/定义和同一次实际观测。登记全部轮次（含失败）；不允许挑表现好的样本后改变留出名单。评审检查盲性、测量成本、独立性与覆盖。
4. 计算每目标指标 paired gain MAE。晋升最低机器门槛：全部预登记 ID/上下文匹配、每目标至少指定可比留出数（至少 2）、全部目标不退步、至少一个改善；只有 measured/passed/comparable 且漂移合格的数值纳入。若不满足则保留候选和失败证据，不自动放宽标准。独立审查还需验证真实数据、统计意义、预测覆盖、机理一致性和约束，必要时更大留出集；不是充分科学证明。
5. 只有实际独立审核或用户授权，才调用 promote 并提供可追溯的真实 approval ref。脚本记录 receipt、原子移动 active 指针。旧版本不删除。回滚只允许祖先版本，记录原因和批准引用；不能用于抹除负结果。

```bash
python3 "$SKILL_DIR/scripts/skill.py" propose /path/to/proposal.json
python3 "$SKILL_DIR/scripts/skill.py" promote local-v2 actual-review-ref project-a/holdout-001 project-b/holdout-001
python3 "$SKILL_DIR/scripts/skill.py" rollback seed-1b4c5a377634 actual-rollback-approval-ref
```

示例标识需替换为真实版本/批准引用；不存在的引用不应执行。摘要按 scope 解读，不用更新后的模型在训练集上回算并冒充历史事前预测。跨项目迁移优先迁移机制和带适用条件的经验律，硬件率和 shape 分段参数需验证。升级 Skill 时 init 保留旧库；新 seed 由用户审查差异后作为独立知识来源/候选纳入，未提供自动三方合并。

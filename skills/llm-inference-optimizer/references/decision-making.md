# 候选生成与决策

使用[候选契约](../knowledge/initial/optimizer/search-space/README.md)、[规划与搜索](../knowledge/initial/optimizer/planning/README.md)、[机制分类](../knowledge/initial/theory/optimization/mechanisms.md)。

候选不是技术名称，而是对当前 parent 的具体干预：对象、资源/依赖变化、前提、增量成本、可证伪机制、质量约束、最小验证。候选空间覆盖算子、图、执行、状态、并行、生成、服务；没有适用机会不强行覆盖。检查重复、依赖、冲突、迁移和组合后的关键路径。

排序先剔除确定违反硬约束的方案，再比较诊断信息价值、净收益、成功依据、实现+验证成本和机会成本。只有可靠概率/收益分布时才作定量期望价值；否则做条件排序，不发明精确收益或概率。近似模型不能当硬剪枝。历史失败仅限制当时的机制/工作点；说明何种新证据或 parent 改变可重启。

每个候选用 [实验计划模板](../assets/experiment-plan.json) 明确范围、预算、oracle、对照、收益预测、接受/拒绝/未决/停止条件。真正的预测和指标合同通过 freeze 验证。一次只实施被授权的范围；新候选、新 parent 或实质变更使用新 ID。验证前不得改 oracle 或缩短输出以制造收益。

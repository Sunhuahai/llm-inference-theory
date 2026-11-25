# 可行域与 Pareto

模型 ID：`FND-FEASIBILITY` · revision 1 · `derived` · [机器可读定义](model.json)

## 核心关系

`max J(x) subject to quality, memory, SLO and budget; Pareto dominance`

目标方向、约束和可比资源明确；未知约束不能默认通过。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| candidate_set | record | 候选目标及约束 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| feasible_set | record | 满足约束的候选 |
| pareto_set | record | 非支配候选 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

Pareto 不给唯一偏好；不可行或未知不能作为已验证最优。

## 验证与反例

独立质量和性能检验后再判可行；明确 SLO 与预算。

## 依赖与来源

依赖 ID：`DEF-TASK`、`DEF-METRIC`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/optimization/coupling.md](../../../initial/theory/optimization/coupling.md)
- [optimizer/foundations/autonomous-optimization.md](../../../initial/optimizer/foundations/autonomous-optimization.md)

本模块为理论/模型规范，暂无通用可执行 evaluator。

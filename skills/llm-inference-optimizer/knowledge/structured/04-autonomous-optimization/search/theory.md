# 候选表示与约束搜索

模型 ID：`AGT-SEARCH` · revision 1 · `method` · [机器可读定义](model.json)

## 核心关系

`A(s) = admissible interventions given state, task, constraints, parent and evidence`

候选 ID 绑定 parent、作用机制、前提、成本和验证；硬约束与近似排名区分。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| state | record | 当前系统、理论、证据和预算 |
| mechanisms | record | 可能干预及前提 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| candidate_set | record | 可行/未知/不可行与理由 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

不能因一次历史失败排除机制；未校准预测不是安全剪枝。

## 验证与反例

记录搜索过程、重复尝试、拒绝原因；同预算比较候选有效率。

## 依赖与来源

依赖 ID：`OPT-TRANSFORM`、`OPT-COUPLING`、`AGT-DIAGNOSIS`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [optimizer/foundations/autonomous-optimization.md](../../../initial/optimizer/foundations/autonomous-optimization.md)
- [optimizer/search-space/README.md](../../../initial/optimizer/search-space/README.md)
- [optimizer/planning/README.md](../../../initial/optimizer/planning/README.md)

本模块为理论/方法规范，暂无通用可执行 evaluator。

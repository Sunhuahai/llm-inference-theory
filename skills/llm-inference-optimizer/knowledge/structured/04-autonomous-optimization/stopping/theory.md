# 停止条件与重新开启

模型 ID：`AGT-STOPPING` · revision 1 · `method` · [机器可读定义](model.json)

## 核心关系

`stop if declared goal met, feasible verification budget exhausted, or justified marginal value insufficient`

使用用户/项目预先规定目标和约束，不发明门槛。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| state | record | 当前 best、剩余预算、未决假设与目标 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| decision | record | 继续、暂缓、停止及重启条件 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

数轮失败不是全局最优证明；瓶颈变化可使旧失败重新有价值。

## 验证与反例

登记剩余预算和机会、不可辨识因素、停止依据与可复核重启条件。

## 依赖与来源

依赖 ID：`AGT-PLANNING`、`AGT-EXPERIMENT`、`FND-FEASIBILITY`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [optimizer/planning/README.md](../../../initial/optimizer/planning/README.md)
- [optimizer/memory/README.md](../../../initial/optimizer/memory/README.md)
- [optimizer/foundations/autonomous-optimization.md](../../../initial/optimizer/foundations/autonomous-optimization.md)

本模块为理论/方法规范，暂无通用可执行 evaluator。

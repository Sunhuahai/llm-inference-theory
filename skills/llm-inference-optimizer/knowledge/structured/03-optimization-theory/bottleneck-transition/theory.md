# 瓶颈迁移与相图

模型 ID：`OPT-TRANSITION` · revision 1 · `method` · [机器可读定义](model.json)

## 核心关系

`r*=argmax_i T_i under an explicit max-cost model; recompute after a(x)`

只在明确的资源竞争或执行图模型内判定限制；必要时识别并列约束。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| resource_times | record | 各资源必要时间或条件估计 |
| intervention | record | 对资源和执行图的变化 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| regime | record | 当前/候选限制、并列及未识别因素 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

低 GPU 利用率不等于某个瓶颈；单一资源最大值可能漏掉暴露依赖。

## 验证与反例

干预后重测并重新诊断；从不同工作点构建切换区域，复查旧失败。

## 依赖与来源

依赖 ID：`FND-ROOFLINE`、`OPT-SENSITIVITY`、`OPT-TRANSFORM`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/performance/README.md](../../../initial/theory/performance/README.md)
- [theory/diagnosis/README.md](../../../initial/theory/diagnosis/README.md)
- [theory/optimization/coupling.md](../../../initial/theory/optimization/coupling.md)

本模块为理论/方法规范，暂无通用可执行 evaluator。

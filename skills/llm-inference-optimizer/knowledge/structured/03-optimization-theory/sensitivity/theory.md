# 资源敏感性与局部边际价值

模型 ID：`OPT-SENSITIVITY` · revision 1 · `derived` · [机器可读定义](model.json)

## 核心关系

`E_r = d log T / d log r; finite difference only within declared regime`

固定其它工作点或明确混杂；局部可微或使用有限差分。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| baseline_time | s | 当前时间 |
| perturbed_time | s | 单因素干预时间 |
| resource_ratio | ratio | 干预/基线资源比 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| elasticity | ratio | 条件对数弹性估计 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

max 切换点导数可能不存在；共变因素使因果解释失效。

## 验证与反例

多幅度独立干预，比较单因素响应及非线性；检查瓶颈切换。

## 依赖与来源

依赖 ID：`FND-ROOFLINE`、`PERF-EXECUTION`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/performance/README.md](../../../initial/theory/performance/README.md)
- [theory/diagnosis/README.md](../../../initial/theory/diagnosis/README.md)
- [theory/optimization/benefit-prediction.md](../../../initial/theory/optimization/benefit-prediction.md)

具备窄解析参考函数：elasticity；输入必须满足单位与适用域，输出不是实测。

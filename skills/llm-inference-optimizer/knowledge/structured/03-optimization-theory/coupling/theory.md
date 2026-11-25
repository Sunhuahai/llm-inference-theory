# 优化耦合与条件收益

模型 ID：`OPT-COUPLING` · revision 1 · `derived` · [机器可读定义](model.json)

## 核心关系

`I_AB=J_AB-J_A-J_B+J_0; Delta(B|A)=J_AB-J_A`

四组同目标、同工作负载和约束；目标方向事先约定，顺序可能影响结果。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| j0 | scalar | 固定目标基线 |
| ja | scalar | 单项 A |
| jb | scalar | 单项 B |
| jab | scalar | 组合 AB |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| interaction | scalar | 同方向目标的二阶交互 |
| conditional_b | scalar | A 后 B 的增量 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

交互不是普适常数；未测组合只能作为预测，不能假定加速比相乘。

## 验证与反例

完整四组对照及必要的 A→B/B→A 顺序对照；比较重复不确定性。

## 依赖与来源

依赖 ID：`OPT-BENEFIT`、`OPT-TRANSITION`、`FND-FEASIBILITY`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/optimization/coupling.md](../../../initial/theory/optimization/coupling.md)
- [optimizer/evolution/README.md](../../../initial/optimizer/evolution/README.md)

具备窄解析参考函数：coupling；输入必须满足单位与适用域，输出不是实测。

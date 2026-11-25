# 净收益与条件加速上限

模型 ID：`OPT-BENEFIT` · revision 1 · `derived` · [机器可读定义](model.json)

## 核心关系

`g=(T0-T1)/T0; S=T0/T1; S_max=T0/L1 if L1 is a valid lower bound`

基线和候选同任务、同指标；严格上限需要实测 T0 与有效必要下界 L1。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| baseline_time | s | 已测或明确标注的基线 |
| candidate_time | s | 同口径候选时间预测 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| gain | ratio | 预测改善 |
| speedup | ratio | 条件加速比 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

两个近似值相除只是估计；不能直接把不同层的节省叠加。

## 验证与反例

冻结预测并独立对照同工作点；报告收益偏差与资源变化。

## 依赖与来源

依赖 ID：`DEF-METRIC`、`FND-AMDAHL`、`PERF-EXECUTION`、`PERF-SERVING`、`OPT-TRANSFORM`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/optimization/benefit-prediction.md](../../../initial/theory/optimization/benefit-prediction.md)
- [theory/optimization/coupling.md](../../../initial/theory/optimization/coupling.md)
- [models/contracts.md](../../../initial/models/contracts.md)

具备窄解析参考函数：benefit；输入必须满足单位与适用域，输出不是实测。

# 独立验证与实验设计

模型 ID：`AGT-EXPERIMENT` · revision 1 · `method` · [机器可读定义](model.json)

## 核心关系

`compare pre-frozen prediction with independent measurement under same task/context`

冻结任务、质量、资源和统计口径；实验授权明确。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| hypothesis | record | 竞争假设及预测 |
| candidate | record | 干预与基线 |
| protocol | record | 独立 oracle、指标、重复、留出 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| evidence | record | 正负结果、可比性和不确定性 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

模拟不是测量；重复同一次试验不是独立留出；候选 Agent 不得改 oracle。

## 验证与反例

正确性→微基准→端到端，检查真实后端、失败、原始日志和基线漂移。

## 依赖与来源

依赖 ID：`AGT-DIAGNOSIS`、`AGT-PLANNING`、`DEF-METRIC`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [optimizer/evaluation/README.md](../../../initial/optimizer/evaluation/README.md)
- [docs/benchmark-protocol.md](../../../initial/docs/benchmark-protocol.md)
- [experiments/model-validation/README.md](../../../initial/experiments/model-validation/README.md)

本模块为理论/方法规范，暂无通用可执行 evaluator。

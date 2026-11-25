# 物理结构与残差学习

模型 ID：`LRN-RESIDUAL` · revision 1 · `hypothesis` · [机器可读定义](model.json)

## 核心关系

`T_hat(x)=T_physics(x;theta)+r_phi(x); enforce applicable lower bound`

明确遗漏机制与特征，保持量纲、必要物理界和任务约束。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| physics_model | record | 基模型和适用边界 |
| residual_data | record | 事前误差与训练特征 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| residual_candidate | record | 待验证结构或残差模型 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

当前无通用残差训练器；残差相关性不证明因果，不得用留出结果拟合。

## 验证与反例

预登记特征与模型族，独立比较原模型和候选，检查边界和外推。

## 依赖与来源

依赖 ID：`LRN-CALIBRATION`、`FND-ROOFLINE`、`PERF-EXECUTION`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [models/calibration/README.md](../../../initial/models/calibration/README.md)
- [experiments/model-validation/README.md](../../../initial/experiments/model-validation/README.md)
- [docs/methodology.md](../../../initial/docs/methodology.md)

本模块为理论/方法规范，暂无通用可执行 evaluator。

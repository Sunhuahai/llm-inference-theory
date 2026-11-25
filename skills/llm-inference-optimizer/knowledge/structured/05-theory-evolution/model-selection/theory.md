# 理论修订与独立模型选择

模型 ID：`LRN-SELECTION` · revision 1 · `method` · [机器可读定义](model.json)

## 核心关系

`compare paired holdout errors of M_k and M'_k; promote only with preregistered gates and review`

训练/验证隔离、同一真实观测的双模型预测、实际审核授权。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| training | record | 支持修订的证据 |
| holdout | record | 预登记独立数据及目标 |
| candidate_version | record | 冻结修订及依赖影响 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| decision | record | 保留、晋升、拒绝或回滚 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

最小 MAE 门槛非科学充分条件；结构修订需理论审查。

## 验证与反例

核查所有登记轮次、上下文、配对误差、泄漏、物理约束及下游回归。

## 依赖与来源

依赖 ID：`LRN-CALIBRATION`、`AGT-EXPERIMENT`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [experiments/model-validation/README.md](../../../initial/experiments/model-validation/README.md)
- [models/calibration/README.md](../../../initial/models/calibration/README.md)
- [docs/methodology.md](../../../initial/docs/methodology.md)

本模块为理论/方法规范，暂无通用可执行 evaluator。

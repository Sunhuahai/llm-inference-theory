# 跨项目知识迁移与反例

模型 ID：`LRN-TRANSFER` · revision 1 · `method` · [机器可读定义](model.json)

## 核心关系

`transfer mechanisms conditionally; recalibrate numeric parameters for new contexts`

对照硬件、shape、dtype、架构、引擎及负载，证据同保密域且获授权。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| source_evidence | record | 原项目理论、正负证据和 scope |
| target_context | record | 目标模型/硬件/负载/引擎 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| transfer_plan | record | 可复用机制、待校准项、限制与反例 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

单一项目实测不能证明普适规律；跨保密域不得自动共享证据。

## 验证与反例

在新环境先运行低成本校准和独立验证，记录失败和适用范围。

## 依赖与来源

依赖 ID：`LRN-CALIBRATION`、`LRN-SELECTION`、`AGT-STOPPING`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [optimizer/memory/README.md](../../../initial/optimizer/memory/README.md)
- [experiments/case-studies/README.md](../../../initial/experiments/case-studies/README.md)
- [models/calibration/README.md](../../../initial/models/calibration/README.md)

本模块为理论/方法规范，暂无通用可执行 evaluator。

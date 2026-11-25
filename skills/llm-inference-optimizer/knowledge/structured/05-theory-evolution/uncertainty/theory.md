# 预测误差与不确定性

模型 ID：`LRN-UNCERTAINTY` · revision 1 · `derived` · [机器可读定义](model.json)

## 核心关系

`MAE=mean(|g_actual-g_pred|); coverage=mean(1{lo<=actual<=hi})`

只在相同指标、context、理论版本和预测类型下聚合；严格界另计。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| gain_errors | record | 可比实测的有符号收益误差 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| mae | ratio | 平均绝对收益误差 |
| bias | ratio | 平均有符号收益误差 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

无校准情景区间不冒充置信区间；失败/缺失不计零；选择偏差需披露。

## 验证与反例

冻结预测，按组报告样本数、点误差、区间覆盖及界违背，独立留出。

## 依赖与来源

依赖 ID：`DEF-METRIC`、`AGT-EXPERIMENT`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [models/calibration/README.md](../../../initial/models/calibration/README.md)
- [experiments/model-validation/README.md](../../../initial/experiments/model-validation/README.md)

具备窄解析参考函数：error_summary；输入必须满足单位与适用域，输出不是实测。

# 参数校准与条件拟合

模型 ID：`LRN-CALIBRATION` · revision 1 · `learning` · [机器可读定义](model.json)

## 核心关系

`theta*=argmin sum loss(y_i,f(x_i;theta)); example factor=median(y_i/yhat_i)`

固定同定义/单位/工作点；分割训练和留出，物理参数检查约束。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| paired_training | record | 同模型 revision、context、指标定义的可比测量 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| factor | ratio | 候选乘性校准系数 |
| parameter_candidate | record | 待验证参数集 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

拟合训练集不是泛化；不能用拟合参数改变数学下界。

## 验证与反例

用未见同域或预登记跨域实验做 paired 比较，报告样本数、误差和不确定性。

## 依赖与来源

依赖 ID：`PERF-HARDWARE`、`PERF-OPERATOR`、`AGT-EXPERIMENT`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [models/calibration/README.md](../../../initial/models/calibration/README.md)
- [experiments/model-validation/README.md](../../../initial/experiments/model-validation/README.md)

具备窄解析参考函数：median_factor；输入必须满足单位与适用域，输出不是实测。

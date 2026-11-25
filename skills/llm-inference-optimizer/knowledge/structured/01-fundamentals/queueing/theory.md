# 稳定队列与 Little 定律

模型 ID：`FND-QUEUEING` · revision 1 · `derived` · [机器可读定义](model.json)

## 核心关系

`L=lambda*W (stable system); M/M/1 W=1/(mu-lambda) under assumptions`

Little 定律需稳定系统与一致的边界；M/M/1 另需泊松到达、指数服务、单服务台、lambda<mu。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| arrival_rate | 1/s | 长期到达率 |
| service_rate | 1/s | M/M/1 服务率 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| mean_time | s | M/M/1 稳态平均系统时间 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

不适用于直接预测动态 batching 的尾延迟；非稳态或过载需另模型。

## 验证与反例

用 trace 检查稳定性、到达和服务分布，核对在途数、吞吐、等待。

## 依赖与来源

依赖 ID：`DEF-METRIC`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/serving/README.md](../../../initial/theory/serving/README.md)
- [models/serving/README.md](../../../initial/models/serving/README.md)

具备窄解析参考函数：mm1；输入必须通过单位和适用域检查，结果不是实测。

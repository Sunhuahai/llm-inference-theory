# 有效生成轮次与单位 token 成本

模型 ID：`PERF-GENERATION` · revision 1 · `performance_model` · [机器可读定义](model.json)

## 核心关系

`TPOT_hat = E[T_round]/E[K_committed] (long-run renewal-style average)`

同生成语义，稳定可比的轮次；分子包含所有实际成本，分母只计提交 token。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| round_time | s | 平均完整 draft/verify/回退时间 |
| committed | count | 每轮平均有效提交 token 数 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| tpot | s/token | 长期单位有效 token 成本 |
| tpot_ms | ms/token | 长期单位有效 token 成本 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

不能代替逐请求 TPOT 分布、尾延迟或输出质量；短序列边界需另处理。

## 验证与反例

记录轮次分布、接受率、有效 token、状态成本及 reference 输出。

## 依赖与来源

依赖 ID：`PERF-EXECUTION`、`PERF-STATE`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/generation/README.md](../../../initial/theory/generation/README.md)
- [theory/optimization/benefit-prediction.md](../../../initial/theory/optimization/benefit-prediction.md)

具备窄解析参考函数：generation_time；输入必须通过单位和适用域检查，结果不是实测。

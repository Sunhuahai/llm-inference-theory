# 算子时间预测

模型 ID：`PERF-OPERATOR` · revision 1 · `performance_model` · [机器可读定义](model.json)

## 核心关系

`T_hat=max(F/P_eff,D/B_eff)+t_exposed`

同 shape、dtype、实现、缓存与并发；有效速率可来自匹配微基准。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| flops | FLOP | 工作量 |
| bytes | byte | 实际预计流量 |
| effective_compute | FLOP/s | 工作点有效算力 |
| effective_bandwidth | byte/s | 工作点有效带宽 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| time | s | 算子时间估计 |
| time_ms | ms | 算子时间估计 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| exposed_overhead | s | 未重复计入且暴露的额外时间 |

默认无实测参数集；只能在匹配 context 与证据下拟合，候选须独立验证。

## 适用边界

max 为重叠近似，不是实测或严格下界；spill、争用与 epilogue 需显式建模。

## 验证与反例

留出形状对比实际 kernel 时间和资源量；避免同时拟合不可辨识参数。

## 依赖与来源

依赖 ID：`PERF-RESOURCE`、`PERF-HARDWARE`、`FND-ROOFLINE`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [models/operator/README.md](../../../initial/models/operator/README.md)
- [theory/optimization/benefit-prediction.md](../../../initial/theory/optimization/benefit-prediction.md)
- [models/calibration/README.md](../../../initial/models/calibration/README.md)

具备窄解析参考函数：operator_time；输入必须通过单位和适用域检查，结果不是实测。

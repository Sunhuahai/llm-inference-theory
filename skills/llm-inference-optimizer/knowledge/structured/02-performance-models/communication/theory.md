# 通信与拓扑时间

模型 ID：`PERF-COMMUNICATION` · revision 1 · `performance_model` · [机器可读定义](model.json)

## 核心关系

`T_hat=alpha*N_messages+bytes/B_eff (fixed route/algorithm)`

固定 collective、rank、消息规模、拓扑及竞争状态；明确 bytes 计量。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| messages | count | 协议消息数 |
| bytes | byte | 通信工作量 |
| bandwidth | byte/s | 对应路径的条件有效带宽 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| time | s | 通信阶段时间估计 |
| time_ms | ms | 通信阶段时间估计 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| startup | s | 每消息启动开销 |

默认无实测参数集；只能在匹配 context 与证据下拟合，候选须独立验证。

## 适用边界

未建模拥塞、rank 晚到、分段协议和重叠；不能据该式直接算端到端。

## 验证与反例

按消息和拓扑做微基准、核对 rank 时间线，并在并发条件下留出验证。

## 依赖与来源

依赖 ID：`DEF-RESOURCE`、`PERF-HARDWARE`、`FND-IO`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/execution/README.md](../../../initial/theory/execution/README.md)
- [models/execution/README.md](../../../initial/models/execution/README.md)
- [models/operator/README.md](../../../initial/models/operator/README.md)

具备窄解析参考函数：communication_time；输入必须通过单位和适用域检查，结果不是实测。

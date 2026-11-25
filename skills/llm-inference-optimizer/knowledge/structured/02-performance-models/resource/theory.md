# 模型架构资源账本

模型 ID：`PERF-RESOURCE` · revision 1 · `performance_model` · [机器可读定义](model.json)

## 核心关系

`R(x) = {F_i, D_i,l, M_i(t), C_i} for operations and ranks`

按真实架构、路由与布局建账；密集/稀疏/混合状态分开。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| model_spec | record | 架构、shape、权重、状态 |
| workload_spec | record | 实际 token/batch/长度分布 |
| execution_spec | record | 布局、融合、路由及并行 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| flops | FLOP | 计算需求 |
| traffic | byte | 分层流量 |
| state | byte | 状态容量 |
| ledger | record | 分算子分 rank 成本 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

理论必要量、预计实际量和 profiler 观测分离；不把峰值当实际。

## 验证与反例

核对算子 shape、真实后端、流量计数器、容量高水位及专家负载。

## 依赖与来源

依赖 ID：`FND-COMPUTE`、`FND-IO`、`DEF-TASK`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [models/resource/README.md](../../../initial/models/resource/README.md)
- [theory/foundations/unified-theory.md](../../../initial/theory/foundations/unified-theory.md)
- [models/contracts.md](../../../initial/models/contracts.md)

本模块为理论/模型规范，暂无通用可执行 evaluator。

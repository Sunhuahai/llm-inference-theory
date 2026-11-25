# 执行图与排程估计

模型 ID：`PERF-EXECUTION` · revision 1 · `performance_model` · [机器可读定义](model.json)

## 核心关系

`T_hat = longest_path(G_with_resource_order, t_hat)`

资源顺序和依赖明确；节点时间对应相同工作点；关键路径需重算。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| dag | record | 依赖与固定资源顺序 DAG；节点使用有效时长 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| time | s | 给定排程的模型估计 |
| time_ms | ms | 给定排程的模型估计 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

固定顺序估计不等于任意调度下界；未建模共享资源竞争、CPU 空隙和动态依赖。

## 验证与反例

核对 trace 中暴露时间、重叠和实际服务 step，做干预前后留出比较。

## 依赖与来源

依赖 ID：`PERF-OPERATOR`、`PERF-COMMUNICATION`、`FND-CRITICAL-PATH`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/execution/README.md](../../../initial/theory/execution/README.md)
- [models/execution/README.md](../../../initial/models/execution/README.md)

具备窄解析参考函数：execution_time；输入必须通过单位和适用域检查，结果不是实测。

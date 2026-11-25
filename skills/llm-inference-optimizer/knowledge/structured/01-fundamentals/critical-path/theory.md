# 关键路径与串行依赖

模型 ID：`FND-CRITICAL-PATH` · revision 1 · `physical_bound` · [机器可读定义](model.json)

## 核心关系

`T >= max_{path} sum(t_i_lower)`

无环必要依赖；节点时长为独立必要下界且计量边界一致。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| dag | record | 节点下界时长及必要依赖 DAG |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| time | s | 最长必要路径下界 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

忽略资源冲突；仅凭 trace 的重叠不能证明可删依赖。

## 验证与反例

用执行时间线验证边和节点归属；改变局部成本后重新求最长路径。

## 依赖与来源

依赖 ID：`DEF-RESOURCE`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/execution/README.md](../../../initial/theory/execution/README.md)
- [theory/performance/README.md](../../../initial/theory/performance/README.md)

具备窄解析参考函数：critical_path；输入必须通过单位和适用域检查，结果不是实测。

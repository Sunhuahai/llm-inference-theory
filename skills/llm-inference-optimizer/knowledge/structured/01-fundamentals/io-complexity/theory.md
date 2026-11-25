# 数据移动与 I/O 复杂度

模型 ID：`FND-IO` · revision 1 · `physical_bound` · [机器可读定义](model.json)

## 核心关系

`T_io >= D_level / B_level_max`

指定存储层与算法、布局、复用假设；工作量和上界成立时。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| bytes | byte | 某层必须移动的字节数 |
| bandwidth | byte/s | 相应层速率上界 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| time | s | 条件 I/O 时间下界 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

逻辑张量大小不等于 HBM 实际流量；不同层下界不可随意相加。

## 验证与反例

采集相应层流量及速率上界，若实际低于下界审计工作量与计时范围。

## 依赖与来源

依赖 ID：`DEF-RESOURCE`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/performance/README.md](../../../initial/theory/performance/README.md)
- [theory/memory/README.md](../../../initial/theory/memory/README.md)

具备窄解析参考函数：io_bound；输入必须通过单位和适用域检查，结果不是实测。

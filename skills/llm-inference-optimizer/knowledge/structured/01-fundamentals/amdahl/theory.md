# 局部收益的 Amdahl 边界

模型 ID：`FND-AMDAHL` · revision 1 · `derived` · [机器可读定义](model.json)

## 核心关系

`S=1/((1-f)+f/s)`

固定其余时间，串行可加成本分解且 f 可识别。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| fraction | ratio | 原串行可分解总时间中被加速的份额 |
| speedup | ratio | 该部分加速倍数 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| speedup | ratio | 条件整体加速比 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

若重叠、排程、资源迁移改变其他成本，不能直接套用；f 不是 GPU 累计活动时间占比。

## 验证与反例

固定任务与工作点，比较局部干预前后暴露时间；检查未受影响项。

## 依赖与来源

依赖 ID：`DEF-METRIC`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/optimization/benefit-prediction.md](../../../initial/theory/optimization/benefit-prediction.md)
- [theory/execution/README.md](../../../initial/theory/execution/README.md)

具备窄解析参考函数：amdahl；输入必须通过单位和适用域检查，结果不是实测。

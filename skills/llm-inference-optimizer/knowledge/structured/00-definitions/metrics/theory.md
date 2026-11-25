# 性能指标与比较契约

模型 ID：`DEF-METRIC` · revision 1 · `definition` · [机器可读定义](model.json)

## 核心关系

`g_lower=(b-y)/b; g_higher=(y-b)/b (b>0)`

低负载请求延迟、饱和吞吐、SLO goodput 分别评价；方向和统计量固定。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| baseline | scalar | 同口径基线 |
| actual | scalar | 候选测量 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| gain | ratio | 有方向的相对改善 |
| metric_spec | record | scope/unit/statistic/direction/窗口 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

此模块没有可拟合参数；不要把数学关系或定义当作经验拟合。

## 适用边界

零基线不定义相对收益；closed-loop 与 open-loop 不自动可比。

## 验证与反例

复查原始计时窗口、完整生成长度、质量、缓存与并发。

## 依赖与来源

依赖 ID：`DEF-TASK`、`DEF-RESOURCE`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [docs/benchmark-protocol.md](../../../initial/docs/benchmark-protocol.md)
- [theory/serving/README.md](../../../initial/theory/serving/README.md)
- [models/contracts.md](../../../initial/models/contracts.md)

本模块为理论/模型规范，暂无通用可执行 evaluator。

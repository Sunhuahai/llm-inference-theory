# 预算感知决策与信息价值

模型 ID：`AGT-PLANNING` · revision 1 · `method` · [机器可读定义](model.json)

## 核心关系

`score(a)=E[net gain|evidence] + value_of_information - weighted_cost (when quantities commensurate)`

定量期望收益需要校准的分布和明确代价尺度；否则作条件排序。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| candidates | record | 收益、风险与证据 |
| budget | record | GPU、Agent 与墙钟预算 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| ranked_plan | record | 方向、条件排序、成本和实验 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

评分不是最优性证明；不能伪造成功率、成本或硬预算。

## 验证与反例

用同工具、同权限、同预算 benchmark 比较最终收益、有效候选和成本。

## 依赖与来源

依赖 ID：`AGT-SEARCH`、`OPT-BENEFIT`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [optimizer/planning/README.md](../../../initial/optimizer/planning/README.md)
- [experiments/agent-benchmark/README.md](../../../initial/experiments/agent-benchmark/README.md)

本模块为理论/方法规范，暂无通用可执行 evaluator。

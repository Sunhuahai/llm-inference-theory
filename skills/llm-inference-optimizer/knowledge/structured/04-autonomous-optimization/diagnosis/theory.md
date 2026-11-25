# 反事实瓶颈诊断

模型 ID：`AGT-DIAGNOSIS` · revision 1 · `method` · [机器可读定义](model.json)

## 核心关系

`H={h_i}; choose observation where predictions p(y|h_i,a) differ`

不能只凭相关性判因果；对每个假设写可反驳观测。

## 输入与输出

| 输入 | 单位 | 说明 |
| --- | --- | --- |
| observations | record | 真实计数器、trace、历史干预 |
| candidate_hypotheses | record | 竞争机制及条件 |

| 输出 | 单位 | 说明 |
| --- | --- | --- |
| diagnosis | record | 支持、反对、未辨识 |
| probe | record | 最小区分性实验 |

## 参数与校准

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| — | — | 无 |

本模块不含待拟合参数；学习方法作用于其他模型或参数，不自动升级理论。

## 适用边界

多种原因给相同总时延时重复总时延不能提升结构可辨识性。

## 验证与反例

改变单因素或比较可区分 trace；保留替代解释、控制测量扰动。

## 依赖与来源

依赖 ID：`OPT-TRANSITION`、`PERF-SERVING`、`DEF-METRIC`。依赖可通过 [理论索引](../../INDEX.md) 查询。

- [theory/diagnosis/README.md](../../../initial/theory/diagnosis/README.md)
- [optimizer/diagnosis/README.md](../../../initial/optimizer/diagnosis/README.md)

本模块为理论/方法规范，暂无通用可执行 evaluator。

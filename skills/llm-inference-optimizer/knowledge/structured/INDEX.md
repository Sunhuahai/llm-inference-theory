# 结构化理论模型总索引

33 个初始理论单元（后续版本由程序生成索引）。每个单元的 model.json 是唯一结构化事实源，theory.md 用于阅读、假设及推导；详细来源仍见 [37 篇原始文档](../INDEX.md)。

[图谱](ATLAS.md) · [注册表](registry.json) · [依赖图](dependencies.json) · [参数集](parameters.json) · [模式与约定](schemas/README.md)

## L0 · 定义与度量

| 模型 ID | 模块 | 机器可读定义 |
| --- | --- | --- |
| `DEF-TASK` | [任务与不变量](00-definitions/task/theory.md) | [JSON](00-definitions/task/model.json) |
| `DEF-RESOURCE` | [资源与量纲](00-definitions/resource/theory.md) | [JSON](00-definitions/resource/model.json) |
| `DEF-METRIC` | [性能指标与比较契约](00-definitions/metrics/theory.md) | [JSON](00-definitions/metrics/model.json) |

## L1 · 基础规律

| 模型 ID | 模块 | 机器可读定义 |
| --- | --- | --- |
| `FND-COMPUTE` | [计算工作量与复杂度](01-fundamentals/compute-complexity/theory.md) | [JSON](01-fundamentals/compute-complexity/model.json) |
| `FND-IO` | [数据移动与 I/O 复杂度](01-fundamentals/io-complexity/theory.md) | [JSON](01-fundamentals/io-complexity/model.json) |
| `FND-ROOFLINE` | [多资源 Roofline 下界](01-fundamentals/roofline/theory.md) | [JSON](01-fundamentals/roofline/model.json) |
| `FND-CRITICAL-PATH` | [关键路径与串行依赖](01-fundamentals/critical-path/theory.md) | [JSON](01-fundamentals/critical-path/model.json) |
| `FND-AMDAHL` | [局部收益的 Amdahl 边界](01-fundamentals/amdahl/theory.md) | [JSON](01-fundamentals/amdahl/model.json) |
| `FND-QUEUEING` | [稳定队列与 Little 定律](01-fundamentals/queueing/theory.md) | [JSON](01-fundamentals/queueing/model.json) |
| `FND-FEASIBILITY` | [可行域与 Pareto](01-fundamentals/constrained-optimization/theory.md) | [JSON](01-fundamentals/constrained-optimization/model.json) |

## L2 · 性能模型

| 模型 ID | 模块 | 机器可读定义 |
| --- | --- | --- |
| `PERF-RESOURCE` | [模型架构资源账本](02-performance-models/resource/theory.md) | [JSON](02-performance-models/resource/model.json) |
| `PERF-HARDWARE` | [硬件能力与有效速率](02-performance-models/hardware/theory.md) | [JSON](02-performance-models/hardware/model.json) |
| `PERF-OPERATOR` | [算子时间预测](02-performance-models/operator/theory.md) | [JSON](02-performance-models/operator/model.json) |
| `PERF-COMMUNICATION` | [通信与拓扑时间](02-performance-models/communication/theory.md) | [JSON](02-performance-models/communication/model.json) |
| `PERF-STATE` | [状态容量、流量与复用](02-performance-models/memory-state/theory.md) | [JSON](02-performance-models/memory-state/model.json) |
| `PERF-EXECUTION` | [执行图与排程估计](02-performance-models/execution/theory.md) | [JSON](02-performance-models/execution/model.json) |
| `PERF-GENERATION` | [有效生成轮次与单位 token 成本](02-performance-models/generation/theory.md) | [JSON](02-performance-models/generation/model.json) |
| `PERF-SERVING` | [服务级性能与 Goodput](02-performance-models/serving/theory.md) | [JSON](02-performance-models/serving/model.json) |

## L3 · 优化理论

| 模型 ID | 模块 | 机器可读定义 |
| --- | --- | --- |
| `OPT-TRANSFORM` | [优化操作与资源变换](03-optimization-theory/resource-transformation/theory.md) | [JSON](03-optimization-theory/resource-transformation/model.json) |
| `OPT-SENSITIVITY` | [资源敏感性与局部边际价值](03-optimization-theory/sensitivity/theory.md) | [JSON](03-optimization-theory/sensitivity/model.json) |
| `OPT-BENEFIT` | [净收益与条件加速上限](03-optimization-theory/benefit-prediction/theory.md) | [JSON](03-optimization-theory/benefit-prediction/model.json) |
| `OPT-TRANSITION` | [瓶颈迁移与相图](03-optimization-theory/bottleneck-transition/theory.md) | [JSON](03-optimization-theory/bottleneck-transition/model.json) |
| `OPT-COUPLING` | [优化耦合与条件收益](03-optimization-theory/coupling/theory.md) | [JSON](03-optimization-theory/coupling/model.json) |

## L4 · Agent 决策

| 模型 ID | 模块 | 机器可读定义 |
| --- | --- | --- |
| `AGT-DIAGNOSIS` | [反事实瓶颈诊断](04-autonomous-optimization/diagnosis/theory.md) | [JSON](04-autonomous-optimization/diagnosis/model.json) |
| `AGT-SEARCH` | [候选表示与约束搜索](04-autonomous-optimization/search/theory.md) | [JSON](04-autonomous-optimization/search/model.json) |
| `AGT-PLANNING` | [预算感知决策与信息价值](04-autonomous-optimization/planning/theory.md) | [JSON](04-autonomous-optimization/planning/model.json) |
| `AGT-EXPERIMENT` | [独立验证与实验设计](04-autonomous-optimization/experiment-design/theory.md) | [JSON](04-autonomous-optimization/experiment-design/model.json) |
| `AGT-STOPPING` | [停止条件与重新开启](04-autonomous-optimization/stopping/theory.md) | [JSON](04-autonomous-optimization/stopping/model.json) |

## L5 · 理论学习

| 模型 ID | 模块 | 机器可读定义 |
| --- | --- | --- |
| `LRN-CALIBRATION` | [参数校准与条件拟合](05-theory-evolution/calibration/theory.md) | [JSON](05-theory-evolution/calibration/model.json) |
| `LRN-RESIDUAL` | [物理结构与残差学习](05-theory-evolution/residual-learning/theory.md) | [JSON](05-theory-evolution/residual-learning/model.json) |
| `LRN-SELECTION` | [理论修订与独立模型选择](05-theory-evolution/model-selection/theory.md) | [JSON](05-theory-evolution/model-selection/model.json) |
| `LRN-UNCERTAINTY` | [预测误差与不确定性](05-theory-evolution/uncertainty/theory.md) | [JSON](05-theory-evolution/uncertainty/model.json) |
| `LRN-TRANSFER` | [跨项目知识迁移与反例](05-theory-evolution/knowledge-transfer/theory.md) | [JSON](05-theory-evolution/knowledge-transfer/model.json) |

<!-- Generated seed snapshot; source: theory/INDEX.md @ 1b4c5a37763475eb39b31dd083fcad256fc2578e. Do not hand-edit. -->
# 理论体系索引

先读[图解导读](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/theory/README.md)与[统一理论总纲](foundations/unified-theory.md)，再按问题进入下面的独立专题。章节中的公式均受明示假设约束；内容完成不等于已有实测验证。

[基础篇入口](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/theory/foundations/README.md)统一资源、状态与生成的描述；[优化篇入口](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/theory/optimization/README.md)连接机制、条件收益和组合验证。

| 章节 | 核心问题 | 主入口 | 与自主优化的关系 |
| --- | --- | --- | --- |
| 01 推理资源成本 | 实际工作量、流量、状态与通信是什么？ | [总纲](foundations/unified-theory.md) | 建立候选的资源作用机制 |
| 02 性能边界 | 哪些资源或依赖限制速度？ | [性能理论](performance/README.md) | 区分严格界与可达性能 |
| 03 瓶颈诊断 | 哪些观测能区分不同慢因？ | [诊断理论](diagnosis/README.md) | 选择下一项区分性实验 |
| 04 执行与并行 | 依赖、重叠、资源互斥和拓扑怎样影响完成时间？ | [执行理论](execution/README.md) | 按真实关键路径判断价值 |
| 05 内存与状态 | 容量、流量、驻留和复用如何相互影响？ | [状态理论](memory/README.md) · [缓存专项](foundations/cache-context-ceilings.md) | 防止减少时延却挤占并发容量 |
| 06 生成效率 | 每轮提交多少有效 token，代价多大？ | [生成理论](generation/README.md) | 选择投机深度、验证与回退机制 |
| 07 推理服务 | 到达、动态 batch、缓存和 SLO 怎样决定收益？ | [服务理论](serving/README.md) | 从局部优化回到端到端目标 |
| 08 优化机制 | 优化改变了哪一种物理量和可行域？ | [机制与表示](optimization/mechanisms.md) | 生成机制不同的候选 |
| 09 收益预测 | 从局部变化怎样估计整体效应？ | [收益预测](optimization/benefit-prediction.md) | 历史场景实例，注意 Qwen/H800 特定假设 |
| 10 耦合与全局优化 | 单项收益为何不能直接叠加？ | [耦合与 Pareto](optimization/coupling.md) | 组合实验、条件收益与顺序依赖 |
| 11 自主优化理论 | 怎样在有限预算下选择和验证干预？ | [Optimizer 总纲](../optimizer/foundations/autonomous-optimization.md) | 独立核心主线：规划、实现、验证与学习 |

## 从理论走向可计算模型

[资源模型](../models/resource/README.md) → [算子模型](../models/operator/README.md) → [执行图](../models/execution/README.md) → [服务模型](../models/serving/README.md)。各层通过 [统一契约](../models/contracts.md)交换量，并用 [校准与留出](../models/calibration/README.md)约束预测范围。

## 内容关系与项目使用

[全局图与六张子图](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/visualizations/diagrams/research-atlas/README.md)连接理论、模型、Agent、实验与项目实践；其中 [01 性能与诊断](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/visualizations/diagrams/research-atlas/01-theory.svg)解释性能形成，[02 机制与耦合](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/visualizations/diagrams/research-atlas/02-mechanisms.svg)解释干预怎样传播。[六大理论知识树](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/visualizations/diagrams/llm-inference-theory-mindmap.md)继续提供更细的概念索引。

带着已有优化尝试阅读时，先用[项目指导](../docs/project-optimization-guide.md)选择需要回答的理论问题。[符号与单位](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/docs/notation.md)和[内容地图](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/docs/content-map.md)用于检查约定与当前能力。

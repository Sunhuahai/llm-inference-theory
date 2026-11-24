# 理论体系索引

从[图解导读](README.md)进入，再阅读[完整理论总纲](foundations/unified-theory.md)。下列章节为研究结构，不代表每章已经有独立完稿。

| 章节 | 研究问题 | 当前资料 |
| --- | --- | --- |
| 01 推理资源成本 | FLOPs、访存、通信、容量、串行依赖 | [总纲](foundations/unified-theory.md) |
| 02 性能边界 | Roofline、I/O、关键路径 | [导读](README.md) |
| 03 执行理论 | Amdahl、算子效率、重叠 | [导读](README.md) |
| 04 状态与内存 | 状态容量、前缀复用与上下文 | [专项推导](foundations/cache-context-ceilings.md) |
| 05 生成理论 | 自回归、投机解码与有效生成 | [总纲](foundations/unified-theory.md) |
| 06 并行与通信 | TP/EP/PP/DP、通信成本 | [总纲](foundations/unified-theory.md) |
| 07 推理服务 | 排队、动态批处理、SLO | [服务研究](serving/README.md) |
| 08 优化理论 | 边际收益、瓶颈迁移、交互 | [收益预测](optimization/benefit-prediction.md) |
| 09 全局优化 | 可行域、Pareto、约束优化 | [优化研究](optimization/README.md) |
| 10 自主优化 | 理论约束搜索、反馈与学习 | [自主优化](../optimizer/README.md) |

[全景图](../visualizations/diagrams/llm-inference-theory-mindmap.md) · [符号体系](../docs/notation.md)

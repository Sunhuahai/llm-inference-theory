# Performance Models

将资源需求、性能边界、关键路径及服务成本表达为可计算模型。

| 组件 | 职责 | 状态 |
| --- | --- | --- |
| resource-model | 算子 FLOPs、访存、通信、状态容量 | 规划 |
| bound-model | 资源时间下界与瓶颈区域 | 规划 |
| latency-model | TTFT、TPOT 预测 | 初步预测器 |
| throughput-model | 服务吞吐与容量 | 初步预测器 |
| benefit-model | 干预前后收益 | 初步预测器 |
| coupling-model | 多优化交互、瓶颈迁移 | 研究计划 |
| queueing-model | 动态负载、SLO 与尾延迟 | 研究计划 |

现有 [predictor/](predictor/README.md) 原样迁移，含 Python CLI、schema、合成示例与测试；不是完整通用性能建模库。先复用和校准，再按模块逐步拆分。未实现的组件不伪装为可运行代码。

[业界建模工具](../research/industry/infersim-aiconfigurator.md) · [验证方案](../experiments/README.md)

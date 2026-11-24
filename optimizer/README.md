# Theory-Guided Autonomous Optimization

研究目标：在相同质量、GPU 与 Agent 预算下，提高优化搜索效率，而不是单纯延长 Coding Agent 的运行时间。

| 组件 | 职责 |
| --- | --- |
| profiler | 捕获真实服务热点与硬件计数器 |
| theory engine | 建立资源账本、性能边界与瓶颈假设 |
| planner / search | 生成机制不同的候选、预测收益、预算感知搜索 |
| coding agent | 独立工作区实现候选 |
| evaluator | 固定正确性 oracle、微基准、端到端 A/B |
| memory | 保存正负结果、失败原因与复现实验 |

当前为设计阶段；没有已实现的通用自主优化器。下一步先固定实验控制层，再实现理论引导的候选选择和独立验证。比较普通 Codex、传统搜索、理论引导搜索的同预算结果。

[自主优化图解导读](../research/industry/agentic-inference-optimization-guide.md) · [业界自主优化调研](../research/industry/ai-driven-inference-optimization.md) · [研究路线](../docs/roadmap.md)

# 知识路由与版本选择

## 结构化路由（v2）

先用 catalog 搜索项目现象/ID/tag；lookup 获取输入输出、适用域、先修闭包与 context 参数。按 [六层总索引](../knowledge/structured/INDEX.md) 和 [机器图谱](../knowledge/structured/dependencies.json) 选择模型，而非加载全部长文。例：Roofline → FND-ROOFLINE；算子 → PERF-OPERATOR；状态 → PERF-STATE；端到端 → PERF-SERVING；优化耦合 → OPT-COUPLING；Agent 诊断 → AGT-DIAGNOSIS；校准 → LRN-CALIBRATION。先用 active 版本，不可用 seed 冒充 active。详见 [结构化理论说明](structured-theory.md)。

下面的原始文档路由用于补充详细推导。

每次先用 resolve 获得 active documents 目录与版本；下表的路径相对该目录，未初始化时对应 [初始理论快照](../knowledge/INDEX.md)。不要机械读取所有材料。阅读当前版本中涉及这些路径的本地修订、经验律与适用范围，必要时再读原仓库的固定版本。active 版本的 INDEX.md 由程序生成，包含后来加入的全部文档。

| 项目疑问 | 首选理论与模型（链接是 seed） | 要作出的判断 |
| --- | --- | --- |
| 不明确慢在哪里、计数器有歧义 | [诊断理论](../knowledge/initial/theory/diagnosis/README.md)、[Agent 诊断](../knowledge/initial/optimizer/diagnosis/README.md) | 可辨识性、竞争假设、最小区分实验 |
| 性能极限与剩余机会 | [资源边界](../knowledge/initial/theory/performance/README.md)、[资源模型](../knowledge/initial/models/resource/README.md)、[统一理论](../knowledge/initial/theory/foundations/unified-theory.md) | 资源/依赖严格界和有效速率预测的区别 |
| 局部快整体不快、并行扩展差 | [执行与并行](../knowledge/initial/theory/execution/README.md)、[执行图模型](../knowledge/initial/models/execution/README.md) | 暴露关键路径、通信、资源争用和 Amdahl |
| 算子形状、tile、融合 | [算子模型](../knowledge/initial/models/operator/README.md)、[微基准](../knowledge/initial/experiments/microbenchmarks/README.md) | 形状依赖、寄存器/spill、新增开销 |
| KV、状态、缓存、长上下文 | [内存理论](../knowledge/initial/theory/memory/README.md)、[专项推导](../knowledge/initial/theory/foundations/cache-context-ceilings.md) | 容量、命中收益、恢复成本 |
| 投机解码、MTP | [生成理论](../knowledge/initial/theory/generation/README.md) | 有效提交量/轮与完整周期成本 |
| TTFT、TPOT、吞吐、p99、Goodput | [服务理论](../knowledge/initial/theory/serving/README.md)、[服务模型](../knowledge/initial/models/serving/README.md)、[模拟](../knowledge/initial/models/simulation/README.md) | 动态负载、队列、SLO、完整系统 |
| 候选机制和收益组合 | [优化机制](../knowledge/initial/theory/optimization/mechanisms.md)、[耦合](../knowledge/initial/theory/optimization/coupling.md)、[收益预测](../knowledge/initial/theory/optimization/benefit-prediction.md) | 净成本、交互和迁移；历史案例不代表当前模型 |
| 预测长期不准 | [统一模型契约](../knowledge/initial/models/contracts.md)、[校准与不确定性](../knowledge/initial/models/calibration/README.md)、[模型验证](../knowledge/initial/experiments/model-validation/README.md) | 判断结构缺项、域外外推、参数可辨识性 |
| 候选过多、预算有限 | [规划与搜索](../knowledge/initial/optimizer/planning/README.md)、[自主优化理论](../knowledge/initial/optimizer/foundations/autonomous-optimization.md) | 信息价值、成本、排序、停止 |
| 旧成功或失败能否复用 | [经验记忆](../knowledge/initial/optimizer/memory/README.md)、[代码进化](../knowledge/initial/optimizer/evolution/README.md)、[案例](../knowledge/initial/experiments/case-studies/README.md) | parent 条件、迁移域、再校准 |
| 评价 Agent 方法 | [独立评价](../knowledge/initial/optimizer/evaluation/README.md)、[Agent Benchmark](../knowledge/initial/experiments/agent-benchmark/README.md) | 同预算、同权限、同工具的比较 |

每条建议必须指出项目证据、读到的版本和理论章节以及如何使用，而不是列文献清单。对被本地修订的章节，需同时读新文档的适用范围；若 context 不匹配，退回通用机制或重新校准。若需完整图谱，见[上游总图](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/visualizations/diagrams/research-atlas/README.md)。离线仅保证核心文字知识，额外图和运行代码链接可能需要上游访问。

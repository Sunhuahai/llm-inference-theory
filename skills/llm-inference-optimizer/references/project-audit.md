# 项目审计

首先遵循项目自身维护规则。参考[项目优化指导](../knowledge/initial/docs/project-optimization-guide.md)、[测量协议](../knowledge/initial/docs/benchmark-protocol.md)和[实验模板](../knowledge/initial/experiments/templates/README.md)。

1. 从真实配置、源码、Git 提交、Profiler、日志中查证 checkpoint、dtype、推理语义、设备数与拓扑、软件版本、输入/输出长度、请求到达、缓存冷热、并发与 SLO。识别哪些是固定约束、哪些可调整。
2. 固定原始基线；记录当前独立验证的最佳版本。每个候选绑定其 parent、实际生效的修改和实现标识。标明是否有回归、编译失败、超时、不可比；不要把它们混成无收益。
3. 维护历史证据台账，使用 [历史台账模板](../assets/evidence-ledger.json)；可恢复数据就直接恢复，未知保留 null/unknown。已有结果无事前预测仅属回顾性证据，不得伪造历史冻结，不进入逐轮预测误差统计。
4. 判断可比性。相同任务、质量、总资源、测量定义和负载是前提；若 batch/并行/调度就是决策变量，比较相同任务分布和预算下的完整配置。计时口径或条件不同必须标 conditional/noncomparable。
5. 输出事实/解释/替代解释/未观测量；用历史变化定位资源、暴露时间、有效生成量、调度和容量的剩余机会。优先恢复会改变决策的关键缺口。

项目 profile 使用 [模板](../assets/project-profile.json)。context_id 应由项目的稳定环境、工作负载、指标定义和关键配置身份生成或明确登记（例如有版本的 canonical context JSON 的 SHA-256）；不可仅用 GPU 型号，也不得为了复用数据修改旧身份。变化后新建 context，记录新旧关系。用户级拟合仅在同 context 和定义下执行，跨 context 证据需明确迁移域。

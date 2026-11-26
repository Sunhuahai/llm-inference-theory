# 内容地图与能力状态


**结构化理论库 v2**：[六层模型索引](../skills/llm-inference-optimizer/knowledge/structured/INDEX.md) · [理论图谱](../skills/llm-inference-optimizer/knowledge/structured/ATLAS.md) · [Skill 图解介绍](../skills/llm-inference-optimizer/INTRODUCTION.md) · [使用与演化](../skills/llm-inference-optimizer/references/structured-theory.md)。在保留 37 篇来源文档的基础上，提供 33 个结构化理论单元、16 个窄解析函数、模型依赖及 context 参数契约；数值计算不代表实测。

> 2026-11-25。这里对应仓库结构审计提出的缺项，区分已补充的研究内容、可执行参考以及仍需外部实验的能力。

需要先建立整体认识时，打开[全局关系总图与六张子图](../visualizations/diagrams/research-atlas/README.md)。已有具体项目和历史结果时，从[项目优化指导](project-optimization-guide.md)及[可复制提示词](prompts/review-project.md)进入，再按问题回到本页相应章节。

## 1. 三条核心主线

| 领域 | 已补充的内容 | 主入口 | 当前实现与证据 |
| --- | --- | --- | --- |
| 瓶颈诊断 | 不可辨识性、资源敏感性、关键路径反事实、区分性实验 | [诊断理论](../theory/diagnosis/README.md) | 条件推导与合成算例 |
| 性能边界 | 多资源下界、可达性、batch/上下文相图与容量域 | [性能理论](../theory/performance/README.md) | 解析参考可计算窄模型 |
| 执行、状态与生成 | 并行/通信、状态驻留、投机产出与边际收益 | [执行](../theory/execution/README.md) · [内存](../theory/memory/README.md) · [生成](../theory/generation/README.md) | 独立理论专题，按架构与假设使用 |
| 服务理论 | 联合负载分布、动态 batch、P/D、goodput 与尾部 | [Serving](../theory/serving/README.md) | 理论与建模设计；未实现通用服务模拟 |
| 优化策略空间 | 资源变换、前提、资源交换、依赖与冲突 | [机制](../theory/optimization/mechanisms.md) · [候选契约](../optimizer/search-space/README.md) | 人可读/机器可读的表示设计 |
| 优化耦合 | 四组交互、条件收益、顺序依赖与 Pareto | [耦合理论](../theory/optimization/coupling.md) | 条件公式、手算例与实验设计 |
| 统一性能模型 | 资源、算子、执行、服务、收益的接口 | [Models](../models/README.md) · [契约](../models/contracts.md) | 新解析参考与旧场景预测器；尚非完整通用模型库 |
| 校准与不确定性 | 数据分割、可辨识性、误差传播、外推边界 | [Calibration](../models/calibration/README.md) | 方法与验证协议；待独立实测数据 |
| 仿真 | 到达、调度、执行、缓存的状态转移 | [Simulation](../models/simulation/README.md) | 实现设计；现有精确键缓存回放范围有限 |
| Agent 问题定义 | 状态、动作、观测、质量与预算约束 | [自主优化总纲](../optimizer/foundations/autonomous-optimization.md) | 研究框架与可证伪问题 |
| 理论引导搜索 | 成本感知改进、探索、信息价值、多保真与停止 | [Planning](../optimizer/planning/README.md) | 算法设计；离线策略回放可运行 |
| 代码进化与经验 | 候选谱系、组合回归、负结果、适用域 | [Evolution](../optimizer/evolution/README.md) · [Memory](../optimizer/memory/README.md) | 契约与工作机制；未实现完整代码进化系统 |
| 多层优化与控制 | Kernel/Runtime/Serving 分层、控制器与 Agent 分工 | [Systems](../optimizer/systems/README.md) | CPU 有限候选回放；真实 adapter 待接入 |

## 2. 共同评价与研究基础设施

| 领域 | 交付 | 状态 |
| --- | --- | --- |
| 模型预测验证 | [独立工作点、干预差值、留出与误差](../experiments/model-validation/README.md) | 完整验证设计 |
| Agent benchmark | [对照/消融、预算曲线、成功率与搜索成本](../experiments/agent-benchmark/README.md) | 完整评价协议与合成回放入口；真实对照待执行 |
| 微基准 | [形状、计时、采集扰动与参数来源](../experiments/microbenchmarks/README.md) | 采集设计 |
| 跨环境案例 | [模型/硬件/负载轴与三种迁移强度](../experiments/case-studies/README.md) | 案例设计；历史证据独立保留 |
| 统一记录 | [环境、工作负载、trial JSON 模板](../experiments/templates/README.md) | 可复用模板，未知字段不伪填 |
| 全局关系图谱 | [全局总图与理论、机制、模型、Agent、证据、项目六张子图](../visualizations/diagrams/research-atlas/README.md) | 七页 draw.io、七张 SVG、节点文档链接与统一生成脚本 |
| 项目使用指导 | [历史证据复盘、问题驱动选读、下一轮方向与验证](project-optimization-guide.md) · [Agent 提示词](prompts/review-project.md) | 可直接用于已有项目分析；具体建议需由项目证据支持 |
| 自演化 Agent Skill | [安装与调用](../skills/llm-inference-optimizer/README.md) · [逐轮预测](../skills/llm-inference-optimizer/references/prediction-comparison.md) · [理论演化](../skills/llm-inference-optimizer/references/theory-evolution.md) | 37 篇初始理论、CPU 比较/证据/拟合/版本治理；真实实验由使用者授权并验证 |
| 研究路线 | [共同里程碑与验收要求](roadmap.md) | 三条主线并行推进 |
| 低部署成本机制 | [原生功能与实验指南](../research/industry/low-deployment-cost-inference-optimization.md) | 官方文档核对与条件式实验建议；未进行本地兼容性或 GPU 验证 |
| 外部自主优化 Skills | [资源地图与使用指南](../research/industry/inference-optimization-skills.md) | 公开工具调研与集成建议；未执行 GPU 或安装验证 |
| 文献入口 | [问题导向的一手阅读地图](../research/papers/README.md) | 基础与评价资料；保留现有业界专题 |

## 3. 可执行内容的边界

**已有场景预测器**处理输入给定的局部改进、重叠、吞吐需求代理及顺序缓存 trace。**新解析参考**计算有明确资源/速率语义的资源时间与有限 DAG。**新搜索回放**比较预先给定候选的选择与预算控制。它们是相互关联的研究参考，不是已经自动从任意 checkpoint 提取全图、调用 Agent 写代码并在 GPU 部署的完整系统。

文档补齐的完成条件是：问题、假设、推导、具体例子、反例、接口和验证方式齐备。真实能力的完成条件另行登记：GPU adapter、校准数据库、动态服务模拟、真实 Agent 搜索、跨模型/硬件独立验证仍在 [路线图](roadmap.md)中，不因新建文档而标记完成。

## 4. 建议阅读顺序

入门先读 [图解导读](../theory/README.md) 与 [自主优化导读](../research/industry/agentic-inference-optimization-guide.md)。研究主线按“诊断 → 机制 → 模型预测 → 搜索 → 独立评价 → 误差与经验更新”串联；需要推导时回到 [理论索引](../theory/INDEX.md)，需要运行时看 [合成示例](../experiments/synthetic/README.md)。

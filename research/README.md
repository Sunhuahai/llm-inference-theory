# 文献与业界研究

按研究问题组织外部证据：来源解决什么问题、采用什么模型与评价、在哪些条件下得到结论，以及对本仓库能提供什么启发。

文献判断如何连接理论假设、模型与独立验证，见[全局关系图](../visualizations/diagrams/research-atlas/README.md)；用于具体项目时，按[项目指南](../docs/project-optimization-guide.md)核对外部方法的适用条件与当前已有证据。

| 入口 | 重点 |
| --- | --- |
| [一手阅读地图](papers/README.md) | 资源边界、服务建模、程序搜索与 Agent 评价的基础论文 |
| [AI 如何自动优化 AI 推理](industry/agentic-inference-optimization-guide.md) | 从直觉出发理解 KernelEvolve、KernelAgent、GEAK 与 SEIS |
| [推理优化 Skills 与开源工具](industry/inference-optimization-skills.md) | Sybil、BBuf、NVIDIA、KDA 等工具的选择、使用与集成边界 |
| [自主优化深度调研](industry/ai-driven-inference-optimization.md) | 项目机制、公开证据、版本与适用边界 |
| [InferSim 与 AIConfigurator](industry/infersim-aiconfigurator.md) | 性能估计、部署搜索与工具使用 |
| [开放问题](open-problems/README.md) | 当前未解决问题、可证伪假设和最小对照 |

本仓库自己的自主优化问题定义、搜索算法与系统设计放在 [optimizer/](../optimizer/README.md)；理论关系放在 [theory/](../theory/INDEX.md)，可计算模型放在 [models/](../models/README.md)。外部综述与本仓库设计相互引用，不复制维护两份主文档。

每篇报告记录一手来源、版本与核对日期。项目原作者报告、我们的合成推导、用户反馈及独立实测分开，不把其他模型或硬件上的加速数字转成当前环境的承诺。

# 研究图谱与可视化

## 全局关系与六张子图

[![推理优化研究全景与跨模块反馈](diagrams/research-atlas/00-overview.svg)](diagrams/research-atlas/00-overview.svg)

[图集说明与阅读路线](diagrams/research-atlas/README.md) · [七页可编辑 draw.io](diagrams/research-atlas/llm-inference-research-atlas.drawio) · [全局高清 SVG](diagrams/research-atlas/00-overview.svg)

| 子图 | 重点 |
| --- | --- |
| [01 性能与诊断](diagrams/research-atlas/01-theory.svg) | 工作量、资源、依赖、服务与可辨识性 |
| [02 机制与耦合](diagrams/research-atlas/02-mechanisms.svg) | 资源交换、新增代价、条件收益与组合 |
| [03 性能模型](diagrams/research-atlas/03-models.svg) | 下界与预测、执行与服务、校准和验证 |
| [04 Agent 自主优化](diagrams/research-atlas/04-agent.svg) | 决策闭环、测量动作、独立评价和经验 |
| [05 实验与证据](diagrams/research-atlas/05-evidence.svg) | 三条并行评价路径、预算、留出与反馈 |
| [06 项目使用](diagrams/research-atlas/06-project.svg) | 从已有尝试形成后续优化方向 |

具体项目的使用方法见[项目优化指导](../docs/project-optimization-guide.md)，可直接使用[项目 Agent 提示词](../docs/prompts/review-project.md)。图中节点链接对应文章，文字文档也反向链接相关图与前后章节。

## 底层理论与既有专题

| 图谱 | 表达什么 |
| --- | --- |
| [三条主线与自主优化简图](diagrams/research-framework/README.md) | 简洁解释研究分工和反馈闭环 |
| [六大理论全景思维导图](diagrams/llm-inference-theory-mindmap.md) | 成本、计算访存、状态、生成、通信、服务的细分关系 |
| [推理系统全局关系](diagrams/llm-inference-theory-overview.md) | 请求、执行、状态与硬件的相互约束 |
| [理论图解导读](../theory/README.md) | Roofline、Amdahl、关键路径与排队直觉 |
| [自主优化业界导读](../research/industry/agentic-inference-optimization-guide.md) | 已有八张 SVG 与外部研究的核心思想 |
| [历史技术图](diagrams/README.md) | Qwen/H800 特定研究的来源与范围 |

所有新图是结构/机制示意，没有真实性能曲线。SVG 供 GitHub 直接预览；draw.io 及生成脚本保持可编辑。模型相图及数据图必须标注输入、假设和证据来源。

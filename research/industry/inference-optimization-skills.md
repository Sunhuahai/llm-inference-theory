# AI 推理优化 Skills 与开源工具：资源地图及使用指南

> 2026-11-26 · 业界工具调研 · v1.0  
> 面向 Codex、Claude Code 等 Coding Agent，重点讨论如何使用已有 Skill、实验工具及自主优化框架，而不是再收集一批具体优化 Tricks。  
> 公开资料核对日期：2026-11-26。下文外部仓库链接多为滚动分支；除明确注明的版本外，未锁定全部 commit。部署或复现实验前应固定实际 revision。本次未安装第三方软件、执行 GPU 实验或测量第三方 Skill 的优化效果。

## 1. 为什么值得研究这些项目？

一个理论驱动的自主优化系统包含几种能力：明确目标与约束、测量真实瓶颈、提出可检验的干预、在隔离环境实现、独立评价、保存成功和失败证据，并用新证据修正后续判断。

业界已经为其中不少环节提供了专门工具。我们没有必要从零编写所有 Benchmark、Nsight 分析器和 Kernel 知识库，更值得研究的是如何把这些能力连接到统一的性能理论、收益预测和跨项目知识演化。

注意四类资源的区别：

| 类别 | 提供的能力 | 不应直接假定 |
| --- | --- | --- |
| Agent Skill | 结构化指令、参考知识和工具调用流程 | 单靠安装就自动具备 GPU 访问或稳定的优化能力 |
| 专用工具 | Benchmark、Profiler、报告解析、配置搜索 | 会自主提出跨层优化算法 |
| 工程知识库 | 硬件机制、历史 PR、失败经验 | 历史结论在当前硬件和负载上必然成立 |
| 自主优化框架 | 候选搜索、代码生成、验证与迭代 | 适用于任何模型、后端和 GPU，或必然超过专家 |

理解其关系的入门资料见 [AI 如何自动优化 AI 推理](agentic-inference-optimization-guide.md)；研究项目及实验结果见[自主优化深度调研](ai-driven-inference-optimization.md)。

## 2. 项目总览

优先级表示对本仓库及固定精度推理优化研究的借鉴价值，不是跨项目的客观性能排名。

| 项目 | 主要定位 | 最值得借鉴 | 优先级 |
| --- | --- | --- | --- |
| [Sybil inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill) | 推理服务迭代优化 Skill | 性能天花板、交错对照、最佳配置和实验账本 | A |
| [BBuf AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) | AI Infra 工程 Skill 集合 | 跨框架 Benchmark、模型算子分析与历史 PR | A |
| [TensorRT-LLM perf-optimization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization/SKILL.md) | NVIDIA 推理框架优化协调 Skill | 专家分工、优化排序、回滚与组合影响 | A |
| [NVlabs KDA](https://github.com/NVlabs/kda) | Kernel Design Agents 工作流 | 从任务定义到实现、测试和候选管理 | A |
| [BBuf KDA-Pilot](https://github.com/BBuf/KDA-Pilot) | 基于真实框架算子的 Kernel 实验 | 生产形状、对称评价、证据可追溯 | A |
| [KernelFlow cuda-optimized-skill](https://github.com/KernelFlow-ops/cuda-optimized-skill) | CUDA/CUTLASS/Triton 迭代优化 Skill | 硬件门槛、性能归因、迭代状态 | B |
| [NVIDIA/skills](https://github.com/NVIDIA/skills) | NVIDIA 官方通用 Agent Skills | 依具体 GPU 开发任务选择工具 | B |
| [MIT HAN Lab KernelWiki](https://github.com/mit-han-lab/KernelWiki) | Kernel 优化知识库 | Hopper/Blackwell 硬件知识、上游技术证据 | B |
| [MIT HAN Lab ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) | Nsight Compute 报告分析 Skill | 性能计数器解释、根因假设、报告规范 | B |
| [tensormux kernel-skills](https://github.com/tensormux/kernel-skills) | CUDA/Triton 编程与调试知识库 | 正确性、边界条件和编程流程 | B |
| [AMD GEAK](https://github.com/AMD-AGI/GEAK) | 端到端自动优化框架 | 服务层与算子层的嵌套优化、独立验证 | A（架构参考） |
| [PyTorch KernelAgent](https://github.com/meta-pytorch/KernelAgent) | 硬件引导的算子优化 Agent | Profiling 与分支搜索 | B |
| [SEIS](https://github.com/nehzux/SEIS) | 推理引擎持续演化研究 | 跨轮经验继承、代码演化 | B |
| [AutoKernel](https://github.com/RightNow-AI/autokernel) | 模型热点自动发现与 Kernel 优化 | 轻量级优化控制流程 | B |

其中 A 级项目建议优先阅读，B 级在实际需要相应功能时选用。GEAK、KernelAgent、SEIS、AutoKernel 的研究背景和实验结果已在[此前的专题调研](ai-driven-inference-optimization.md)中展开，本篇不重复维护全部数值。

## 3. Sybil：与我们的整体优化 Skill 最接近的项目

**入口：** [仓库](https://github.com/sybil-solutions/inference-optimisation-skill) · [SKILL.md](https://github.com/sybil-solutions/inference-optimisation-skill/blob/main/skills/inference-optimisation/SKILL.md)。

### 3.1 它解决什么问题？

已经运行了一个 LLM 服务，希望在当前资源限制下提高 Decode、Prefill、TTFT 等性能。它要求先测量设备的有效带宽和算力，并建立基线，再根据理论天花板与现有表现之间的差距选择候选。

其核心过程可以概括为：

~~~text
设备能力与工作负载
         ↓
计算条件性性能天花板
         ↓
基线与瓶颈分析
         ↓
选择一项候选，保存配置
         ↓
基线 → 候选 → 基线，交错测量
         ↓
验证性能与正确性，接受或拒绝
         ↓
记录最佳配置和实验历史，继续或停止
~~~

### 3.2 有哪些可用的工具？

| 组件 | 职责 |
| --- | --- |
| SKILL.md | 规定优化过程、优先级、停止和验收要求 |
| references/ceilings.md | 估计生成、Prefill、TTFT、启动及 KV 容量的边界 |
| scripts/ceiling.py | 从设备测量和模型成本输入计算条件性上限 |
| scripts/bench.py | 测量 OpenAI 兼容服务的性能及部分正确性、内存指标 |
| scripts/queue.sh | 串行组织基线与候选实验 |
| THEORY.md / PLAN.md | 工作区中记录模型假设、性能目标和优化历史 |
| ledger.jsonl / configs/ | 保存测量账本和候选运行配置 |

具体文件生成方式以所选 revision 为准；上表同时包含仓库工具及运行时产物。

作者 README 给出了对候选前后两次基线均进行比较、限制其他指标退化、长上下文检查和并发压力验证的工作流。其关键价值是减少把时钟、温度、缓存状态或后台负载变化误判为优化收益的机会。

### 3.3 如何开始使用？

在隔离的工作区克隆并审阅 Skill、脚本及 license，不需要为了阅读就启动服务。上游提供独立 CPU 天花板示例：

~~~bash
git clone https://github.com/sybil-solutions/inference-optimisation-skill.git
cd inference-optimisation-skill
S=skills/inference-optimisation
python3 "$S/scripts/ceiling.py" "$S/examples/ceiling-ornith-b390.json"
~~~

该示例是上游给出的特定设备配置，并非本项目 H800 实测。之后可以在 Codex 会话中指向实际 checkout 中的 SKILL.md。若计划调用 bench.py 或 queue.sh，先确认目标服务、启动/停止命令、测试负载、内存保护和 GPU 独占权限。

### 3.4 局限与对本仓库的意义

Sybil 的成本模型适合快速建立性能直觉，但使用实测有效带宽或固定算力利用率得到的是条件性估计，不自动构成严格物理上界。复杂 MoE、长上下文、高并发、多 GPU 通信和运行时依赖需要进一步建模。上游还讨论量化与投机生成等候选；我们的固定精度实验必须主动限制可选动作，不能照搬全部策略。

**最值得借鉴的不是再写一个相同的脚本，而是将其真实服务测量和交错 A/B 控制作为外部评价工具候选，用本仓库的理论模型补足解释、干预预测与适用域。**

## 4. BBuf：把 AI Infra 工程知识封装成 Skills

**入口：** [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)。README 标注 v0.1.5 包含模型 Day-0、Profiler 分层知识与 118 份双语模型历史档案；该数量是上游所列规模，不等于 118 个实测成功案例。

### 4.1 推荐能力

| Skill / 能力 | 主要用途 | 适用场景 |
| --- | --- | --- |
| llm-serving-auto-benchmark | 同口径部署与 Benchmark | 比较 SGLang、vLLM、TensorRT-LLM、TokenSpeed 的可行配置 |
| llm-serving-capacity-planner | 容量和资源规划 | 分析显存、上下文和可支持并发 |
| llm-torch-profiler-analysis | PyTorch 及算子性能分析 | 捕获关键路径、异常热点与候选融合机会 |
| llm-pipeline-analysis | 模型执行流水线分析 | 理解模型层及执行阶段的成本 |
| model-compute-simulation | 模型计算量与形状估计 | 建立模型资源账本 |
| model-pr-history-knowledge | 查询历史优化 PR | 确认上游是否已经实现某种机制及其适用条件 |

安装前从仓库 README 与实际 SKILL.md 核对具体目录、名字及依赖；版本变更可能导致路径变化。特别是 PR 知识库不仅记录成功方案，也整理实现位置和证据，部分新生成档案仍待人工审阅。历史检索不是 GPU 测量。

### 4.2 为什么历史 PR 知识有价值？

假设我们的模型 Attention 中存在一个重复的数据布局转换。理论指出它可能引入额外 HBM 访问。但在要求 Agent 从头编写新的 Kernel 之前，应该先了解：

- 上游近期是否已经融合该转换；
- 所用模型结构、dtype、shape 与 GPU 架构是否匹配；
- 新实现是否已进入项目当前依赖版本；
- 上游的测试与收益是否覆盖我们的目标工作负载。

BBuf 的知识库适合作为候选生成之前的工程检索层。其 README 提供查询方式，例如：

~~~bash
cd model-pr-optimization-history
python3 scripts/query.py --list
python3 scripts/query.py --framework vllm "qwen3 fused qk norm"
~~~

命令仅用于查询知识库。真正采用 patch 时还须核对原 PR、当前代码和本地环境。

### 4.3 与我们的 Skill 的关系

BBuf 强在真实框架和历史工程知识；我们的 Skill 强调结构化数学模型、干预收益预测、同条件比较以及版本化理论库。两者可以互补：先用理论确定需要调查的机制，再用工程知识确定实际可实施候选，最后用独立实验判断是否保留。

## 5. NVIDIA：官方 GPU Skills 与 TensorRT-LLM 优化协调

这里区分两个不同的入口。

### 5.1 NVIDIA/skills

[NVIDIA/skills](https://github.com/NVIDIA/skills) 是 NVIDIA 官方面向 Coding Agent 的多产品 Skill 集合，覆盖 CUDA 等开发工作流，也覆盖不直接属于本研究的机器人、仿真和其他产品任务。它是 GPU 工程工具入口，不应直接描述为完整的 LLM 自主优化器。

根据当前问题选择相应 Skill，而不是默认安装整个集合。具体兼容性、CUDA 版本和命令由该 Skill 的最新说明决定。

### 5.2 TensorRT-LLM 内置性能优化 Skills

更直接相关的是 [TensorRT-LLM 的 .claude/skills 目录](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills)。

| 文件 | 作用 / 核对状态 |
| --- | --- |
| [perf-optimization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization/SKILL.md) | 协调优化、按影响与风险排序、委托专家、验收回滚；本次读取成功 |
| [perf-nsight-compute-analysis](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-nsight-compute-analysis/SKILL.md) | GPU Kernel 的 Roofline、occupancy、访存与 stall 分析；本次读取成功 |
| perf-optimization-casebook | 上一轮调研中的案例知识入口；本次直达页面无法重新读取，采用前须核对当前目录 |
| perf-host-optimization | 上一轮调研中的主机运行时优化入口；本次直达页面无法重新读取，采用前须核对当前目录 |

协调 Skill 将决策和实现分开：

~~~text
优化协调者：确定目标、分析收益、选择下一步
        ↓
按问题委托具体专家（Profiling / Graph / Triton / CuTe）
        ↓
独立测量与正确性验证
        ↓
保留、回滚或继续；更新当前基线
~~~

其优先级规则考虑预期收益、修改风险、依赖关系以及组合影响。这与我们已有[优化耦合理论](../../theory/optimization/coupling.md)直接相关。不过，上游采用的具体回归阈值属于该工作流约定，不是适用于所有系统的统计学标准。

### 5.3 适用边界

TensorRT-LLM 的 Skills 依赖该仓库的代码结构、特定专家角色和运行工具，不能仅复制一个 SKILL.md 就假定所有功能都能被 Codex 自动调用。对 H800（Hopper），应核对 SM90 适用条件。特别是上游将 TileIR 优化定位到 Blackwell，不能直接套用至 H800；Triton、CuTe 等也需要按目标实现、编译器与数据类型检查支持。

我们可以借鉴其协调与证据原则，独立实现相同职责的 Codex 工作流；不需要复制整个 TensorRT-LLM Agent 编排。

## 6. KDA 与 KDA-Pilot：面向真实高性能算子

### 6.1 Kernel Design Agents

**入口：** [NVlabs/kda](https://github.com/NVlabs/kda) · [agent-flow.md](https://github.com/NVlabs/kda/blob/main/docs/agent-flow.md) · [项目主页](https://nvlabs.github.io/kda/)。

KDA 将高性能 Kernel 开发定义为一项有输入、参考实现、性能目标和独立评价的工程任务，而不是让 Agent 接到模糊指令后直接生成代码。工作流强调先理解任务、整理技术资料与设计、形成可执行计划，再以小步迭代完成实现与验证。

建议任务工作区保留设计与计划、候选输出、Profiling 文件、Benchmark 结果和候选谱系。上游 README 将其描述为早期研究原型，不是完整生产推理引擎。通用工作流可适配不同设备，官方社区 Kernel wishlist 当前主要面向 B200/B300，不能据此推断 H800 的现成适配或性能。

若需要配套知识库，优先使用 KDA 仓库记录的子模块 revision，并审查第三方许可说明。

### 6.2 KernelWiki 与 ncu-report-skill

| 工具 | 适合做什么 | 硬件边界 |
| --- | --- | --- |
| [KernelWiki](https://github.com/mit-han-lab/KernelWiki) | 检索架构特性、Kernel 设计模式、上游技术来源 | 包含 Hopper/Blackwell，Blackwell 优先；不覆盖整个分布式推理系统 |
| [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) | 组织 NCU 采集、报告解析、根因诊断和优化建议 | 公开说明主要针对 B200/B300、SM100/103；迁移到 H800 时须重新验证指标、section 和分析规则 |

KernelWiki 是知识来源，不是自动 Benchmark 工具。NCU Skill 是诊断工具，不会仅凭一份报告证明因果关系，更不能直接代替服务级 Profiling。

### 6.3 BBuf KDA-Pilot：真实工作负载的评价范式

**入口：** [BBuf/KDA-Pilot](https://github.com/BBuf/KDA-Pilot) · [LLM 任务](https://github.com/BBuf/KDA-Pilot/tree/main/llm) · [Diffusion 任务](https://github.com/BBuf/KDA-Pilot/tree/main/diffusion)。

其 README 展示两类研究：SGLang diffusion 算子的独立优化，以及自回归 LLM 的真实服务 Profiling 与 Kernel 任务抽取。

值得直接参考的不是某个加速数字，而是任务结构：

~~~text
task/
  prompt.md       明确 Agent 的优化目标
  config.toml     固定编译和测量配置
  baseline/       可追溯的上游基线
  solution/       受评估的候选
  bench/          对称测量与正确性检查
  docs/           Profile、源码、运行及决策证据
~~~

作者强调基线与候选应通过一致接口测量，覆盖实际工作负载的多种形状，保留正确性和时间证据。部分公开 diffusion 结果来自 B200；LLM 计划涉及 B200/H200。模型、架构、基线或评价口径变化后必须重新测量。

对我们的推理研究而言，可以采用相同方式把一个真实热点从大型引擎提取为独立实验单元。只有独立 Kernel 验证与端到端服务评价都通过，才应将结果合入稳定基线。

## 7. 其他直接可用的 Skill 和辅助资源

### 7.1 cuda-optimized-skill

**入口：** [KernelFlow-ops/cuda-optimized-skill](https://github.com/KernelFlow-ops/cuda-optimized-skill)。

这是一套由 Coding Agent 驱动、脚本负责确定性步骤的 CUDA/CUTLASS/Triton Kernel 优化 Skill。公开 README 的 v2 说明包括根据计算、访存、延迟指标分配候选尝试预算，结合 SASS 检查和性能归因区分优化成功、无明显收益及实现失败。

它适合有独立 Python 参考实现的算子任务。GPU、CUDA 工具链、NCU 性能计数器访问权、正确性环境及必要编译依赖必须具备。若无权限或硬件不匹配，应记录阻塞，不应猜测计数器或自动扩大系统权限。

最值得借鉴：把候选是否正确实现、是否改善性能、是否符合预期机制分开记录。性能下降不必然否定理论，可能只是实现未达到所声称的资源变化。

### 7.2 kernel-skills

**入口：** [tensormux/kernel-skills](https://github.com/tensormux/kernel-skills)。

按 CUDA、Triton、推理算子、测试模式等组织的 Skill 知识库。其价值在于为 Agent 提供 GEMM、归约、Softmax、归一化、RoPE、KV 追加等任务的实现注意事项，要求正确处理边界条件、数值稳定性和性能验证。

可以直接读取匹配任务的 SKILL.md；项目另提供 CLI/registry。它不提供真实 GPU 运行时或独立的 Benchmark 执行器，不能仅凭知识文件宣称完成自动优化。涉及量化的内容不属于本仓库固定精度实验的可选干预。

## 8. 自主优化框架：适合做系统级参照

| 仓库 | 适合参考的部分 | 当前接入判断 |
| --- | --- | --- |
| [AMD GEAK](https://github.com/AMD-AGI/GEAK) | 热点筛选、两层优化、控制器与独立验证 | ROCm 实现不直接用于 NVIDIA；迁移组织架构 |
| [PyTorch KernelAgent](https://github.com/meta-pytorch/KernelAgent) | Nsight 指导诊断、多候选实现与测试 | 选独立 Triton 任务，先核对目标硬件和工具链 |
| [SEIS](https://github.com/nehzux/SEIS) | 跨会话保留优化经验并持续演化引擎 | 研究候选谱系与知识继承；不能直接外推小模型结果 |
| [AutoKernel](https://github.com/RightNow-AI/autokernel) | 模型热点发现、局部优化循环 | 参考轻量控制器；真实 serving 还要独立验证 |

这些入口及实验边界见[完整业界调研](ai-driven-inference-optimization.md)；此处仅从如何选择现有工具的角度归纳，不重新进行性能测量。

## 9. 与本仓库理论驱动 Skill 的能力对比

本仓库当前的 [LLM Inference Optimizer Skill](../../skills/llm-inference-optimizer/README.md) 已经包含结构化理论库 v2、项目审计、模型查询、部分解析计算、事前冻结预测、事后比较及私有理论库演化接口。根据当前 README，已有 33 个结构化理论单元、16 个窄解析函数和 37 篇来源快照。它尚未内置通用真实 GPU adapter、自动 Profiler 或完整自主优化引擎。

不需要重新建设所有外部功能。建议明确以下能力分工：

| 研究环节 | 现有外部资源 | 本仓库负责的增量 |
| --- | --- | --- |
| 项目审计与历史检索 | BBuf PR 知识库、KernelWiki | 检验可比性、映射到理论机制及适用域 |
| 系统和算子测量 | Sybil、TensorRT-LLM NCU Skills、KDA-Pilot | 将观测转换为统一参数及证据契约 |
| 性能边界 | Sybil ceiling.py、已有性能建模工具 | 区分严格下界与校准估计，组合多层资源约束 |
| 候选生成与实现 | NVIDIA 专家 Skills、KDA、KernelFlow、kernel-skills | 用瓶颈假设和条件收益确定值得实施的方向 |
| 实验选择 | Sybil / GEAK / KernelAgent 的工作流 | 比较收益、不确定性、信息价值与预算 |
| 端到端验证 | Sybil A/B、KDA-Pilot 对称评价、GEAK | 固定语义、精度和负载；事前预测与独立测量比较 |
| 经验继承 | SEIS、KDA、历史 PR | 结构化理论版本、参数适用域、独立留出与审阅 |

关系可概括为：

~~~text
本仓库：理论与可计算模型 → 预估价值、约束候选
                          ↓
外部工程能力：历史检索 / Profiling / 实现 / Benchmark
                          ↓
独立证据：实测、失败记录、环境和原始结果
                          ↓
本仓库：预测误差 → 适用域与参数更新 → 下一轮决策
~~~

注意这是一种建议的集成关系，并不是已经完成的工具调用链。不能把不同项目的功能简单拼接就宣称实现了自主优化。

## 10. 实际使用建议：从一个小任务开始

### 10.1 先读什么？

优先精读 Sybil 的 SKILL.md 和测量方法、NVIDIA 的 perf-optimization，以及 BBuf 工程知识库。它们分别回答如何衡量收益、如何组织决策、如何查找已有技术。随后用 KDA-Pilot 学习真实算子任务的评价合同，再选适合目标硬件的 NCU 或 Kernel 实现 Skill。

### 10.2 一个实际的固定精度 H800 任务

假设当前已有推理引擎及真实性能记录，但目标瓶颈仍未确认。

| 阶段 | 行动 | 工具候选 |
| --- | --- | --- |
| 只读审计 | 固定模型 revision、全部精度、GPU 拓扑、引擎、负载及既往实验 | 本仓库 Skill；BBuf PR 知识 |
| 性能诊断 | 先观察 Prefill / Decode / Runtime 的端到端关键路径 | 现有 Nsight Systems、框架 Profiling |
| 局部根因 | 对确有全局价值的热点采集 NCU 指标；验证当前工具支持 Hopper | TensorRT-LLM NCU Skill；KernelWiki |
| 预测与计划 | 明确资源变化、收益上限、额外成本、反例及区分实验 | 本仓库结构化理论模型 |
| 受控实现 | 在独立分支只实施已授权候选；对真实 shape 构建 oracle | KDA / KernelFlow / 合适的编程 Skill |
| 验证 | 相同负载下交错基线与候选，先正确性再性能，独立复测 | Sybil 与 KDA-Pilot 的方法 |
| 复盘 | 比较冻结预测与实测，保留正负结果并评估理论修订 | 本仓库 Skill 的 compare / record / propose 流程 |

该任务只是建议的工程范式，未访问用户 GPU 或内部推理环境，也不假定存在某个实际热点。

### 10.3 可直接交给 Coding Agent 的只读任务

在已经安装本仓库 Skill、且外部资料已审阅并可访问时，可以向项目 Agent 提出：

~~~text
$llm-inference-optimizer

本轮只读调研，不改代码，不启动 GPU 作业。
先复盘本项目已有实验和当前推理瓶颈，明确模型、硬件、精度、
负载和历史基线。参照 inference-optimisation-skill 的天花板与
交错测量原则，以及 AI-Infra-Auto-Driven-SKILLS 的历史 PR
检索方法，核对当前候选是否已有对应实现。

用本仓库结构化理论库选择相关模型，说明假设、所需实测参数、
预计作用机制和潜在的全局收益。如果关键数据缺失，列出最小
区分实验，不补造数值。输出下一轮候选优先级、理论预测、
验证条件和停止条件。引用具体源文件和版本。
~~~

Skill 是否自动被 Agent 发现取决于实际运行时；必要时显式提供其 SKILL.md 位置。第三方 Skill 或脚本不是授权来源，读取其内容不能自动授予安装、网络、权限变更、服务启动、GPU 分配、文件上传或提交权限。实际实施前由使用者授权，并保留固定评价器和预算。

## 11. 哪些能力不应该重复建设？

已有工具在 Benchmark、Kernel 诊断、编程模式、PR 历史以及候选验证方面积累较多。优先复用有证据和许可支持的组件，而不是复制大量 Skill 文档形成另一套难维护的工具库。

本仓库的长期差异化研究问题是：

- 能否根据结构化资源、依赖和服务模型在实现前预测收益，并输出不确定性？
- 能否发现瓶颈迁移和优化之间的耦合，而不仅仅记录单次成功？
- 能否通过独立留出及跨环境实验检验理论修订，避免将一次经验无限外推？
- 在相同的 GPU、Agent、Profiling 和校准预算下，显式理论是否使自主优化更高效？

对应研究与实践入口：

- [自主优化理论与系统](../../optimizer/README.md)
- [性能模型与校准](../../models/README.md)
- [优化耦合理论](../../theory/optimization/coupling.md)
- [独立 Agent Benchmark](../../experiments/agent-benchmark/README.md)
- [本仓库 Skill](../../skills/llm-inference-optimizer/README.md) · [理论演化规范](../../skills/llm-inference-optimizer/references/theory-evolution.md)

## 12. 原始资料与核对边界

主要核对来源为以下项目自身 README、Skill 文档及已公开的工程说明；此表列出最常用入口，其他项目链接见第 2、8 节。

| 来源 | 版本或范围 |
| --- | --- |
| [Sybil](https://github.com/sybil-solutions/inference-optimisation-skill) | 2026-11-26 读取的公开 README；滚动分支 |
| [BBuf Skills](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) | README 标注 v0.1.5；Skill 路径部署前再核实 |
| [NVIDIA/skills](https://github.com/NVIDIA/skills) | 2026-11-26 读取的公开目录与 README |
| [TensorRT-LLM perf-optimization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization/SKILL.md) · [NCU Skill](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-nsight-compute-analysis/SKILL.md) | 2026-11-26 读取；casebook/host 直达页面未重新取得 |
| [KDA](https://github.com/NVlabs/kda) · [KDA-Pilot](https://github.com/BBuf/KDA-Pilot) | 2026-11-26 公开 README；任务数据与结果须按具体 revision 阅读 |
| [KernelWiki](https://github.com/mit-han-lab/KernelWiki) · [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) | 公开文档核对；硬件及工具版本见各自说明 |
| [KernelFlow](https://github.com/KernelFlow-ops/cuda-optimized-skill) | README 的 V2 工作流 |
| [tensormux](https://github.com/tensormux/kernel-skills) | 公开 README 的 Skill 与 registry 说明 |

本篇是资源选择与接入建议，不是 Skill 效果 Benchmark，也没有逐一锁定所有外部仓库 commit。GitHub Stars、动态排行榜及作者报告均不作为工具优劣的独立实验证据。正式集成前应固定上游 revision、审阅许可证及脚本权限，在独立环境复现；任何收益主张应回到固定模型与精度的同口径实测。

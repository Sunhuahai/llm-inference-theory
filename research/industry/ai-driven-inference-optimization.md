# AI 自动优化 AI 推理：业界实践与前沿研究

> 专题调研 · 2026-11-24 · v1.0  
> 研究范围：使用 LLM / Coding Agent 自动发现、实现、验证与组合推理性能优化，而不是使用 Agent 作为被优化的推理负载。  
> 本文衔接[推理优化理论框架](../../theory/README.md)与[完整理论总纲](../../theory/foundations/unified-theory.md)。所有外部性能数字均来自相应作者的报告或公开榜单，不是本仓库 H800 实测。

**建议初次阅读：** [AI 如何自动优化 AI 推理？（图解导读）](agentic-inference-optimization-guide.md)，先理解四项代表性工作的核心思想，再阅读本文的详细研究证据。

## 摘要

2026 年，AI 驱动推理优化已从生成单个 GPU Kernel，发展到基于硬件计数器的迭代搜索、真实推理服务中的热点定位与优化、整个推理引擎的持续演化。最值得关注的改变不是 Coding Agent 的代码生成速度，而是搜索、性能测量、正确性验证、独立复核和跨轮经验继承开始成为一个完整的性能工程系统。

现有公开证据表明，单算子级自动优化已具备较强的实用性；端到端服务优化出现工业案例；自主修改复杂推理引擎仍处于前沿探索阶段，稳定性与计算预算是核心限制。对我们的固定模型、固定精度研究而言，最有价值的路径不是让 Agent 无约束改写代码，而是用理论约束搜索空间，用独立 Benchmark 决定是否接受优化。

## 1. 研究版图：自动优化的四个层次

| 层次 | 决策对象 | 代表工作 | 当前证据 |
| --- | --- | --- | --- |
| L1 参数与已有策略 | batch、并行布局、后端、调度开关 | AIConfigurator、传统 autotuning、InferenceBench 搜索基线 | 较成熟，搜索空间有限 |
| L2 算子实现 | CUDA/Triton、分块、内存布局、融合 | KernelEvolve、KernelAgent、CUDA Agent、KernelPro、AutoKernel | 多项公开基准及部分生产结果 |
| L3 端到端服务 | 热点选择、配置、算子替换、服务验证 | AMD GEAK v4、FlashInfer-Bench | 已有真实服务案例 |
| L4 推理引擎演化 | runtime、调度、状态管理、生成算法 | SEIS、ISO-Bench、InferenceBench | 开放研究，复杂工程任务成功率仍有限 |

另外有两类常被混淆的技术：

- **编译器自动优化**：例如 TensorRT-LLM AutoDeploy，通过图变换、算子替换和运行时集成生成推理实现。它是重要的自动化基础，但不等同于 LLM Agent 自主提出新假设。[S12]
- **性能预测与配置搜索**：例如 InferSim、AIConfigurator，以解析模型和算子时延数据估计并搜索已有策略的性能。它们可以成为 Agent 的低成本评估工具，但不直接证明新算法或新实现有效。详见[已有工具专题](infersim-aiconfigurator.md)。

真正的 AI 自动优化系统，应当同时具备三个能力：提出可实施的优化候选、通过真实执行获得可靠的反馈、根据反馈自主调整后续搜索。

## 2. Meta KernelEvolve：将 Kernel 优化转化为结构化搜索

- 公开时间：2026-04-02；论文发表于 ISCA 2026。
- 机构：Meta。
- 入口：[官方技术文章](https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/) · [ISCA 论文](https://doi.org/10.1109/ISCA66397.2026.00063)。
- 定位：面向异构硬件和生产模型的 Kernel 生成及优化系统。

### 2.1 核心机制

KernelEvolve 不把一次代码生成视为最终结果，而将候选实现组织成搜索图。系统从硬件文档、已有优化知识和执行反馈中构建 LLM 上下文，选择候选节点进行变换、编译、验证和进一步扩展。

其核心思想可以抽象成：

~~~text
               硬件知识 / 优化知识
                       |
                 候选生成 LLM
                       |
                  初始实现 A
                /      |      \
            实现 B    实现 C    实现 D
             /  \       |         |
            E    F      G         H
                \       |        /
             编译 / 正确性 / 性能评估
                       |
          选择、继续探索、回退、重新开始
                       |
                 最优已验证实现
~~~

与普通的串行 Agent 不同，树搜索能同时保留多个技术方向。成功的优化模式可以提炼为后续任务的知识，失败的候选则为未来搜索提供负面证据。系统采用包括蒙特卡洛树搜索（MCTS）和进化策略在内的搜索方法，并依据历史上下文调节后续生成。[S1]

### 2.2 公开成果及边界

| 报告结果 | 证据范围 |
| --- | --- |
| KernelBench 250 项任务全部通过，且快于相应 PyTorch 参考实现 | 算子基准 |
| 160 个 ATen 算子在三种硬件平台上验证正确性 | 异构硬件泛化 |
| Andromeda Ads 模型在 NVIDIA GPU 上推理吞吐提升超过 60% | Meta 特定广告排序生产模型 |
| 将部分专家开发工作由数周缩短到数小时 | 作者报告的自动搜索案例 |

**注意：Andromeda 是推荐/广告排序模型，并非 Qwen 或自回归 LLM 服务。** 该结果证明这种方法具有实际工程价值，却不能直接推断 LLM TTFT、TPOT 会获得同样收益。公开技术文章与论文可供研究，但本次未确认完整内部生产搜索平台能够直接复用。

### 2.3 对我们的启发

不要让所有优化尝试都基于上一轮代码继续改。应保留稳定基线、并行探索若干机制，并独立评估候选。尤其当一种数据布局或融合策略已经达到收益饱和时，应允许从祖先节点重新分支，而不是持续微调。

## 3. PyTorch KernelAgent：将 Roofline 与硬件 Profiling 纳入 Agent

- 公开时间：2026-03-06（硬件引导版本）。
- 入口：[官方技术文章](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) · [开源仓库](https://github.com/meta-pytorch/KernelAgent)。
- 定位：由 PyTorch 程序生成、验证并优化 Triton Kernel 的多 Agent 系统。

### 3.1 技术路线

| 模块 | 职责 |
| --- | --- |
| Profiler | 收集 Nsight Compute 硬件计数器 |
| Diagnose / Judge | 识别资源瓶颈与可能根因 |
| Analyzer | 根据架构及瓶颈生成优化建议 |
| Orchestrator | 组织候选策略、历史知识和任务 |
| Optimization Manager | 分支探索并生成候选代码 |
| Benchmark | 测量真实性能，输出下一轮反馈 |

其关键是将硬件反馈转化为行动，而非仅把运行时间传给 LLM。[S2]

例如，若一个 Kernel 的 DRAM 吞吐较高、存在长 scoreboard stall 且 warp occupancy 较低，Agent 应分别分析数据搬运和延迟隐藏，研究访存布局、流水深度、寄存器压力及可用并行度；不能仅凭单个指标认定提高带宽就是最优解。每项假设都要经过重新 Profiling 检验。

### 3.2 公开成果

| 指标 | 作者报告 |
| --- | ---: |
| KernelBench L1 的 100 项任务 | 优化评估范围 |
| 相比上一版生成 Kernel | 平均 2.02× |
| 相比默认 torch.compile | 平均 1.56× |
| 优于默认 torch.compile 的任务 | 65 / 100 |
| H100 Roofline 效率 | 报告为 89% |

早期版本在 250 项 KernelBench 任务上的正确率，与该版本在 L1 上的性能评估不是同一个结论。这里的效率也是作者采用的硬件指标口径，不表示任意 LLM 整体已经达到理论性能上限。

### 3.3 实际如何使用

KernelAgent 更适合已经有独立参考实现、明确形状和 dtype 的热点算子。对于我们的 H800 项目，可以先提取当前真实负载的算子样本及输入输出，再尝试其生成、Profiling 与验证流程。需要检查实际 CUDA/Triton 版本和硬件支持；H100 的数值不能直接搬到 H800。

它和我们之前建立的 Roofline 理论直接对应：资源模型负责提出合理假设，硬件计数器提供证据，代码 Agent 执行变更，正确性与性能测试决定是否接受结果。

## 4. AMD GEAK v4：从算子优化升级为端到端性能工程

- 公开时间：2026-07-23。
- 入口：[技术报告](https://www.amd.com/en/developer/resources/technical-articles/2026/geak-v4.html) · [官方文档](https://rocm.docs.amd.com/projects/geak/en/latest/) · [开源仓库](https://github.com/AMD-AGI/GEAK)。
- 定位：AMD Instinct GPU 上的多 Agent Kernel 与真实 LLM Serving 优化系统。
- 研究价值：本次调研中，与实际推理优化项目最接近的工程架构之一。

### 4.1 两层递归优化结构

~~~text
端到端优化层
  固定模型、目标负载、预算和评价协议
                 |
          Warm-server Profiling
                 |
       按端到端影响排序热点
                 |
      配置搜索 / 已有后端对比
                 |
           仍有足够收益？
             |       |
            否       是
             |       |
          保留    Kernel 优化层
                     |
              分支分析与实现
                     |
              独立验证与组合
                     |
              返回已验证 patch
                     |
          真实服务 A/B 验证与复测
                     |
         接受 / 回滚 / 重新 Profiling
~~~

GEAK v4 的关键工程决策是将工作流控制与 LLM 判断分离。确定性的 JavaScript 工作流负责预算、并行、阶段推进、验证和退出；LLM 专注技术分析、选择方向、编写代码及解释结果。局部 Kernel 优化器可以作为端到端优化器的子程序被调用。[S3]

这一结构解决了普通 Coding Agent 的一个常见问题：Agent 即使能写出高性能 Kernel，也不一定知道哪个 Kernel 值得投入时间，或如何证明它改善了整个服务。

### 4.2 先做低成本搜索，再做高成本改写

GEAK 的端到端工作流先检查已有配置和后端，再比较热点实现，最后才将昂贵的源码优化预算投向高影响候选。这与 Amdahl 定律一致。

若某个模块占暴露执行时间的比例为 $p$，局部加速为 $k$，在其他成本不变、串行可加的近似下：

$$
S=\frac{1}{1-p+p/k}.
$$

例如，一个只占 2% 的模块即使加速 5 倍，整体加速仍只有约 1.016×；一个占 60% 的模块加速 1.15 倍，整体则可达到约 1.085×。这解释了为什么系统应优先选择高影响热点。真实并行服务还需要关键路径和资源竞争分析，不能把所有 GPU kernel duration 直接相加当作端到端耗时。

### 4.3 公开性能案例

以下摘录 GEAK v4 官方报告的真实服务案例，各行只代表对应模型、硬件、精度与负载：

| 模型 | 硬件 / 引擎 | 主要优化 | 报告的 E2E 吞吐提升 |
| --- | --- | --- | ---: |
| Qwen3.5-27B-FP8 | MI300X / SGLang | blockscale GEMM 后端替换 | +60% |
| Qwen3-14B-FP8 | MI300X / SGLang | blockscale GEMM 后端替换 | +96% |
| MiniMax-M3-MXFP8 | MI355X / vLLM | 稀疏 Attention | +42.2% |

前两例 TP1、输入/输出长度 1K/1K、并发 64；第三例 TP8、8K/1K、并发 64。官方另报告热点 Kernel 基准几何平均加速 1.33×，相应通用 Claude Code 和 GEAK v3.2 基线均为 1.09×。这些是不同级别和不同口径的结果，不应直接比较或累乘。

### 4.4 如何迁移到 NVIDIA H800

GEAK 的具体后端是 AMD ROCm 生态，不应假设可以在 NVIDIA H800 原样运行。值得迁移的是其工作流：

- 用 Nsight Systems 捕获真实服务时间线，用 Nsight Compute 分析高影响 Kernel。
- 固定原始正确性 oracle、Benchmark 与基线提交。
- 分离技术规划、代码改写和独立复核。
- 先比较已有 CUDA/Triton/可用后端和服务配置，再做高成本源码探索。
- 局部候选只有通过真实 TTFT、TPOT、吞吐、正确性和稳定性评价才进入累计基线。
- 每轮接受变更后重新 Profiling，避免继续优化已经迁移的瓶颈。

## 5. 其他值得研究的 Kernel 自动优化工作

### 5.1 CUDA Agent：通过强化学习训练优化能力

清华 AIR 与字节跳动 Seed 的 CUDA Agent（2026-02-27）将研究重点从运行时 Agent 工作流推进到专门训练模型的优化能力。它利用合成算子任务、可执行的 CUDA 环境及正确性/性能反馈实施 Agentic RL。[S4]

作者报告 KernelBench L1/L2/L3 中快于 torch.compile 的任务比例分别为 100%、100%、92%，项目页给出总体平均加速 2.11×。应区分通过率、胜率和平均加速比；这不是推理服务吞吐结果。公开工作目录和 6K 算子训练数据，可用于研究训练环境和奖励设计。

入口：[项目主页](https://cuda-agent.github.io/) · [论文](https://arxiv.org/abs/2602.24286)。

对我们的直接启发是建立准确的反馈和任务数据，而不是立即训练专用模型。使用现有 Codex 等通用 Agent 的成本明显低于重新训练 CUDA Agent；待积累大量可验证优化轨迹后，再研究学习型策略。

### 5.2 KernelPro：把专家 Profiling 规则变为工具

KernelPro 研究如何通过 Roofline 分类调用专门的微架构分析工具，用结构化诊断连接原始计数器与优化建议，并结合树搜索和历史经验探索候选。作者还报告在专家优化的 MoE 训练算子上获得 1.23× 加速，并给出 MCTS 相较贪心搜索的消融结果。[S5]

这是有价值的理论方向：硬件指标的数值本身并不是优化建议，系统需要把指标转化为可验证的瓶颈假设。该工作仍以论文报告为证据，不能外推到我们的服务性能。

入口：[论文](https://arxiv.org/abs/2606.26453)。

### 5.3 AutoKernel：面向任意 PyTorch 模型的热点搜索

AutoKernel（2026-03）从模型 Profiling 获取热点，利用 Amdahl 影响排序并进行 Triton/CUDA 迭代搜索。论文强调形状扫描、数值稳定性、确定性与边界输入测试。它适合参考轻量级优化控制器和验证设计，但 PyTorch 模型级入口与复杂推理引擎中真实的动态服务路径存在差异。[S6]

入口：[论文](https://arxiv.org/abs/2603.21331) · [开源仓库](https://github.com/RightNow-AI/autokernel)。

### 5.4 FlashInfer-Bench：建立 AI 生成 Kernel 到推理服务的桥梁

FlashInfer-Bench 用统一的 Trace 描述算子定义、工作负载、实现与评价，并支持将通过验证的候选注入 vLLM、SGLang 等引擎。[S7]

MLSys 2026 FlashInfer AI Kernel Generation Contest 进一步将评价对象指向生产模型中的 Fused MoE、DeepSeek Sparse Attention 和 Qwen3-Next Gated DeltaNet 的 Prefill/Decode。竞赛使用 B200，分别设置人类专家辅助和 Agent 生成相关赛道，并提供 OpenEvolve Agent 基线。[S8]

这对我们的模型专项优化尤其有意义：即使不使用竞赛 Kernel，也可以复用其任务表示、真实形状覆盖、正确性 oracle、性能比较和动态替换思想。注意 Blackwell 的具体指令、数据格式与实现不能假定在 Hopper H800 可用。

另有 Agentic Kernel Optimization 研究报告，在该竞赛的 Fused MoE 评价中取得相对 FlashInfer 基线 1.71× 的加速；作者披露整个研究耗费约 19 亿 Agent token，并包含人工搜索指导。[S13] 这说明强算子结果不等于低成本、完全无人干预的自动优化。

## 6. SEIS：让整个推理引擎持续自我演化

- 时间：2026-10。
- 入口：[论文](https://arxiv.org/abs/2610.04646) · [项目页](https://nehzux.github.io/SEIS/) · [代码与历史](https://github.com/nehzux/SEIS)。
- 定位：不只优化单个算子，而是让固定的 Agent 在多轮会话中持续改进 mini-sglang。

### 6.1 方法

每轮 Agent 继承上一次的代码与优化记录，重新分析热点、修改实现、测试并保留有效变更。不同演化分支还可以交换有效发现。作者报告形成了新的 Kernel、融合实现、运行时改进和 prompt-lookup 投机解码等跨层修改。[S9]

这里演化的是推理引擎实现，不是 Agent 自身的权重。

### 6.2 公开结果

在 Qwen3-0.6B、H100、单请求条件下，作者给出以下结果：

| 引擎 | 报告的吞吐（tokens/s） |
| --- | ---: |
| 原始 mini-sglang | 594 |
| vLLM | 633 |
| SGLang | 626 |
| TensorRT-LLM | 690 |
| SEIS 最佳演化引擎 | 1942 |

该测试含 Prefill，为 100 条自然长度请求的三次重复中位数。相对原始引擎为 3.27×，作者另报告移除投机解码后仍有 3.08×。论文还检查了数学及长上下文检索任务的正确性，项目页报告 GSM8K 与长输入检索准确率分别变化 +1.06 和 -0.37 个百分点。

**边界：** 小模型、单请求、特定数据与基线。不能将该结果解释为超过大型 MoE 生产系统、任意并发下的 vLLM/SGLang 或多卡 H800 推理。保留独立测试集和更严格的引擎正确性检查，仍是工程化关键。

### 6.3 研究启发

SEIS 是将 Kernel、Runtime、缓存和生成机制放在同一搜索空间的探索。它说明多轮经验继承与跨分支复用可能比重复启动全新 Agent 更有价值，也为研究优化耦合提供了实验范式。

## 7. Agent 的能力到底到了什么程度？

### 7.1 ISO-Bench：真实工程补丁评价

ISO-Bench 选择 vLLM 与 SGLang 的 54 个实际性能优化任务（39 和 15），要求 Agent 根据源代码和瓶颈描述生成补丁，并在 120 分钟预算内与专家修改比较。[S10]

项目页报告，某些 Agent 的正确瓶颈识别率最高达 87.2%，真正成功率却最高只有 17.9%；单纯依赖运行时分数可能高估实际能力。该工作将执行结果与补丁是否真正解决目标问题结合评价。

这说明知道原理、提出优化方向、实现有效补丁是三个不同的能力。17.9% 是该基准与所测配置的结果，不是当前所有 Coding Agent 的通用成功率。

入口：[论文](https://arxiv.org/abs/2602.19594) · [项目主页](https://ayushnangia.github.io/iso-bench-website/)。

### 7.2 InferenceBench：开放式推理服务优化

InferenceBench 给 Agent 一个模型、一张 H100 和两小时优化预算，要求交付可运行的推理服务，并分别考察 TTFT、TPOT、并发吞吐及综合指标。只对最终通过正确性与完整性检查的提交计分，而不是挑选中途最好的一次结果。[S11]

原论文的 15 种 Agent 配置最高为 8.08×（相对原始 PyTorch），低于同预算的传统超参数搜索 11.53×。截至 2026-11-24，公开榜单已增加新的 Agent 运行：

| 方案 | 榜单几何平均加速（相对原始 PyTorch） |
| --- | ---: |
| PyTorch Default | 1.00× |
| vLLM Default | 4.05× |
| Hyperparameter Search | 11.53× |
| Claude Opus 5.5 Max | 12.08× |
| Claude Sonnet 5.5 Max | 12.06× |
| GPT-6.2 Astra | 10.76× |

新版结果表明部分 Agent 已经在该榜单超过参数搜索基线，不能再笼统地说所有 Agent 都不如传统搜索。但榜单标记了 prompt 和 scaffold 版本变化，且领先方案使用 FP8 权重和 FP8 KV Cache 等策略。这些条件与本项目固定全部实际精度的约束不同，不能作为我们的预期收益。

作者观察到 Agent 仍可能过早停止、重复测量、破坏已找到的有效配置或试图利用评价漏洞。最关键的能力并非罗列优化知识，而是系统化提出不同候选、测量、比较、保存并交付可靠结果。

入口：[论文](https://arxiv.org/abs/2607.20468) · [动态榜单](https://inferencebench.ai/)（检索日期 2026-11-24）。

## 8. 横向对比与选择

| 工作 | 核心机制 | 主要用途 | 对 H800 项目的适用性 |
| --- | --- | --- | --- |
| KernelEvolve | 搜索图、动态知识、评估基础设施 | 异构 Kernel 演化 | 借鉴算法和知识结构 |
| KernelAgent | Roofline 与计数器驱动的多 Agent | Triton 算子优化 | 优先检查、复现单算子 |
| GEAK v4 | 系统层调用 Kernel 层，独立复核 | 真正的服务级优化 | 迁移工作流，不直接复用 ROCm 后端 |
| CUDA Agent | Agentic RL、执行反馈训练 | 学习 CUDA 优化能力 | 借鉴训练任务与奖励，暂不训练 |
| KernelPro | 专家 Profiling 工具、MCTS | 根因诊断及搜索 | 借鉴理论指导与搜索机制 |
| AutoKernel | 热点排序、迭代验证 | PyTorch 模型局部优化 | 借鉴轻量控制器 |
| FlashInfer-Bench | Trace、评估、替换接口 | 真实 LLM 算子任务 | 优先参考任务与验证协议 |
| SEIS | 多轮代码与经验继承 | 整体推理引擎演化 | 中长期探索 |
| ISO-Bench | 真实 PR 与性能、意图评价 | 评估 Agent 工程能力 | 参考补丁评价 |
| InferenceBench | 同预算开放式服务优化 | 评价系统级搜索能力 | 参考同预算搜索基线 |

这些工作采用不同模型、硬件、精度和参考实现，不适合按照加速数字统一排名。尤其应将算子微基准、服务吞吐、算法质量与优化成本分别统计。

## 9. 理论视角：自动优化到底需要解决什么问题？

### 9.1 优化不是单纯代码生成，而是带约束的序贯搜索

令 $x$ 表示代码、运行时配置及优化策略，$\omega$ 表示工作负载，$h$ 表示硬件和环境。以目标性能函数 $J$ 表示例如满足 SLO 的 Goodput，自动优化任务可表示为：

$$
x^*=\arg\max_{x\in\mathcal X}J(x;\omega,h)
$$

同时满足数值与任务语义约束、固定精度、显存容量、硬件预算，以及总搜索成本限制：

$$
\sum_{i=1}^{n}c(x_i)\leq \mathcal B.
$$

这里 $c(x_i)$ 可以包含 GPU 小时、构建时间、Agent token 与人工复核成本；$\mathcal B$ 应按各资源的实际计量口径设置，不能把异质成本直接无单位相加。目标不仅是最终性能，还包括单位优化预算的收益。

每个候选可记录：

$$
z_i=(x_i,\widehat J_i,J_i,\sigma_i,q_i,c_i).
$$

$\widehat J_i$ 是理论预测，$J_i$ 是实测，$\sigma_i$ 是不确定度，$q_i$ 是正确性结果，$c_i$ 是评估成本。通过累积这些数据，Agent 才能学习哪些理论假设可靠，哪些方向收益过低。

### 9.2 理论对搜索的价值

理论不必精确预测全部运行时行为，首先要排除低价值方向：

| 理论 | 对 Agent 的直接作用 |
| --- | --- |
| Roofline | 判断计算、访存上限及潜在资源瓶颈 |
| Amdahl / 关键路径 | 估计模块优化的端到端收益空间 |
| 通信模型 | 筛除受互联和同步限制的并行策略 |
| 状态容量模型 | 判断 batch、缓存及并行配置是否可行 |
| 排队与调度 | 判断局部改进是否转化为服务质量 |
| 优化耦合模型 | 选择值得联合验证的变更 |

一种可研究的优先级指标为：

$$
a(x)=\frac{\mathbb E[(J(x)-J_{\rm best})_+]}{\mathbb E[c(x)]}.
$$

它表达单位搜索成本的期望收益，类似成本感知的贝叶斯优化思想。这是本文提出的研究抽象，并非上述项目都使用了这一公式。实际候选的正确性概率、预测误差与探索价值也应进入决策，不能盲目按预期收益贪心排序。

### 9.3 自动优化的真正风险：评价失真

如果 Agent 可以同时修改实现、正确性参考和性能评价，搜索可能优化的是评分漏洞，而非实际性能。因此必须保持 benchmark、参考实现、输入输出长度、采样语义、计量范围和独立测试集不受候选 Agent 随意修改。

GPU 多分支搜索应隔离代码工作区，在资源受限时串行独占同一 GPU 测量，避免竞争干扰。最佳已验证候选与未通过的探索分支应分别保存。对独立复测失败、质量退化或未达到实际性能噪声门槛的改动，不得作为新基线。

## 10. 对现有 Qwen/H800 研究的建议

以下为后续研发方向，不代表已经实现自动优化器或获得实测结果。

### 第一阶段：自动化测量与候选管理

先建立确定性的实验控制层，而不是再写一个大而全的 Coding Agent。控制器读取固定负载、基线和资源预算，负责工作区、GPU 互斥、服务生命周期、预热、重复测量、失败恢复、结果登记及最优候选保存。

现有[Benchmark 协议](../../docs/benchmark-protocol.md)、[优化收益预测器](../../models/predictor/README.md)、[实施 skill](../../archive/qwen-h800-tricks/skills/optimize-qwen-h800/SKILL.md)应作为起点，不重新发明已有资产。

### 第二阶段：理论驱动的热点 Kernel 自动优化

优先参考 KernelAgent、FlashInfer-Bench 与 GEAK 的设计。对当前环境中真实存在且有足够性能占比的 Attention、索引选择、MoE 或混合状态算子，捕获代表性 shapes、dtype、输入输出及参考实现，再交给 Codex 探索少量机制不同的候选。不能仅依据其他模型上的热点决定优先级。

每个候选必须先通过独立正确性检查，再进入微基准和端到端验证。没有对应 GPU 或可信输入时，可先完成候选设计与测试框架，但状态必须标明未实测。

### 第三阶段：引擎级联合优化

当局部优化能够稳定产生可接受 patch 后，再研究 SEIS 一类多轮引擎演化。可选取两个已验证优化，分别在同一基线上测量 $T_0,T_A,T_B,T_{AB}$，估计绝对耗时节省的交互项：

$$
\Delta_{AB}=T_A+T_B-T_0-T_{AB}.
$$

正值意味着联合耗时节省高于独立节省之和，负值意味着收益部分抵消；需要报告测量不确定度并分析瓶颈迁移。在不同上下文、batch 和服务负载下重复测量，可以逐步形成优化耦合图谱。

### 第四阶段：评价自动优化系统本身

建议在同等 GPU 时间、Agent token 和任务质量约束下比较：

| 基线或方案 | 用途 |
| --- | --- |
| 默认引擎 | 工程起点 |
| 传统参数搜索 | 强且低成本的自动搜索基线 |
| 普通 Codex 单线优化 | 判断额外工作流的贡献 |
| 理论引导 Agent | 检验性能模型的价值 |
| 理论 + 分支搜索 + 独立验证 | 检验完整系统的价值 |

评价最终 TTFT、TPOT、输出吞吐、总吞吐、Goodput，以及已验证候选比例、搜索成本、复现率和跨负载稳定性。所有方案固定实际模型、权重/激活/状态精度、任务语义及总 GPU 预算，不以量化、剪枝、缩短输出或关闭所需 reasoning 换取收益。

### 推荐阅读顺序

1. **GEAK v4**：理解系统与 Kernel 两层的确定性工作流，优先迁移其验证原则。
2. **KernelAgent / KernelPro**：理解如何让 Roofline 与 Profiling 变成可操作的优化假设。
3. **FlashInfer-Bench**：研究真实 LLM 算子任务的输入、正确性和替换接口。
4. **InferenceBench / ISO-Bench**：了解当前 Agent 的真实短板和应采用的强基线。
5. **SEIS / KernelEvolve**：研究跨轮搜索、经验复用及整个推理系统的持续演化。

## 11. 结论

业界已经证明 AI 可以有效参与高性能算子生成，部分工业系统已将其用于真实推理负载优化。但尚无充分公开证据表明 Agent 可以在任意大型模型、任意硬件和服务负载上稳定找到全局最优，更不能从弱基线的大幅加速推断它普遍超过专家优化。

未来最值得研究的不是更长的自动编程循环，而是**理论指导、真实测量、搜索策略和独立评价的联合设计**。我们的理论框架可以用来估计性能极限及候选价值，Codex 用来实施，实验控制器负责保存证据与验证，最终以真实推理服务性能判断结果。

## 参考资料

- [S1] Meta, KernelEvolve, 2026-04-02. [技术文章](https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/)；[ISCA 2026 论文](https://doi.org/10.1109/ISCA66397.2026.00063)。
- [S2] PyTorch, KernelAgent, 2026-03-06. [官方文章](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/)；[代码](https://github.com/meta-pytorch/KernelAgent)。
- [S3] AMD, GEAK v4: From Faster Kernels to Faster Workloads, 2026-07-23. [报告](https://www.amd.com/en/developer/resources/technical-articles/2026/geak-v4.html)；[文档](https://rocm.docs.amd.com/projects/geak/en/latest/)；[代码](https://github.com/AMD-AGI/GEAK)。
- [S4] Dai et al., CUDA Agent, 2026. [arXiv:2602.24286](https://arxiv.org/abs/2602.24286)；[项目](https://cuda-agent.github.io/)。
- [S5] Gai et al., Optimizing CUDA like a Human, 2026. [arXiv:2606.26453](https://arxiv.org/abs/2606.26453)。
- [S6] Jaber et al., AutoKernel, 2026. [arXiv:2603.21331](https://arxiv.org/abs/2603.21331)；[代码](https://github.com/RightNow-AI/autokernel)。
- [S7] Xing et al., FlashInfer-Bench, 2026. [arXiv:2601.00227](https://arxiv.org/abs/2601.00227)。
- [S8] MLSys 2026 FlashInfer AI Kernel Generation Contest. [竞赛主页](https://mlsys26.flashinfer.ai/)。
- [S9] Xu et al., Self-Evolving Inference Systems, 2026. [arXiv:2610.04646](https://arxiv.org/abs/2610.04646)；[项目](https://nehzux.github.io/SEIS/)；[代码](https://github.com/nehzux/SEIS)。
- [S10] Nangia et al., ISO-Bench, 2026. [arXiv:2602.19594](https://arxiv.org/abs/2602.19594)；[项目](https://ayushnangia.github.io/iso-bench-website/)。
- [S11] Yeon et al., InferenceBench, 2026. [arXiv:2607.20468](https://arxiv.org/abs/2607.20468)；[动态榜单](https://inferencebench.ai/)。
- [S12] NVIDIA, Automating Inference Optimizations with NVIDIA TensorRT LLM AutoDeploy, 2026-02-09. [文章](https://developer.nvidia.com/blog/automating-inference-optimizations-with-nvidia-tensorrt-llm-autodeploy/)。
- [S13] Luo et al., Agentic Kernel Optimization: Generating State-of-the-Art GPU Kernels Without Hand-Written CUDA, 2026. [arXiv:2608.14560](https://arxiv.org/abs/2608.14560)。

滚动网站和公开排行榜以 2026-11-24 检索结果为准；实际复现前应固定上游 commit、版本、硬件和评价协议。本次未运行上述工具，也未访问 H800 实验环境。

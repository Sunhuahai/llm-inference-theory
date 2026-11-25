# 一手阅读地图：由问题找到方法

> 原始资料核对日期：2026-11-25。此处提供理论与评价基础，不是最新工作排行榜；外部结果不转换为本仓库性能承诺。

| 研究问题 | 一手资料 | 对应仓库内容 | 借鉴的范围 |
| --- | --- | --- | --- |
| 工作量如何约束性能？ | [Roofline 原始技术报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) | [性能边界](../../theory/performance/README.md) | 算术强度与资源上界；不把峰值下界当可达时间 |
| 怎样解释真实硬件观测？ | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | [诊断](../../theory/diagnosis/README.md) · [微基准](../../experiments/microbenchmarks/README.md) | 指标语义、replay 与采集扰动；版本需随环境固定 |
| 如何增加每轮有效生成？ | [Speculative Decoding, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | [生成效率](../../theory/generation/README.md) | 分布保持的接受/拒绝与产出成本；具体实现另行检查 |
| 如何从局部模型推到服务？ | [Vidur v2, 2024](https://arxiv.org/abs/2405.05465v2) | [仿真](../../models/simulation/README.md) · [模型验证](../../experiments/model-validation/README.md) | profiling、算子预测和离散事件模拟的组合 |
| 分离阶段何时有价值？ | [DistServe v3, OSDI 2024](https://arxiv.org/abs/2401.09670v3) | [服务理论](../../theory/serving/README.md) | P/D 干扰、阶段分配与服务约束 |
| 分块如何改变吞吐/延迟？ | [Sarathi-Serve v3, OSDI 2024](https://arxiv.org/abs/2403.02310v3) | [服务模型](../../models/serving/README.md) | 调度粒度与混合 batch，不能照搬单点收益 |
| 如何用有限实验寻找更好配置？ | [Practical Bayesian Optimization v2, 2012](https://arxiv.org/abs/1206.2944v2) | [搜索与规划](../../optimizer/planning/README.md) | 预测驱动选择及实验成本建模；本仓库启发式不等于完整 BO |
| 如何表示实现搜索空间？ | [Ansor, OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/zheng) | [候选空间](../../optimizer/search-space/README.md) | 分层张量程序表示、代价模型和搜索 |
| 如何同时评价正确与快速？ | [KernelBench v1, 2025](https://arxiv.org/abs/2502.10517v1) | [Agent benchmark](../../experiments/agent-benchmark/README.md) | 正确性联合性能阈值；kernel 任务不等同于完整服务 |

## 阅读时提取什么

每项资料记录五件事：它解决的对象与动作空间、观测和代价模型、正确性及评价基线、支持结论的设置、哪些机制能够迁移。只摘录加速数字会丢失适用条件。

公开结果若需要复现，应另登记论文版本、源码 SHA、硬件、模型、负载和测量方法。论文报告与自己复现放在不同证据条目里。新的接口、成本分摊、预算曲线和条件推导属于本仓库设计时，明确标注，不能借引用暗示原作者证明了这些具体结论。

## 进阶入口

[业界自主优化图解](../industry/agentic-inference-optimization-guide.md)与[深度调研](../industry/ai-driven-inference-optimization.md)保留 KernelEvolve、KernelAgent、GEAK、SEIS 等已有项目的机制和证据范围；[InferSim 与 AIConfigurator](../industry/infersim-aiconfigurator.md)侧重工具使用。自主优化的本仓库框架统一放在 [optimizer/](../../optimizer/README.md)，避免把外部综述和自己的算法设计混为同一层。

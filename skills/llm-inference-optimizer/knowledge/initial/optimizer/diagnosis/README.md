<!-- Generated seed snapshot; source: optimizer/diagnosis/README.md @ 1b4c5a37763475eb39b31dd083fcad256fc2578e. Do not hand-edit. -->
# 从性能观测到优化候选

> 状态：接口设计与研究假设。理论依据见[瓶颈诊断](../../theory/diagnosis/README.md)。

## 1. 诊断的输出是一组可检验解释

“TPOT 高”是一种现象。HBM 访问、低并行度、通信等待、host 供给不足等都可能产生相似结果。诊断模块应返回解释及区分它们所需的证据，而非仅输出一个 `memory-bound` 标签。

例如“增加融合可能减少中间流量”还不够完整。完整判断应包含：哪些中间张量穿过哪一存储边界、减少多少字节、是否位于当前关键路径、增加多少寄存器/工作区，以及哪个小实验能检验此解释。

## 2. 输入契约

已有项目通常先有零散日志与历史优化结果。按[项目指导](../../docs/project-optimization-guide.md)恢复可比性、候选父版本和机制台账，再准备下列输入；缺失项只在会改变当前判断时阻塞结论。

| 输入 | 必需的上下文 | 用途 |
| --- | --- | --- |
| 服务指标 | 时间窗、负载、错误与输出长度 | 确定真正优化目标 |
| 执行时间线 | stream/rank、依赖、重叠、host gap | 识别暴露开销 |
| 硬件计数器 | 工具版本、单位、采集范围 | 支持资源瓶颈假设 |
| 资源账本 | shape、dtype、FLOPs、分层流量、状态容量 | 计算边界与成本变化 |
| 基线标识 | checkpoint、代码 SHA、后端、硬件、负载 hash | 判断证据可否复用 |

计数器缺失应保留 `unknown`。GPU 显存容量利用率不能替代 HBM 带宽利用率，累计 Kernel duration 不能直接替代服务关键路径时间。Nsight Compute 的 replay 与计数器采集可能改变执行行为，因此诊断 profile 与正式无重采集压测应独立记录。[1]

## 3. 假设接口

```yaml
diagnosis_id: diag-example
status: hypothesis
baseline_id: synthetic-baseline
scope: decode / fixed-shape / one-rank
observation_refs: []
hypothesis: 重复中间张量访存限制当前执行路径
alternatives:
  - 低并行度使延迟隐藏不足
  - host 供给间隙占主导
resource_boundary: HBM
supporting_evidence: unknown
contradicting_evidence: unknown
discriminating_probe: 同 shape 下比较融合前后的流量及关键路径
probe_cost_estimate: unknown
invalidation: 流量下降但无 profiler 外的延迟变化，且无新瓶颈
candidate_mechanisms:
  - remove_intermediate_memory_traffic
confidence_kind: uncalibrated_assessment
```

这里的文字字段是待补证据的示例，不是已完成诊断。无校准的判断不能写成“90% 置信度”。

## 4. 选择区分性实验

若假设 $H_1,H_2$ 对某次干预都预测相同结果，这个干预很难区分二者。应优先寻找能使两种解释产生不同预测的实验。例如固定执行形状，对内存流量和 host 提交分别做小范围干预；保持其他条件一致，再看哪个改变了正式指标。

可使用假设分歧、预期信息增益或简单的诊断决策表选择实验；它们属于研究方法，未经校准不应当作可靠概率。诊断成本与优化成本共享预算，因此有时直接评估一个低成本候选更划算。

## 5. 从诊断进入搜索

每个候选引用至少一个 `diagnosis_id`，说明改变的资源、影响目标及失效条件。搜索后返回三类偏差：

- **机制不成立：** 实现未减少预期成本，或原实现已覆盖该优化。
- **机制成立但净收益消失：** 新增成本、容量下降或瓶颈迁移抵消收益。
- **证据不足：** 变化落在测量波动内，或缺少区分实验。

这三种结果应分别进入[记忆](../memory/README.md)，不能统一写成“该技术无效”。

## 参考资料

1. NVIDIA. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)，核验 2026-11-25。用于计数器、replay 与采集扰动边界；本文的诊断契约为本仓库设计。

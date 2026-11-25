# 理论驱动的瓶颈诊断

从[诊断理论](../knowledge/initial/theory/diagnosis/README.md)与[执行理论](../knowledge/initial/theory/execution/README.md)开始，结合[资源模型](../knowledge/initial/models/resource/README.md)。

区分 forward（工作量与执行系统推导性能）和 inverse（根据观测反推约束）。仅凭 GPU 利用率、总 TPOT 或 Roofline 区域不能唯一识别慢因。

- 计算：检查有效 FLOPs、形状、并行度、资源占用。
- 访存：区分权重、激活、状态和中间流量及各层带宽，确认减少的是实际流量而非理论字节。
- 通信与执行依赖：检查关键路径、rank 等待、重叠和争用。
- 资源容量：检查状态生命周期、碎片、并发上限，容量变化可能间接改变吞吐。
- 生成与服务：检查有效提交 token、排队、阶段干扰、SLO。

对每个关键现象维护竞争假设：证据支持/反对、预测分歧、能推翻的最小观测。优先已有 trace，若不足则提出受控小实验；不自动启动 GPU。若不同假设同预测一个总延迟，重复该延迟不解决可辨识性。新候选必须注明预计消除的暴露成本与引入的代价，不能将全部 Kernel 时间当成可消除时间。结合[耦合理论](../knowledge/initial/theory/optimization/coupling.md)重新分析当前 parent，不机械复制旧排名。

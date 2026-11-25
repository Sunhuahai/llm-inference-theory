# 独立评价和实验权限

优先依据[测量协议](../knowledge/initial/docs/benchmark-protocol.md)、[独立评价](../knowledge/initial/optimizer/evaluation/README.md)、[模型验证](../knowledge/initial/experiments/model-validation/README.md)。

冻结 oracle、输入输出语义、精度、硬件预算、负载/时间窗口、统计定义和接受条件。量化、降低状态精度、改变模型、提前终止输出等如果未被明确允许，不得冒充同条件加速。候选 Agent 不得修改独立评价脚本来获得更好结果；独立审查真实后端是否生效、reference 实现、形状覆盖、重复与原始日志。性能测试分 warmup 与计时，必要时交错 A/B，保留异常和回归。

实现前记录理论预测与预测的资源、执行、服务层作用；实现后先正确性，再微基准，再端到端。只在授权预算内行动，失败同样记录实际消耗。硬约束没有通过就不能接受；统计不足则未决。Profiler 的计时和服务性能不等价。

理论模型的 promotion 检查的是预先登记的同工作点 paired predictions 对同一实际值的误差，不等于系统优化接受。至少两条留出记录只是程序最低门槛；独立评估者检查真正未见、重复是否独立、是否有选择泄漏、域和样本覆盖、模型物理约束、实际误差差异是否有意义。若关键证据不充分，不晋升；保存失败与修订候选。不要由 Agent 自行编造评审引用。

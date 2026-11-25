# 内存与状态：容量、流量和驻留时间

> 日期：2026-11-25。性质：`derived` 条件性成本分析与 `hypothesis` 验证设计。数值例为合成数据。

状态优化有三种不同的收益：少占空间、少搬数据、少做重算。它们可以同时出现，也可以只出现一种。把“节省了多少显存”直接换算为“加速了多少”会漏掉最关键的容量与执行耦合。

## 1. 先分别记容量与流量

一次部署的瞬时容量约束为：

$$
M_{\rm weights}+M_{\rm state}(t)+M_{\rm workspace}(t)
+M_{\rm graph}(t)+M_{\rm other}(t)+M_{\rm reserve}
\le M_{\rm available}.
$$

各项必须按实际物理分配定义，避免 allocator reserved memory 与其中已分配对象重复相加。容量是某个时刻占据的 Byte；流量是在某段执行中跨某个存储边界搬运的 Byte。相同容量可以被读取一次或多次，跨请求共享也不保证 HBM 只读取一次。

对各层结构相同的普通全注意力，逻辑 KV 每 token 字节数为 $\kappa=2Lh_{\rm kv}ds$。独立请求的逻辑容量是 $\kappa\sum_i\ell_i$。若每页容纳 $p$ 个 token、无跨请求共享，则一种简单物理模型为：

$$
M_{\rm pages}=\kappa p\sum_i\left\lceil\frac{\ell_i}{p}\right\rceil
+M_{\rm metadata}.
$$

它包含页内空余；共享页必须只计一次，copy-on-write 后的新页另计，实际预留及外部碎片也需单独处理。PagedAttention 研究的核心之一就是状态分配与共享效率；分页本身不改变每个 KV 元素的精度。[1]

## 2. 不同架构的状态不能套一个公式

| 结构 | 典型逻辑状态增长 | 建模时需要核对 |
| --- | --- | --- |
| 全注意力 | 每请求随上下文增长 | KV heads、每层布局、实际读取量 |
| 滑动窗口注意力 | 单层可随有效窗口受限 | 是否仍保留全局层/额外历史状态 |
| recurrent / 状态空间层 | 递归状态可不随长度增长 | 中间 checkpoint、批量验证及回滚缓冲 |
| 混合结构 | 各类状态相加 | 恢复边界必须使所有状态一致 |
| MoE | 权重驻留和状态分别核算 | 专家放置、触达集合、临时 dispatch 缓冲 |

这些是建模分类，真实结构以 checkpoint 与引擎实现为准。固定精度实验不通过更换模型架构或压低状态精度获得所谓同条件收益。

## 3. 容量变化如何影响吞吐

只有状态大小与其他占用近似固定时，才可使用 $b_{\max}=\lfloor M_0/(\kappa\ell)\rfloor$。若工作区、Graph 或缓存预算随 batch 变化，应直接解完整容量约束。

**合成算例：** 扣除固定占用后，状态可用空间为 24 GiB，每个活跃请求占 1 GiB。基线能容纳 24 个请求，每步 10 ms，每请求每步提交一个 token；候选使步骤降为 8 ms，但需要额外 8 GiB 工作区，因而只能容纳 16 个请求。

$$
X_0=24/0.010=2400\ {\rm tokens/s},\qquad
X_1=16/0.008=2000\ {\rm tokens/s}.
$$

这个固定形状稳态例中，单步延迟降低 20%，可容纳 batch 下的吞吐却下降约 16.7%。实际还应重测两个 batch 下的耗时，不能假定原微基准的 8 ms 会自动保持。低负载时，额外空间也可能不约束 batch，因此结论随负载改变。

## 4. 缓存的价值取决于复用与机会成本

缓存对象 $i$ 的时间价值可在简化可加模型下写成：

$$
V_i=\mathbb E\left[\sum_{j=1}^{N_i}
(T_{{\rm recompute},ij}-T_{{\rm restore},ij})\right]
-T_{{\rm store},i}-C_{{\rm capacity},i}.
$$

$C_{{\rm capacity},i}$ 是按同一目标单位折算的容量机会成本，例如缓存挤占活跃请求空间后增加的等待。它不能直接填成字节数。共享依赖、未知复用距离与负载变化使得按价值/字节排序只是一种启发式，不能保证全局最优。

前缀 token 命中率也不是计算节省率。标准因果注意力中，缓存前缀 $m$、新后缀 $q$ 时，后缀仍要与前缀状态交互；细节见[统一理论总纲](../foundations/unified-theory.md)第 3 节。混合状态的键、位置、模型 revision、采样相关状态和恢复边界都要核对，专项背景见[前缀与上下文推导](../foundations/cache-context-ceilings.md)。后者包含历史 Qwen 案例，不能自动代表所有模型。

## 5. 驻留时间连接容量与服务

在稳定系统、每请求状态独立且计量边界一致的条件下：

$$
\overline M_{\rm state}=\lambda\,
\mathbb E\left[\int_0^{T_{\rm resident}}m_i(u)\,du\right].
$$

这是平均占用关系，不能替代峰值可行性检查。输出更长、请求暂停更久或缓存保留更久，都会改变状态驻留积分。共享对象要按物理对象计量，不能把同一页算入多个请求后再次求和。服务层影响见[推理服务](../serving/README.md)。

**可证伪预测：** 一个只降低页内浪费、保持固定 batch 与执行路径的分配策略，应首先表现为容量改善。如果同时观察到显著时间变化，需测定缓存局部性、地址计算或调度变化来解释，不能直接归因为“容量少了，所以带宽更快”。

Agent 应分别预测驻留容量、访问流量、恢复成本和可行 batch；独立记录 OOM、驱逐及命中长度分布。模型入口见[性能模型](../../models/README.md)，候选表达见[自主优化](../../optimizer/README.md)，证据记录见[实验平台](../../experiments/README.md)。

## 参考资料

1. Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，2023；核对日期 2026-11-25。
2. Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，2022；核对日期 2026-11-25。用于理解减少中间流量与改变状态容量的区别。
3. 本文容量账本、驻留积分与教学算例是明确假设下的建模；已有更细的假设及推导见[统一理论总纲](../foundations/unified-theory.md)。

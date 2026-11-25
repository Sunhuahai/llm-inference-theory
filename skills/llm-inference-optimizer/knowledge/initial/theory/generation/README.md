<!-- Generated seed snapshot; source: theory/generation/README.md @ 1b4c5a37763475eb39b31dd083fcad256fc2578e. Do not hand-edit. -->
# 生成效率：每个有效 token 需要多少执行成本

> 日期：2026-11-25。性质：`derived` 条件推导；数值全部为合成算例。分布保持条件以具体算法与实现为准。

推理优化既可以减少一次执行的时间，也可以增加一次执行最终提交的有效 token 数。投机解码的关键就在后一项：草稿及验证增加了工作，但如果提交数量增加得更快，单位输出成本仍可能下降。

## 1. 使用总产出除以总时间

对一个无额外空闲、串行执行的生成组，周期 $j$ 的耗时为 $T_j$，最终提交数为 $K_j$：

$$
X=\lim_{n\to\infty}\frac{\sum_{j=1}^n K_j}{\sum_{j=1}^n T_j}
=\frac{\mathbb E[K]}{\mathbb E[T]}
$$

最后一步需要相应稳态/遍历条件和有限期望。不要使用 $\mathbb E[K/T]$ 代替。**合成例：** 两个周期分别为 $(K,T)=(1,1\,\mathrm{ms})$、$(9,3\,\mathrm{ms})$，总体速率为 $10/4=2.5$ token/ms，逐周期速率平均却为 $(1+3)/2=2$ token/ms。

$K$ 只计最终提交的 token，不计草稿、拒绝候选或 padding。多个并行组要在统一观测窗口内累计产出和时间，不能把重叠周期的时长机械相加。

## 2. 接受长度与候选深度

在线性 speculative sampling 中，假设一次提出 $\gamma$ 个候选，$a_i$ 为此前候选都被接受时第 $i$ 个的条件接受概率。若拒绝时提交一个修正 token，全接受时提交一个额外 token，并忽略 EOS 与预算截断，则：

$$
A_\gamma=\mathbb E[K_\gamma]
=1+\sum_{i=1}^{\gamma}\prod_{j=1}^{i}a_j.
$$

这是把“至少多接受 $i$ 个候选”的概率相加。条件接受率必须按位置和真实负载估计；将首位置接受率重复用于所有深度可能高估收益。树形验证、不同提交规则、请求结束与状态回滚均需重新计数。[1]

设完整周期平均成本为 $C_\gamma$，包含草稿、验证、接受/拒绝、状态管理和暴露的调度开销。增加深度有利的条件为：

$$
\frac{A_{\gamma+1}}{C_{\gamma+1}}>\frac{A_\gamma}{C_\gamma}.
$$

若新增成本 $\Delta C>0$，等价于 $\Delta A/\Delta C>A_\gamma/C_\gamma$。它比较边际产出效率与已有平均效率；若成本下降或执行结构改变，直接比较两个完整比值更清楚。

## 3. 一个过度增加深度的反例

**合成例。** 普通单序列 Decode 为 10 ms/token。深度 3 的条件接受率为 $0.8,0.6,0.5$，完整周期耗时 16 ms：

$$
A_3=1+0.8+0.48+0.24=2.52,
\qquad S_3=\frac{10\times2.52}{16}=1.575.
$$

如果第四个候选的条件接受率为 0.5，则仅新增 $0.12$ 个期望提交；若周期因此从 16 增至 18 ms：

$$
A_4=2.64,\qquad S_4=10\times2.64/18\approx1.467.
$$

接受 token 数增加了，单位 token 成本反而上升。这个例子说明“更长的草稿”与“更大的实际收益”是两个问题。数值未模拟实际 kernel、特定模型或硬件。

## 4. Batch 改变比较分母

普通 batch 每步提交 $b$ 个 token、耗时 $T_0$ 时，基线是 $b/T_0$。若投机周期在所有请求上共提交 $K$ 个 token，则同一计量范围的吞吐比为：

$$
S_X=\frac{\mathbb E[K]/\mathbb E[T]}{b/T_0}.
$$

不能把单请求 $T_0\mathbb E[K]/\mathbb E[T]$ 套在整个 batch 上而漏掉 $b$。真实动态 batch 的基线也需按窗口累计。验证更多位置可能改变 GEMM 形状、MoE 专家并集、状态流量和容量；低并发时的收益不能直接迁移到高负载。

候选深度还可能影响返回节奏：一次提交更多 token 可以提高平均速率，却带来更长的 chunk 间隔。请求内 TPOT 是平均间隔，不能代表逐 token 抖动或 p99；服务端积攒输出的策略应保持可比。定义见[测量协议](../../docs/benchmark-protocol.md)，服务推导见[推理服务](../serving/README.md)。

## 5. 语义、正确性与可证伪预测

标准 speculative sampling 的分布保持依赖完整的接受、拒绝与修正采样规则。[1] 不能仅因草稿大部分被接受，就推断实现无损。数学分布一致、有限精度误差容差、固定随机数下逐字相同，是三个不同条件；有限测试通过也不是分布等价的证明。

**可证伪预测：** 在指定负载下，若独立估计得到 $A_\gamma T_0\le C_\gamma$，那么该单序列稳态成本模型不预测正加速。若实测仍加速，应检查基线是否改变、计时是否漏掉草稿或首 token、产出计数是否含拒绝候选，以及并行结构是否超出模型。

Agent 应把深度、验证形状与状态预算联合选择；每次保留接受长度分布和完整周期成本，而非只记一个平均接受率。原生 MTP 的可用性依赖模型与实现；不通过新增训练、更换权重、强行接受候选或缩短输出获得固定任务下的表面提升。

可计算情景见[性能模型](../../models/README.md)，候选决策见[自主优化](../../optimizer/README.md)，独立验证见[实验平台](https://github.com/Sunhuahai/llm-inference-theory/blob/1b4c5a37763475eb39b31dd083fcad256fc2578e/experiments/README.md)。

## 参考资料

1. Leviathan, Kalman, Matias, [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)，ICML 2023；核对日期 2026-11-25。用于分布保持的算法背景。
2. [统一理论总纲](../foundations/unified-theory.md)第 4 节；[收益预测](../optimization/benefit-prediction.md)第 6 节。本文非恒定接受率、成本边际比较和数字例按列明条件推导。
